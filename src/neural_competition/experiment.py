"""A sequential sweep, with resumable runs and self-contained checkpoints."""

import csv
import json
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import ExperimentConfig, OptimizerConfig
from .diagnostics import collect_before, optimizer_buffers, probe_losses, summarize
from .model import ShallowReLU, loss, make_data
from .optimizers import MuonAdam


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@contextmanager
def output_lock(output):
    path = output / ".lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(
            f"Output is locked: {path}. Check its PID before removing a stale lock."
        ) from None
    with os.fdopen(fd, "w") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def choose_device(requested="auto"):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this Python process")
    return torch.device(requested)


def make_optimizer(model, spec: OptimizerConfig):
    if spec.name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=spec.lr)
    if spec.name == "momentum":
        return torch.optim.SGD(model.parameters(), lr=spec.lr, momentum=spec.momentum)
    if spec.name == "muon":
        return MuonAdam(model, spec)
    return torch.optim.Adam(model.parameters(), lr=spec.lr)


def append_row(writer, handle, row):
    writer.writerow(row)
    handle.flush()


def run_one(
    config, spec, width, seed, data, output, device, *, initial_parameters=None, start_update=0
):
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    model = ShallowReLU(
        config.input_dim, width, seed, zero_output=config.student_output_scale == 0
    ).to(device)
    if config.student_output_scale:
        with torch.no_grad():
            model.output.mul_(config.student_output_scale)
    if initial_parameters is not None:
        d = config.input_dim
        with torch.no_grad():
            model.weight.copy_(initial_parameters[:, :d])
            model.bias.copy_(initial_parameters[:, d])
            model.output.copy_(initial_parameters[:, d + 1])
    optimizer = make_optimizer(model, spec)
    train_generator = torch.Generator().manual_seed(config.data_seed + 10_000 + seed)
    # Every width and optimizer sees the same minibatch sequence within a seed.
    indices = torch.randint(
        config.n_train, (config.steps + start_update, config.batch_size), generator=train_generator
    )[start_update:].to(device)
    probe_indices = data["probe_indices"].to(device)
    x_probe = data["x_train"][probe_indices]
    y_probe = data["y_train"][probe_indices]
    previous_step = torch.zeros_like(model.parameter_blocks())
    checkpoint_steps = set(config.checkpoint_steps) | {0, config.steps - 1}
    pending_losses = set(config.loss_checkpoints)
    started = time.perf_counter()
    state = {
        "status": "running",
        "width": width,
        "seed": seed,
        "optimizer": spec.__dict__,
        "device": str(device),
        "started_at": now(),
        "updates_completed": 0,
        "initial_updates": start_update,
    }
    write_json(output / "state.json", state)

    try:
        with (
            (output / "training.csv").open("w", newline="") as training_handle,
            (output / "metrics.csv").open("w", newline="") as metrics_handle,
        ):
            training_writer = csv.DictWriter(
                training_handle,
                fieldnames=["step", "batch_loss_before", "step_norm", "elapsed_seconds"],
            )
            training_writer.writeheader()
            metrics_writer = None
            for step in range(config.steps):
                optimizer.zero_grad(set_to_none=True)
                x = data["x_train"][indices[step]]
                y = data["y_train"][indices[step]]
                value = loss(model, x, y)
                crossed_losses = []
                if pending_losses and step % config.loss_check_every == 0:
                    with torch.no_grad():
                        current_mse = 2 * loss(model, data["x_train"], data["y_train"]).item()
                    crossed_losses = sorted(v for v in pending_losses if current_mse <= v)
                    pending_losses.difference_update(crossed_losses)
                checkpoint = step in checkpoint_steps or bool(crossed_losses)
                log = checkpoint or step % config.log_every == 0
                if log and not torch.isfinite(value).item():
                    state.update(status="diverged", updates_completed=step, ended_at=now())
                    write_json(output / "state.json", state)
                    print(f"  diverged at step {step}", flush=True)
                    return state

                snapshot = (
                    collect_before(
                        model, optimizer, data, x_probe, y_probe, config.sharpness_batches
                    )
                    if checkpoint
                    else None
                )
                parameters_before = model.parameter_blocks()
                value.backward()
                if checkpoint:
                    snapshot["sampled_gradient"] = (
                        model.blocks(tuple(p.grad for p in model.parameters())).detach().cpu()
                    )
                optimizer.step()
                actual_step = model.parameter_blocks() - parameters_before

                if checkpoint:
                    with torch.no_grad():
                        train_after = loss(model, data["x_train"], data["y_train"])
                        test_after = loss(model, data["x_test"], data["y_test"])
                    snapshot.update(
                        {
                            "format_version": 2,
                            "loss_targets_crossed": crossed_losses,
                            "step": step,
                            "updates_completed": step + 1,
                            "width": width,
                            "seed": seed,
                            "optimizer": spec.__dict__,
                            "parameters_after": model.parameter_blocks().cpu(),
                            "actual_step": actual_step.cpu(),
                            "previous_step": previous_step.cpu(),
                            "optimizer_after": optimizer_buffers(model, optimizer),
                            "probe_losses_after": probe_losses(model, x_probe, y_probe).cpu(),
                            "train_loss_after": float(train_after.cpu()),
                            "test_loss_after": float(test_after.cpu()),
                        }
                    )
                    metrics = {
                        "step": step,
                        "updates_completed": step + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        **summarize(snapshot),
                    }
                    save_torch(checkpoints / f"step_{step:06d}.pt", snapshot)
                    if metrics_writer is None:
                        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=list(metrics))
                        metrics_writer.writeheader()
                    append_row(metrics_writer, metrics_handle, metrics)
                    print(
                        f"  step {step:5d}  train={metrics['train_loss']:.5g}  test={metrics['test_loss']:.5g}"
                        f"  conflict={metrics['pair_cosine_negative_fraction']:.3f}"
                        f"  sharpness={metrics['batch_sharpness']:.4g}",
                        flush=True,
                    )

                if log:
                    append_row(
                        training_writer,
                        training_handle,
                        {
                            "step": step,
                            "batch_loss_before": float(value.detach().cpu()),
                            "step_norm": float(actual_step.norm().cpu()),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                    )
                    state.update(
                        updates_completed=step + 1,
                        last_checkpoint=step if checkpoint else state.get("last_checkpoint"),
                    )
                    write_json(output / "state.json", state)
                previous_step = actual_step

        with torch.no_grad():
            state.update(
                status="complete",
                updates_completed=config.steps,
                ended_at=now(),
                elapsed_seconds=time.perf_counter() - started,
                final_train_loss=float(loss(model, data["x_train"], data["y_train"]).cpu()),
                final_test_loss=float(loss(model, data["x_test"], data["y_test"]).cpu()),
            )
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            state["status"] = "diverged"
        save_torch(
            output / "final.pt",
            {
                "parameters": model.parameter_blocks().cpu(),
                "optimizer_state": optimizer.state_dict(),
                "state": state,
            },
        )
        write_json(output / "state.json", state)
        return state
    except BaseException as error:
        state.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
            ended_at=now(),
        )
        write_json(output / "state.json", state)
        raise


def run_experiments(config: ExperimentConfig, output, device="auto", *, resume=False):
    config.validate()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"Output already contains files: {output}. Use --resume or another path."
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    device = choose_device(device)
    with output_lock(output):
        config_path = output / "config.json"
        if (
            config_path.exists()
            and ExperimentConfig.load(config_path).to_dict() != config.to_dict()
        ):
            raise ValueError("Saved configuration differs; use a new output directory")
        write_json(config_path, config.to_dict())
        write_json(
            output / "environment.json",
            {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "device": str(device),
                "cpu_threads": torch.get_num_threads(),
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "dtype": "float32",
                "sharpness_device": "cpu",
                "sharpness_dtype": "float64",
            },
        )
        if not (output / "data.pt").exists():
            save_torch(output / "data.pt", make_data(config))
        cpu_data = torch.load(output / "data.pt", map_location="cpu", weights_only=True)
        data = {
            key: value.to(device) if key in {"x_train", "y_train", "x_test", "y_test"} else value
            for key, value in cpu_data.items()
        }
        jobs = [
            (width, seed, spec)
            for width in config.widths
            for seed in config.seeds
            for spec in config.optimizers
        ]
        manifest = {"status": "running", "started_at": now(), "total_runs": len(jobs), "runs": {}}
        write_json(output / "manifest.json", manifest)
        try:
            for number, (width, seed, spec) in enumerate(jobs, start=1):
                run_id = f"w{width:04d}_seed{seed}_{spec.label}"
                run_output = output / run_id
                state_path = run_output / "state.json"
                if resume and state_path.exists():
                    state = json.loads(state_path.read_text())
                    if state["status"] in {"complete", "diverged"}:
                        manifest["runs"][run_id] = state["status"]
                        write_json(output / "manifest.json", manifest)
                        continue
                print(f"[{number}/{len(jobs)}] {run_id}  device={device}  lr={spec.lr}", flush=True)
                state = run_one(config, spec, width, seed, data, run_output, device)
                manifest["runs"][run_id] = state["status"]
                write_json(output / "manifest.json", manifest)
            manifest.update(status="complete", ended_at=now())
        except BaseException as error:
            manifest.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error),
                ended_at=now(),
            )
            raise
        finally:
            write_json(output / "manifest.json", manifest)
    return output
