#!/usr/bin/env python3
"""Change the rate at identical saved states in the oscillatory SGD pilot."""

import hashlib
import json
import signal
from pathlib import Path

import torch

from neural_competition.config import ExperimentConfig
from neural_competition.experiment import output_lock, run_one, save_torch, write_json


def main():
    root = Path(__file__).resolve().parents[2]
    config = ExperimentConfig.load(root / "configs/studies/pier_stability_continuation.json")
    output = root / "runs/pier_stability_continuation"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    data = torch.load(root / "runs/pier_stability_pilot/data.pt", weights_only=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    with output_lock(output):
        config_path = output / "config.json"
        if (
            config_path.exists()
            and ExperimentConfig.load(config_path).to_dict() != config.to_dict()
        ):
            raise ValueError("Continuation configuration differs from saved data")
        current_sources = []
        for seed in config.seeds:
            source = root / f"runs/pier_stability_pilot/w0016_seed{seed}_sgd_4/final.pt"
            current_sources.append(
                {
                    "seed": seed,
                    "path": str(source.relative_to(root)),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        sources_path = output / "sources.json"
        if sources_path.exists() and json.loads(sources_path.read_text()) != current_sources:
            raise ValueError(
                "Source states changed; preserve this study and use a new output directory"
            )
        write_json(config_path, config.to_dict())
        save_torch(output / "data.pt", data)
        sources = []
        manifest = {
            "status": "running",
            "total_runs": len(config.seeds) * len(config.optimizers),
            "runs": {},
        }
        for seed in config.seeds:
            source = root / f"runs/pier_stability_pilot/w0016_seed{seed}_sgd_4/final.pt"
            initial = torch.load(source, weights_only=True)["parameters"]
            sources.append(
                {
                    "seed": seed,
                    "path": str(source.relative_to(root)),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            for spec in config.optimizers:
                run_id = f"w0016_seed{seed}_{spec.label}"
                print(run_id, flush=True)
                state_path = output / run_id / "state.json"
                if state_path.exists():
                    previous = json.loads(state_path.read_text())
                    if previous["status"] in {"complete", "diverged"}:
                        manifest["runs"][run_id] = previous["status"]
                        continue
                state = run_one(
                    config,
                    spec,
                    16,
                    seed,
                    data,
                    output / run_id,
                    torch.device("cpu"),
                    initial_parameters=initial,
                    start_update=20000,
                )
                manifest["runs"][run_id] = state["status"]
                write_json(output / "manifest.json", manifest)
        write_json(output / "sources.json", sources)
        manifest["status"] = "complete"
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
