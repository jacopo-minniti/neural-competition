#!/usr/bin/env python3
"""Run the first sweep: python scripts/run_experiments.py --output runs/first."""

import argparse
import signal
from pathlib import Path

from neural_competition.config import ExperimentConfig
from neural_competition.experiment import run_experiments


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/first.json")
    parser.add_argument("--output", type=Path, default=root / "runs/first")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed/diverged runs; restart incomplete runs from initialization",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny end-to-end check with the same measurements",
    )
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    config = ExperimentConfig.load(args.config)
    if args.smoke:
        config.widths = [config.teacher_width, config.teacher_width * 4]
        config.seeds = [0]
        config.steps = 8
        config.n_train, config.n_test = 128, 64
        config.batch_size, config.probe_batches, config.sharpness_batches = 16, 8, 2
        config.checkpoint_steps = [0, 1, 7]
        config.optimizers = [config.optimizers[0]] + [
            o for o in config.optimizers[1:] if o.name == "momentum"
        ][:1]
    result = run_experiments(config, args.output, args.device, resume=args.resume)
    print(f"Results: {result}", flush=True)


if __name__ == "__main__":
    main()
