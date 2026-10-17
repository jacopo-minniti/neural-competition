"""The complete experimental protocol is a small, saved JSON configuration."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    label: str
    momentum: float = 0.9

    def validate(self) -> None:
        if self.name not in {"sgd", "momentum", "adam"}:
            raise ValueError(f"Unknown optimizer: {self.name}")
        if self.lr <= 0 or not 0 <= self.momentum < 1:
            raise ValueError("Invalid learning rate or momentum")
        if not self.label or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.label
        ):
            raise ValueError("Optimizer labels must contain only a-z, 0-9, '_' or '-'")


@dataclass
class ExperimentConfig:
    input_dim: int = 20
    teacher_width: int = 8
    n_train: int = 8192
    n_test: int = 4096
    data_seed: int = 2026
    widths: list[int] = field(default_factory=lambda: [8, 9, 10, 16, 128])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 3])
    steps: int = 100000
    batch_size: int = 64
    probe_batches: int = 64
    checkpoint_steps: list[int] = field(
        default_factory=lambda: [
            0,
            1,
            20,
            100,
            700,
            1999,
            4999,
            9999,
            19999,
            39999,
            59999,
            79999,
            99999,
        ]
    )
    sharpness_batches: int = 4
    log_every: int = 1000
    cpu_threads: int = 4
    optimizers: list[OptimizerConfig] = field(
        default_factory=lambda: [OptimizerConfig("sgd", 1.0, "sgd_large")]
    )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text())
        if "optimizers" in raw:
            raw["optimizers"] = [OptimizerConfig(**item) for item in raw["optimizers"]]
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        positive = [
            self.input_dim,
            self.teacher_width,
            self.n_train,
            self.n_test,
            self.steps,
            self.batch_size,
            self.probe_batches,
            self.log_every,
            self.cpu_threads,
        ]
        if min(positive) < 1 or self.probe_batches < 2:
            raise ValueError("Sizes must be positive; at least two probe batches are needed")
        if not self.widths or min(self.widths) < self.teacher_width:
            raise ValueError("Each student must be at least as wide as the teacher")
        if not self.seeds or min(self.seeds) < 0:
            raise ValueError("Provide nonnegative initialization seeds")
        if len(set(self.widths)) != len(self.widths) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Widths and seeds must be unique")
        if not 0 <= self.sharpness_batches <= self.probe_batches:
            raise ValueError("Invalid number of sharpness probe batches")
        if not self.optimizers or len({o.label for o in self.optimizers}) != len(self.optimizers):
            raise ValueError("Provide optimizers with unique labels")
        if any(s < 0 or s >= self.steps for s in self.checkpoint_steps):
            raise ValueError("Checkpoint steps must index an update in [0, steps)")
        for optimizer in self.optimizers:
            optimizer.validate()

    def to_dict(self) -> dict:
        return asdict(self)
