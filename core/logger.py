from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


class ExperimentLogger:
    def __init__(self, log_dir: str | Path, console_interval: int = 1000) -> None:
        self.log_dir = Path(log_dir)
        self.console_interval = max(1, int(console_interval))
        self.writer = SummaryWriter(str(self.log_dir)) if SummaryWriter else None

    def scalar(self, name: str, value: float | int | None, step: int) -> None:
        if value is None:
            return
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def scalars(self, values: dict[str, Any], step: int) -> None:
        for name, value in values.items():
            if isinstance(value, (int, float)):
                self.scalar(name, value, step)

    def maybe_console(self, step: int, total_steps: int, values: dict[str, Any]) -> None:
        if step % self.console_interval != 0 and step != total_steps:
            return
        compact = []
        for key, value in values.items():
            if isinstance(value, float):
                compact.append(f"{key}={value:.3f}")
            elif value is not None:
                compact.append(f"{key}={value}")
        print(f"[step {step}/{total_steps}] " + " ".join(compact), flush=True)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
