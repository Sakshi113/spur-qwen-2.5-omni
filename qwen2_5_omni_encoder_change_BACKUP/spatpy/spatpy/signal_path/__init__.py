from pathlib import Path

from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalPathConfig:
    fs: float = 48000.0
    block_size_ms: Optional[float] = None
    block_size: Optional[int] = None
    nmic: int = 1
    batch_mode: bool = True

    def __post_init__(self):
        if self.block_size is None:
            if self.block_size_ms is None:
                self.block_size_ms = 10.0
            self.block_size = int(self.fs / 1000.0 * self.block_size_ms)
        else:
            self.block_size_ms = float(self.block_size) / (self.fs / 1000.0)
