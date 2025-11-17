import torch
from torch import nn
from typing import Optional
import torch.fx
import numpy as np


class Reblocker(nn.Module):
    def __init__(
        self,
        block_size: int,
        reblock_dim: Optional[str] = None,
        result_dim: Optional[str] = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.reblock_dim = "sample" if reblock_dim is None else reblock_dim
        self.result_dim = "frame" if result_dim is None else result_dim

    def forward(self, x):
        x = x.align_to(self.result_dim, self.reblock_dim, ...)
        sz = self.block_size
        nblock = (x.shape[0] * x.shape[1]) // self.block_size
        names = x.names
        remaining_dims = x.shape[2:]
        x = x.rename(None)
        x = torch.flatten(x, 0, 1)
        blocks = x[: nblock * sz, ...].reshape(nblock, sz, *remaining_dims)
        blocks.names = names
        return blocks


@torch.fx.wrap
def apply_along_axis(func1d, axis: str, x: torch.Tensor, *args) -> torch.Tensor:
    y = []
    x = x.align_to(axis, ...)
    vs = torch.split(x, 1, 0)
    for (i, v) in enumerate(vs):
        yi = func1d(v, *args)
        names = []
        for (j, n) in enumerate(yi.names):
            if n == axis:
                yi = yi.squeeze(j)
            else:
                names.append(n)
        y.append(yi.rename(None))
    y = torch.stack(y)
    y.names = (axis,) + tuple(names)
    return y


@torch.fx.wrap
def forward_single_frame(
    axis: str,
    state: torch.Tensor,
    frame_size: int,
    history_length: int,
    x: torch.Tensor,
) -> torch.Tensor:
    x = x.align_to(axis, "frame", ...)

    if state is None:
        state = torch.zeros((history_length,) + x.shape[2:])
        state.names = (axis,) + x.names[2:]

    state = state[frame_size:, ...]
    state = torch.cat((state, x.rename(None)[:, -1, ...]), 0)
    return state.align_to(axis, "frame", ...)


class HistoryBuffer(nn.Module):
    def __init__(
        self,
        axis: str,
        frame_size: int,
        history_length: int,
        conv: bool = True,
        batch_mode: bool = False,
    ):
        super().__init__()
        self.axis = axis
        self.frame_size = frame_size
        self.history_length = history_length
        self.state = None
        self.batch_mode = batch_mode

    def forward_batch(self, x):
        orig = x
        x = x.align_to(..., "frame", self.axis)
        names = x.names
        x = x.rename(None).flatten(-2, -1).unsqueeze(-1)

        state_frames = (
            (self.history_length + self.frame_size - 1) // self.frame_size
        ) - 1
        state_size = state_frames * self.frame_size

        if self.state is None:
            self.state = x.new_zeros(x.shape[:-2] + (state_size, x.shape[-1]))
            x = torch.cat((self.state, x), -2)
        else:
            x[..., :state_size, :] += self.state
        y = x.unfold(-2, self.history_length, self.frame_size).squeeze(dim=-2).transpose(-1, -2)
        self.state = y[..., -state_frames:].flatten(-2, -1).unsqueeze(-1)
        y.names = names[:-2] + (self.axis, "frame")
        return y.align_as(orig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.batch_mode:
            self.state = forward_single_frame(
                self.axis, self.state, self.frame_size, self.history_length, x
            )
            return self.state.clone()
        return self.forward_batch(x)


class IIRSmoother(nn.Module):
    def __init__(self, state=None):
        super().__init__()
        self.state = state

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            self.state = x
        
        alpha = alpha.refine_names("band")
        y = self.state * alpha.align_as(self.state)
        y += x * (1.0 - alpha.align_as(x))
        self.state = y.clone()
        return y.to(dtype=x.dtype)


class MinimumFollower(torch.nn.Module):
    def __init__(self, alpha, state=None):
        super().__init__()
        self.alpha = alpha
        self.state = state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            self.state = x

        self.state *= 1.0 + self.alpha.align_as(self.state)
        self.state = torch.min(self.state.rename(None), x.rename(None)).refine_names(
            *self.state.names
        )
        return self.state.clone().to(dtype=x.dtype)


class InvertibleDCT(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, name=None) -> torch.Tensor:
        """Invertible DCT which operates along the last dimension of x"""
        orig_names = x.names
        if name is None:
            name = x.names[-1]
        N = x.shape[-1]
        n = x.new_ones(N).cumsum(0) - 1
        k = (x.new_ones(N).cumsum(0) - 1).unsqueeze(1)
        y = torch.matmul(x.rename(None), torch.cos(np.pi / N * (n + 0.5) * (k + 0.5)))
        y.names = orig_names[:-1] + (name,)
        return y * ((2 / N) ** 0.5)
