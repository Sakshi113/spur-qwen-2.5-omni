import numpy as np
import torch
from torch import nn

from spatpy.signal_path import SignalPathConfig
from spatpy.signal_path.primitives import HistoryBuffer

from ufb_banding.ufb import TransformParams, TransformCoefs
from ufb_banding.banding import BandingParams, BandingCoefs
from ufb_banding.backends.torch import dft_mod_matrix
from typing import Optional, Tuple, Union


class InverseTransform(nn.Module):
    def __init__(
        self,
        transform: Union[TransformParams, TransformCoefs],
        sigpath: Optional[SignalPathConfig] = None,
    ):
        super().__init__()
        if sigpath is None:
            coefs = transform
        else:
            coefs = TransformCoefs(params=transform, block_size=sigpath.block_size)
        self.mod_matrix = (
            dft_mod_matrix(coefs, synthesis=True)
            .to(dtype=torch.complex64)
            .refine_names("bin", "sample")
        )
        self.hist = HistoryBuffer(
            "sample",
            coefs.block_size,
            coefs.pad_syn_len,
            batch_mode=True if sigpath is None else sigpath.batch_mode,
        )
        self.coefs = coefs
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bins = x.align_to(..., "frame", "bin")
        names = bins.names
        block_size = self.coefs.block_size
        window_len = self.coefs.pad_syn_len
        nframe = bins.shape[-2]
        syn_nblocks = (self.coefs.pad_ana_len + block_size - 1) // block_size
        samples = torch.matmul(bins, self.mod_matrix).rename(None).real
        samples.names = names[:-1] + ("sample",)
        delay_line = samples.align_to(..., "sample", "frame")
        state_size = (syn_nblocks - 1) * block_size
        if self.state is not None:
            delay_line[..., :state_size, 0] += self.state.rename(None).flatten(-2, -1)
        names = delay_line.names
        pcm = (
            torch.nn.functional.fold(
                delay_line.rename(None),
                (block_size * (nframe + syn_nblocks - 1), 1),
                (window_len, 1),
                stride=(block_size, 1),
            )
            .squeeze(0)
            .squeeze(1)
            .squeeze(-1)
        )
        pcm = pcm.unflatten(-1, (nframe + syn_nblocks - 1, block_size))
        pcm.names = bins.names[:-2] + ("frame", "sample")
        self.state = pcm[:, -(syn_nblocks - 1) :, :]
        return pcm[:, : -(syn_nblocks - 1), :]

    @property
    def fbin(self):
        return torch.from_numpy(
            BandingCoefs.get_bin_freq(self.sp.fs, self.tfm_coefs.nb_modulations)
        ).float()

    @property
    def nbin(self):
        return self.tfm_coefs.nb_modulations


class BinGainCalculator(nn.Module):
    def __init__(
        self,
        banding: Union[BandingParams, BandingCoefs],
        sigpath: Optional[SignalPathConfig] = None,
        trainable: bool = False,
        dtype=torch.float32,
    ):
        super().__init__()
        if sigpath is None:
            coefs = banding
        else:
            coefs = BandingCoefs(banding, dt_ms=sigpath.block_size_ms, fs=sigpath.fs)
        band_matrix = (
            torch.from_numpy(coefs.band_matrix).to(dtype).refine_names("band", "bin")
        )
        self.coefs = coefs
        self.band_matrix = nn.Parameter(band_matrix) if trainable else band_matrix

    def forward(self, band_gains: torch.Tensor) -> torch.Tensor:
        band_gains = band_gains.align_to(..., "band")
        return torch.matmul(band_gains, self.band_matrix)


class MultichannelConvolver(nn.Module):
    def __init__(self, ir, axis=None, keep_frames=False, padding=None):
        super().__init__()
        if axis is None:
            axis = "sample"
        self.keep_frames = keep_frames
        self.axis = axis
        self.ir = (
            torch.from_numpy(ir).permute(1, 2, 0).refine_names("filter", "ch", axis)
        )
        self.n_in = self.ir.shape[-1]
        self.n_out = self.ir.shape[0]

    def forward(self, pcm):
        pcm = pcm.align_to(..., "ch", "frame", self.axis)
        names = pcm.names
        frame_size = pcm.shape[-1]
        nframe = pcm.shape[-2]
        other_sizes = pcm.shape[:-3]
        # concatenate along frame dimension
        pcm = pcm.rename(None).flatten(-2, -1)
        # create batch of size 1
        extra_dims = 3 - pcm.dim()
        pcm = pcm.reshape((1,) * extra_dims + pcm.shape)
        y = nn.functional.conv1d(pcm, self.ir.rename(None), padding="same")
        if self.keep_frames:
            y = y.squeeze(0).reshape(*other_sizes, self.n_out, nframe, frame_size)
            y.names = names[:-3] + ("ch", "frame", self.axis)
        else:
            y = y.squeeze(0)
            y.names = names[:-3] + ("ch", self.axis)
        return y
