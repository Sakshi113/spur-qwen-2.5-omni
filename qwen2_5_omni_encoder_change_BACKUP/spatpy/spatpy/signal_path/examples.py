import os
import sys

import torch
from torch import nn
import acid_bf

from ufb_banding.banding import BandingShape
from ufb_banding.banding.spatial import SpatialBandingParams
from ufb_banding.ufb import TransformParams
import spatpy.signal_path
from spatpy.signal_path.analysis import (
    ForwardBanding,
    ForwardTransform,
    PowerVector,
)
from spatpy.signal_path.primitives import apply_along_axis
from spatpy.signal_path.io import load_file, heatmap
from spatpy.signal_path.tracing import SignalPathInterpreter, SignalPathTracer

from dataclasses import dataclass, field
from typing import Optional


class ExampleChain(nn.Module):
    def __init__(
        self,
        sp: spatpy.signal_path.SignalPathConfig,
        banding=None,
        dt_ms=10.0,
        transform=None,
        fmin=50.0,
        fmax=8000.0,
        nband=25,
        band_shape=BandingShape.TRI,
    ):
        super().__init__()
        if banding is None:
            if transform is None:
                transform = TransformParams.RaisedSine()
            banding = SpatialBandingParams.Mel(
                dt_ms, fmin, fmax, nband, band_shape, transform
            )
        self.xfm = ForwardTransform(banding.transform_params, sigpath=sp)
        self.banding = ForwardBanding(banding, sigpath=sp)
        self.pv = PowerVector(banding, sigpath=sp)
        self.bands = None

    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        bins = self.xfm(pcm)
        self.bands = self.banding(bins)
        pv = self.pv(bins)
        return pv


def process_file(filename: str, max_frames=None, batch_mode=False) -> torch.Tensor:
    config, frames = load_file(filename, batch_mode=batch_mode)
    if max_frames is not None:
        frames = frames[:max_frames, ...]

    # chain = ExampleChain(config)
    # if not batch_mode:
    #     pv = apply_along_axis("frame", chain, frames)
    # else:
    #     pv = chain(frames)

    # bands = chain.bands
    # fig = heatmap(bands.align_to("ch", ...), db=True)
    # fig.show()

    config.batch_mode = False
    chain_to_trace = ExampleChain(config)
    gm, mm = SignalPathTracer.signal_path_to_mermaid(
        chain_to_trace, frames, show=True, filename="example_chain_diagram.html"
    )
    return pv, fig


if __name__ == "__main__":
    pv, fig = process_file("in.wav", batch_mode=True)
    fig.write_html("out.html")
