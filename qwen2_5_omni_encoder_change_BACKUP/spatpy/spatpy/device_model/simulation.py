import numpy as np
from spatpy.geometry import Point, PointCloud
from spatpy.device_model import DeviceModel
from spatpy.placement import goldberg_distribution
import xarray as xr
from scipy.signal.windows import hann

import json
from spatpy.device_model.pml import FEMGridPML, VoxelGrid
from typing import Optional
from scipy.io import savemat, loadmat
import sys
import argparse
import json
from pathlib import Path
from typing import Dict
from spatpy.device_model.response import ImpulseResponse, SampledResponse
from dataclasses import dataclass


def simulate_near_field(
    spec: Dict,
    sources: Optional[PointCloud] = None,
    verbose=True,
    max_steps=None,
    trace_interval=None,
    plot_interval=None,
    spec_dir=None,
    resolution=1e-3,
):

    phone = DeviceModel.from_spec(
        spec,
        spec_dir=spec_dir,
    )
    if sources is None:
        sources = goldberg_distribution(3)

    body_voxels = VoxelGrid.from_device_model(phone, resolution=resolution)
    mic_points = phone.mics.locs.points
    irs = []
    mic_labels = []
    mic_pos = np.zeros((len(mic_points), 3))
    max_ir_len = 0
    MaxR = max(50, np.max(body_voxels.cells.shape) + 1)
    MicRadius = MaxR * 1.5
    sources = sources.expand_by(MicRadius * resolution)
    traces = []
    for (i, p) in enumerate(mic_points):
        MicLoc = body_voxels.xyz_indices(p.x, p.y, p.z)
        GridRadius = int(
            np.ceil(
                max(
                    np.max(sources.x),
                    np.max(sources.y),
                    np.max(sources.z),
                )
                / resolution
            )
        )
        ctr = GridRadius + 1
        print(
            f'mic {i+1} ("{p.tag}") requires receiver radius of'
            f" {MicRadius:.1f} units ({(MicRadius*resolution):.5f}m)"
        )
        print(
            f'mic {i+1} ("{p.tag}") requires grid size of'
            f" {GridRadius*2+1} units ({(GridRadius*2 + 1) * resolution:.5f}m)"
        )
        voxels = VoxelGrid(GridRadius, GridRadius, GridRadius, scale=resolution)
        nx, ny, nz = body_voxels.cells.shape
        mx = ctr - MicLoc[0]
        my = ctr - MicLoc[1]
        mz = ctr - MicLoc[2]
        voxels.cells.values[
            mx : (mx + nx), my : (my + ny), mz : (mz + nz)
        ] = body_voxels.cells.values
        # make sure we are not trying to say the mic is in a wall
        assert not voxels.cells.values[ctr, ctr, ctr]
        mic_pos[i, 0] = p.x
        mic_pos[i, 1] = p.y
        mic_pos[i, 2] = p.z
        pml = FEMGridPML(voxels=voxels, verbose=verbose)
        mic_name = p.tag
        mic_labels.append(mic_name)
        ir, fs = pml.run(
            in_locs=[[ctr, ctr, ctr]],
            out_locs=sources,
            increment=1,
            max_steps=max_steps,
            trace_interval=trace_interval,
            plot_interval=plot_interval,
            desc=f'Simulating mic {i + 1}/{len(mic_points)} ("{mic_name}")',
        )
        if trace_interval is not None:
            traces.append(pml.trace)
        irs.append(ir * np.atleast_2d(hann(ir.shape[0])).T)
        max_ir_len = max(max_ir_len, ir.shape[0])

    irs = [
        np.vstack((ir, np.zeros((max_ir_len - ir.shape[0], ir.shape[1]))))
        for ir in irs
    ]
    irs = np.stack(irs)
    # (mic, frame, source) -> (frame, source, mic)
    irs = irs.transpose((1, 2, 0))
    result = {
        "DeviceSpec": json.dumps(spec),
        "IRs": irs,
        "InSig": pml.input_sig,
        "FSample": pml.fs,
        "Locs": np.vstack((sources.x, sources.y, sources.z)),
        "MicPos": mic_pos,
        "MicLabel": mic_labels,
        "VOX_Scale": voxels.scale,
        "VOXGrid": pml.is_solid,
    }
    return result, traces


def make_parser():
    parser = argparse.ArgumentParser(
        description="FEM simulation of device impulse responses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "infile",
        help="Device specification .json file or .mat simulation result",
    )
    parser.add_argument(
        "--out-dir", help="Response output directory", default="."
    )
    parser.add_argument(
        "--max-steps", help="Maximum number of steps per microphone", type=int
    )
    parser.add_argument(
        "--resolution",
        help="Model device at this resolution (mm)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--plot-interval",
        help="If set, produce a live plot as the simulation runs",
        type=int,
    )
    parser.add_argument(
        "--trace-interval",
        help=(
            "If set, produce a visualisation showing the simulation where each"
            " frame corresponds to [trace-interval] steps of the model"
        ),
        type=int,
    )
    return parser


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = make_parser()
    options = parser.parse_args(args)

    out_path = Path("." if options.out_dir is None else options.out_dir)
    if options.infile.endswith(".json"):
        with open(options.infile, "rt") as fobj:
            spec = json.load(fobj)

        near_field_result, _ = simulate_near_field(
            spec,
            spec_dir=Path(options.infile).parent,
            max_steps=options.max_steps,
            trace_interval=options.trace_interval,
            plot_interval=options.plot_interval,
            resolution=options.resolution * 1e-3,
        )
        filename = out_path / (spec["name"] + "_FEM_IRs.mat")
        savemat(filename, near_field_result)
    else:
        assert options.infile.endswith(".mat")
        near_field_result = loadmat(options.infile, squeeze_me=True)
        spec = json.loads(near_field_result["DeviceSpec"])
        phone = DeviceModel.from_spec(spec)

    plot_filename = out_path / (spec["name"] + "_response.html")
    near_plot_filename = out_path / (spec["name"] + "_near_response.html")
    response = SampledResponse.from_near_field_simulation(
        near_field_result, near_plot_filename=near_plot_filename
    )
    response.plot_bands(
        body_trace=phone.body_trace,
        with_2d=False,
        phase_as_intensity=True,
        write_html=plot_filename,
    )
    return response


if __name__ == "__main__":
    # main(
    #     [
    #         "find_x5_pro_FEM_IRs.mat",
    #     ]
    # )
    main()
