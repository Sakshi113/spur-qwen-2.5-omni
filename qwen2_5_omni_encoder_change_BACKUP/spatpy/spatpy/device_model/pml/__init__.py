from dataclasses import dataclass, field
from types import SimpleNamespace
import numpy as np
from typing import Optional
from scipy.signal import cheb2ord, cheby2, lfilter
import time
import platform
import plotly.graph_objects as go

import xarray as xr
from pathlib import Path

from spatpy.plot import VolumeSlider
from spatpy.geometry import PointCloud, Point
from spatpy.device_model.grid import VoxelGrid
import numpy as np
from ctypes import c_float, c_uint64, c_uint8, POINTER, CDLL
from pathlib import Path
import plotly.express as px
from tqdm import tqdm
import gc
from spatpy.device_model.pml.cuda import FEMGridPMLCUDA

# import faulthandler
# faulthandler.enable()
c_float_1d = POINTER(c_float)
c_uint8_1d = POINTER(c_uint8)
c_uint64_1d = POINTER(c_uint64)


def p_float(x):
    return x.ctypes.data_as(c_float_1d)


def p_uint8(x):
    return x.ctypes.data_as(c_uint8_1d)


def p_uint64(x):
    return x.ctypes.data_as(c_uint64_1d)


dll_ext = (
    ".dylib"
    if platform.system() == "Darwin"
    else (".so" if platform.system() == "Linux" else ".dll")
)
libmal = CDLL(
    Path(__file__).parent.parent.parent
    / "data"
    / "libmalfeasance"
    / platform.system()
    / platform.machine()
    / f"libmalfeasance{dll_ext}"
)
libmal.fem_grid_pml_step.restype = None
libmal.fem_grid_pml_step.argtypes = [
    c_float_1d,  # float *outSig,
    c_float_1d,  # float *G,
    c_float,  # float stepRate,
    c_float,  # float stepAlpha,
    c_uint64_1d,  # uint64_t *wallInds,
    c_uint64_1d,  # uint64_t *inInds,
    c_float_1d,  # float *inSig,
    c_uint8_1d,  # uint8_t *wallMask,
    c_float_1d,  # float *Qx2,
    c_float_1d,  # float *Qy2,
    c_float_1d,  # float *Qz2,
    c_float_1d,  # float *PMLCoefs,
    c_uint64_1d,  # uint64_t *outInds,
    c_uint8_1d,  # uint8_t *PML_en,
    c_uint64_1d,  # uint64_t *pLossInds,
    c_uint64,  # uint64_t nX,
    c_uint64,  # uint64_t nY,
    c_uint64,  # uint64_t nZ,
    c_uint64,  # uint64_t nWalls,
    c_uint64,  # uint64_t nIn,
    c_uint64,  # uint64_t nOut,
    c_uint64,  # uint64_t nPML,
    c_uint64,  # uint64_t nSamp,
    c_uint64,  # uint64_t nPLoss
]


class FEMGridPMLCBackend:
    def __init__(
        self,
        order,
        g,
        step_rate,
        step_alpha,
        wall_inds,
        in_inds,
        wall_mask,
        Qx2,
        Qy2,
        Qz2,
        pml_coefs,
        out_inds,
        pml_en,
        p_loss_inds,
        matlab_binary_dir=None,
    ):
        self.step_rate = step_rate
        self.step_alpha = step_alpha
        self.nx, self.ny, self.nz = g.shape[1:]
        self.nwall = wall_inds.shape[0]
        self.nin = in_inds.shape[0]
        self.nout = out_inds.shape[0]
        self.npml = pml_coefs.shape[1]
        self.nploss = p_loss_inds.shape[0]

        if matlab_binary_dir is not None:
            p = Path(matlab_binary_dir)
            matlab_sizes = np.fromfile(p / "sizes.bin", np.uint64)
            # matlab_in_sig = np.fromfile(p / "inSig.bin", np.float32).reshape(
            #     in_sig.shape, order=order
            # )
            matlab_g = np.fromfile(p / "G.bin", np.float32).reshape(
                g.shape, order=order
            )
            matlab_wall_inds = np.fromfile(
                p / "wallInds.bin", np.uint64
            ).reshape(wall_inds.shape, order=order)
            matlab_in_inds = np.fromfile(p / "inInds.bin", np.uint64).reshape(
                in_inds.shape, order=order
            )
            matlab_out_inds = np.fromfile(p / "outInds.bin", np.uint64).reshape(
                out_inds.shape, order=order
            )
            matlab_p_loss_inds = np.fromfile(
                p / "pLossInds.bin", np.uint64
            ).reshape(p_loss_inds.shape, order=order)
            matlab_Qx2 = np.fromfile(p / "Qx2.bin", np.float32).reshape(
                Qx2.shape, order=order
            )
            matlab_Qy2 = np.fromfile(p / "Qy2.bin", np.float32).reshape(
                Qy2.shape, order=order
            )
            matlab_Qz2 = np.fromfile(p / "Qz2.bin", np.float32).reshape(
                Qz2.shape, order=order
            )
            matlab_pml_coefs = np.fromfile(
                p / "PMLCoefs.bin", np.float32
            ).reshape(pml_coefs.shape, order=order)
            matlab_wall_mask = np.fromfile(
                p / "wallMask.bin", np.uint8
            ).reshape(wall_mask.shape, order=order)
            matlab_pml_en = np.fromfile(p / "PML_en.bin", np.uint8).reshape(
                pml_en.shape, order=order
            )

        self.wall_inds = wall_inds + 1
        self.in_inds = in_inds + 1
        self.out_inds = out_inds + 1
        self.wall_mask = wall_mask.ravel(order=order)
        self.Qx2 = Qx2.ravel(order=order)
        self.Qy2 = Qy2.ravel(order=order)
        self.Qz2 = Qz2.ravel(order=order)
        self.pml_coefs = pml_coefs.ravel(order=order)
        self.pml_en = pml_en.ravel(order=order)
        self.p_loss_inds = p_loss_inds.ravel(order=order)
        self.g = g.ravel(order=order)
        self.order = order

    def step(self, in_sig):
        in_sig = in_sig.ravel(order=self.order)
        nsamp = in_sig.shape[0]
        out_sig = np.zeros(nsamp * self.nout, np.float32, order=self.order)
        libmal.fem_grid_pml_step(
            p_float(out_sig),
            p_float(self.g),
            self.step_rate,
            self.step_alpha,
            p_uint64(self.wall_inds),
            p_uint64(self.in_inds),
            p_float(in_sig),
            p_uint8(self.wall_mask),
            p_float(self.Qx2),
            p_float(self.Qy2),
            p_float(self.Qz2),
            p_float(self.pml_coefs),
            p_uint64(self.out_inds),
            p_uint8(self.pml_en),
            p_uint64(self.p_loss_inds),
            self.nx - 1,
            self.ny - 1,
            self.nz - 1,
            self.nwall,
            self.nin,
            self.nout,
            self.npml,
            nsamp,
            self.nploss,
        )
        return out_sig.reshape(nsamp, self.nout)


# Stolen from //depot/rd/r/MatlabCommon/FEM_Grid_PML.m

# The methods used in this class (for implementing the PML) are derived
# from this paper:
#
# "A Reflectionless Discrete Perfectly Matched Layer"
# Albert Chern
# Journal of Computational Physics 381 (2019): 91-109
@dataclass
class FEMGridPML:
    voxels: VoxelGrid
    output_gain: float = 1.0
    cell_size_m: float = 1e-3
    c_m_per_s: float = 343.3
    pml_enable_xyzse: np.ndarray = field(
        default_factory=lambda: np.ones((3, 2), dtype=np.uint8)
    )
    p_absorber_xyz: Optional[np.ndarray] = None

    pml_width: int = 32
    max_sigma: float = 0.15
    step_rate: float = 0.5
    trace_interval: Optional[int] = None
    plot_interval: Optional[int] = None
    verbose: bool = True
    use_torch: Optional[bool] = None

    # After some imperical experiments, a StepAlpha value of about 0.08
    # proved to be optimal. We use a slightly different value, that just
    # happen to force StepRate := 0.5
    step_alpha: float = (
        2 / np.sqrt(3) - 1
    ) / 2  # (about 0.774) fudge factor to correct the speed of sound at HF
    max_w: float = 0  # 0.2; PML max value of PML modification
    fs: Optional[float] = None

    def reset_simulation_state(
        self,
        ir_tail_countdown_steps: int = 150,
        trace_interval: Optional[int] = None,
        plot_interval: Optional[int] = None,
        input_sig: Optional[np.ndarray] = None,
        in_locs=None,
        out_locs=None,
    ):
        # We figure out when the SIM is done by tracking the peak output and
        # waiting for the tail of the IR to fall far enough below the peak
        self.trace = None
        self.trace_interval = trace_interval
        self.xyz_slices = None
        self.cbar_ax = None
        self.plot_interval = plot_interval
        self.input_sig = input_sig
        self.sample_num = 0
        self.ir_tail_countdown_steps = ir_tail_countdown_steps
        self.max_steps_to_go = ir_tail_countdown_steps
        self.peak_ir_level = 0
        self.peak_ir_sample_num = 0
        self.input_inds = None
        self.input_state = None

        self.grid = np.zeros(
            (4, self.sizeX + 1, self.sizeY + 1, self.sizeZ + 1),
            dtype=np.float32,
            order=self.order,
        )
        rng = np.random.default_rng()
        rng.random(out=self.grid, dtype=np.float32)
        self.grid *= 1e-21

        sigma = (
            np.sin(
                np.pi
                / 2
                * np.hstack(([0, 0.2], np.arange(1, 2 * self.pml_width - 1)))
                / (2 * self.pml_width - 1)
            )
            ** 2
        ).astype(np.float32, order=self.order)
        sigma = sigma / sigma[-1] * self.max_sigma

        make_xyzs = lambda a: SimpleNamespace(
            xp=a[:, None, None],
            xm=np.flipud(a)[:, None, None],
            yp=a[None, :, None],
            ym=np.flipud(a)[None, :, None],
            zp=a[None, None, :],
            zm=np.flipud(a)[None, None, :],
        )

        self.pml_sigmaPd = make_xyzs(-sigma[1::2])
        self.pml_sigmaVd = make_xyzs(np.exp(-sigma[::2]))
        self.pml_sigmaVi = make_xyzs(
            (1 - np.exp(-sigma[::2])) * self.step_rate / 2
        )
        self.pml_sigmaQd = make_xyzs(np.exp(-sigma[1::2]))
        self.pml_sigmaQi = make_xyzs(
            -self.step_rate * np.exp(-sigma[1::2] / 2) / 2
        )

        self.pml_coefs = (
            np.stack(
                (
                    self.pml_sigmaPd.xm.T,
                    self.pml_sigmaVd.xm.T,
                    self.pml_sigmaVi.xm.T,
                    self.pml_sigmaQd.xm.T,
                    self.pml_sigmaQi.xm.T,
                )
            )
            .squeeze()
            .astype(np.float32, order=self.order)
        )

        pw = self.pml_width
        self.pml_Qx = np.zeros(
            (2, pw, self.sizeY, self.sizeZ), np.float32, order=self.order
        )  # [ 2 x pw x YS x ZS ]
        self.pml_Qy = np.zeros(
            (2, self.sizeX, pw, self.sizeZ), np.float32, order=self.order
        )  # [ 2 x XS x pw x ZS ]
        self.pml_Qz = np.zeros(
            (2, self.sizeX, self.sizeY, pw), np.float32, order=self.order
        )  # [ 2 x XS x YS x pw ]

        self.Qx = np.moveaxis(self.pml_Qx, 0, -1)
        self.Qx[:, :, :, 1] = np.flip(self.Qx[:, :, :, 1], 0)
        self.Qy = np.moveaxis(self.pml_Qy, 0, -1)
        self.Qy[:, :, :, 1] = np.flip(self.Qy[:, :, :, 1], 1)
        self.Qz = np.moveaxis(self.pml_Qz, 0, -1)
        self.Qz[:, :, :, 1] = np.flip(self.Qz[:, :, :, 1], 2)

        self.Qx = self.Qx.astype(np.float32, order=self.order)
        self.Qy = self.Qy.astype(np.float32, order=self.order)
        self.Qz = self.Qz.astype(np.float32, order=self.order)

        if in_locs is not None:
            if isinstance(in_locs, Point):
                mic_points = PointCloud([in_locs])
                in_locs = self.voxels.point_indices(mic_points).T
            _, input_inds, input_state = self.signal_vars(in_locs)
            self.input_inds = input_inds
            self.input_state = input_state

        self.output_gain = 1.0
        self.output_inds = None
        self.output_sig = None
        if out_locs is not None:
            if isinstance(out_locs, PointCloud):
                out_locs = self.voxels.point_indices(out_locs).T
            output_gain, output_inds, _ = self.signal_vars(out_locs)
            self.output_inds = output_inds
            self.output_gain = output_gain
            self.output_sig = np.zeros((128, len(out_locs)), order=self.order)

        if self.input_inds is not None:
            backend_cls = FEMGridPMLCUDA if self.use_torch else FEMGridPMLCBackend
            self.backend = backend_cls(
                self.order,
                self.grid,
                self.step_rate,
                self.step_alpha,
                self.wall_inds,
                self.input_inds,
                self.walls_int,
                self.Qx,
                self.Qy,
                self.Qz,
                self.pml_coefs,
                self.output_inds,
                self.pml_enable_xyzse,
                self.p_loss_inds,
            )

    def __post_init__(self):
        should_use_torch = False
        try:
            import torch
            should_use_torch = torch.cuda.is_available()
        except:
            pass
        if self.use_torch is None:
            self.use_torch = should_use_torch
        self.order = 'C' if self.use_torch else 'F'
        self.is_solid = self.voxels.cells.values
        c = self.c_m_per_s
        d = self.cell_size_m
        if self.fs is None:
            self.fs = c / d / self.step_rate
        # The PML will make our grid larger
        pw = self.pml_width
        self.pwx = pw * self.pml_enable_xyzse[0, 0]
        self.pwy = pw * self.pml_enable_xyzse[1, 0]
        self.pwz = pw * self.pml_enable_xyzse[2, 0]
        pwX = pw * self.pml_enable_xyzse[0, 1]
        pwY = pw * self.pml_enable_xyzse[1, 1]
        pwZ = pw * self.pml_enable_xyzse[2, 1]

        pwx, pwy, pwz = self.pwx, self.pwy, self.pwz
        sX, sY, sZ = self.is_solid.shape
        self.sizeX = sX + pwx + pwX
        self.sizeY = sY + pwy + pwY
        self.sizeZ = sZ + pwz + pwZ

        pwx, pwy, pwz = self.pwx, self.pwy, self.pwz

        nx = pwx + sX
        ny = pwy + sY
        nz = pwz + sZ
        tmp_wall = np.zeros(
            (4, self.sizeX + 1, self.sizeY + 1, self.sizeZ + 1),
            dtype=np.uint8,
            order=self.order,
        )
        tmp_wall[1, 0, :, :] = 1
        tmp_wall[1, -1, :, :] = 1

        tmp_x = tmp_wall[1, (pwx + 1) : nx, pwy:ny, pwz:nz]
        xdiff = np.diff(self.is_solid, axis=0)
        tmp_x[xdiff] = 1
        tmp_wall[2, :, 0, :] = 1
        tmp_wall[2, :, -1, :] = 1
        tmp_y = tmp_wall[2, pwx:nx, (pwy + 1) : ny, pwz:nz]
        ydiff = np.diff(self.is_solid, axis=1)
        tmp_y[ydiff] = 1
        tmp_wall[3, :, :, 0] = 1
        tmp_wall[3, :, :, -1] = 1
        tmp_z = tmp_wall[3, pwx:nx, pwy:ny, (pwz + 1) : nz]
        zdiff = np.diff(self.is_solid, axis=2)
        tmp_z[zdiff] = 1

        # For the faces [of the grid] where we have a PML, we extend the
        # walls [those walls at are perpendicular to the face] to the
        # boundary of the PML:
        tmp_wall[2:4, :pwx, :, :] = np.tile(
            tmp_wall[2:4, pwx : (pwx + 1), :, :], (1, pwx, 1, 1)
        )
        tmp_wall[2:4, (-pwX - 1) : -1, :, :] = np.tile(
            tmp_wall[2:4, (-pwX - 2) : (-pwX - 1), :, :], (1, pwX, 1, 1)
        )
        tmp_wall[1, :, :pwy, :] = np.tile(
            tmp_wall[1, :, pwy : (pwy + 1), :], (1, 1, pwy, 1)
        )
        tmp_wall[3, :, :pwy, :] = np.tile(
            tmp_wall[3, :, pwy : (pwy + 1), :], (1, 1, pwy, 1)
        )
        tmp_wall[1, :, (-pwY - 1) : -1, :] = np.tile(
            tmp_wall[1, :, (-pwY - 2) : (-pwY - 1), :], (1, 1, pwY, 1)
        )
        tmp_wall[3, :, (-pwY - 1) : -1, :] = np.tile(
            tmp_wall[3, :, (-pwY - 2) : (-pwY - 1), :], (1, 1, pwY, 1)
        )
        tmp_wall[1:3, :, :, :pwz] = np.tile(
            tmp_wall[1:3, :, :, pwz : (pwz + 1)], (1, 1, 1, pwz)
        )
        tmp_wall[1:3, :, :, (-pwZ - 1) : -1] = np.tile(
            tmp_wall[1:3, :, :, (-pwZ - 2) : (-pwZ - 1)], (1, 1, 1, pwZ)
        )
        self.wall_inds = (
            np.argwhere(tmp_wall.ravel(order=self.order))
            .astype(np.uint64)
            .ravel(order=self.order)
        )

        # p_absMask = np.zeros(tmp_wall.shape, dtype=np.uint8, order=self.order)
        # Now - do we have any P locations where we want to simulate an
        # absorbing adjacent wall?
        # for k = 1 : size(this.P_AbsorberXYZ,2)
        #     p = this.P_AbsorberXYZ(:,k);
        #     assert( all(round(p)==p) && all(p>=1) && all(p<=[sX;sY;sZ]), ...
        #     "Pressure loss points lie outside the grid");
        #     p = p + [pwx;pwy;pwz];
        #     assert( ...
        #     tmp_Wall(2,p(1),p(2),p(3)) || tmp_Wall(2,p(1)+1,p(2),p(3)) || ...
        #     tmp_Wall(3,p(1),p(2),p(3)) || tmp_Wall(3,p(1),p(2)+1,p(3)) || ...
        #     tmp_Wall(4,p(1),p(2),p(3)) || tmp_Wall(4,p(1),p(2),p(3)+1) ,  ...
        #     "Pressure loss point not adjacent to a wall");
        #     P_absMask(1,p(1),p(2),p(3)) = true;
        # self.p_loss_inds = (
        #     np.argwhere(p_absMask).astype(np.uint64).flatten(order=self.order)
        # )
        self.p_loss_inds = np.array([], dtype=np.uint64, order=self.order)
        if np.size(self.p_loss_inds) > 0:
            self.p_loss_inds += 1

        # Build a more compact uint8 wall mask:
        # Extend in size, to make it easier for c-code for-loops, which will
        # sometimes attempt to look 1 element beyond the end of the array in
        # each dimension
        self.walls_int = np.zeros(
            (self.sizeX + 2, self.sizeY + 2, self.sizeZ + 2),
            dtype=np.uint8,
            order=self.order,
        )
        np.sum(
            np.logical_not(tmp_wall[1:, :, :, :])
            * np.array([1, 2, 4], dtype=np.uint8)[:, None, None, None],
            0,
            out=self.walls_int[:-1, :-1, :-1],
        )

    def plot_volume(self, show=False, decimate_by=3):
        g = self.grid[0, :, :, :]
        x, y, z = np.mgrid[
            0 : g.shape[0] : decimate_by,
            0 : g.shape[1] : decimate_by,
            0 : g.shape[2] : decimate_by,
        ]
        g = g[x, y, z]
        fig = go.Figure(
            data=go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=g.flatten(),
                opacity=0.2,
                opacityscale="uniform",
                surface_count=7,
                colorscale="Plasma",
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )
        if show:
            fig.show()
        return fig

    @staticmethod
    def plot_trace(trace, show=False):
        v = np.stack(trace)
        plotter = VolumeSlider()
        sim = xr.DataArray(
            v,
            dims=("frame", "x", "y", "z"),
            coords=dict(
                frame=range(v.shape[0]),
                x=range(v.shape[1]),
                y=range(v.shape[2]),
                z=range(v.shape[3]),
            ),
        )
        fig = plotter.plot(sim)
        if show:
            fig.show()
        return fig

    def step(self, increment=1):
        # ### (in) Inject the input signal (usually into P)
        # Apply the input samples by injecting into the grid
        in_sig_tmp = np.zeros((increment, self.input_sig.shape[1])).astype(
            np.float32, order=self.order
        )
        nSv = max(0, min(increment, self.input_sig.shape[0] - self.sample_num))
        in_sig_tmp[:nSv, :] = self.input_sig[
            self.sample_num : (self.sample_num + nSv), :
        ]
        in_sig_tmp = (np.cumsum(in_sig_tmp, 0) + self.input_state).astype(
            np.float32, order=self.order
        )

        out_sig = self.backend.step(in_sig_tmp)
        self.input_state = in_sig_tmp[-1, :]

        # Get the output signal(s):
        if self.sample_num + increment > self.output_sig.shape[0]:
            self.output_sig = np.vstack(
                (
                    self.output_sig,
                    np.zeros(
                        (
                            self.sample_num
                            + increment
                            - self.output_sig.shape[0],
                            self.output_sig.shape[1],
                        )
                    ),
                )
            )
        self.output_sig[self.sample_num : (self.sample_num + increment), :] = (
            self.output_gain * out_sig
        )

        self.latest_level = max(
            10**-20,
            np.mean(
                np.sum(
                    self.output_sig[
                        self.sample_num : (self.sample_num + increment), :
                    ]
                    ** 2,
                    0,
                )
            ),
        )
        self.peak_ir_level = max(self.peak_ir_level, self.latest_level)
        self.max_steps_to_go -= increment
        if self.latest_level > 1e-4 * self.peak_ir_level:
            self.max_steps_to_go = self.ir_tail_countdown_steps

        self.sample_num = self.sample_num + increment

        if self.trace_interval or self.plot_interval:
            g = self.grid[0, :, :, :]
            frame = g.copy()
            if (
                self.trace_interval
                and self.sample_num % self.trace_interval == 0
            ):
                if self.trace is None:
                    self.trace = []
                self.trace.append(frame)

            if self.plot_interval and self.sample_num % self.plot_interval == 0:
                import seaborn as sns
                import matplotlib.pyplot as plt

                if self.xyz_slices is None:
                    # https://stackoverflow.com/a/68860651
                    grid_kws = {"width_ratios": (0.3, 0.3, 0.3, 0.05), "wspace": 0.2}
                    fig, (x, y, z, cbar_ax) = plt.subplots(
                        1, 4, gridspec_kw=grid_kws, figsize=(12, 4)
                    )
                    self.xyz_slices = (x, y, z)
                    self.cbar_ax = cbar_ax
                sns.heatmap(
                    ax=self.xyz_slices[0],
                    data=frame[frame.shape[0] // 2, :, :],
                    cmap="viridis",
                    cbar_ax=self.cbar_ax,
                )
                sns.heatmap(
                    ax=self.xyz_slices[1],
                    data=frame[:, frame.shape[1] // 2, :],
                    cmap="viridis",
                    cbar_ax=self.cbar_ax,
                )
                sns.heatmap(
                    ax=self.xyz_slices[2],
                    data=frame[:, :, frame.shape[2] // 2],
                    cmap="viridis",
                    cbar_ax=self.cbar_ax,
                )
                plt.draw()
                plt.pause(0.01)

        return self.max_steps_to_go < 1

    def signal_vars(self, locs):
        # Look at all the input_locs. They should be integer values, or one
        # index might be int+0.5 (the velocity value)
        locs = np.array(locs).T
        loc_int = np.ceil(locs).astype(np.uint64)
        sz = locs.shape
        loc_offset = (2 * (loc_int - locs)).astype(np.uint64)
        assert np.all(
            np.logical_or(loc_offset == 0, loc_offset == 1)
        ) and np.all(np.sum(loc_offset, 0) <= 1), (
            "All input loc indices should be int or int+0.5 and at most one is"
            " non-int"
        )
        loc_pxyz = np.sum(np.atleast_2d([1, 2, 3]).T * loc_offset, 0).astype(
            np.uint64, order=self.order
        )
        gain = np.atleast_2d(np.sqrt(1 + 2 * (loc_pxyz.T > 0)))
        inds = np.ravel_multi_index(
            (
                loc_pxyz.flatten(order=self.order),
                self.pwx + loc_int[0, :],
                self.pwy + loc_int[1, :],
                self.pwz + loc_int[2, :],
            ),
            self.grid.shape,
            order=self.order,
        ).astype(np.uint64, order=self.order)
        state = np.zeros((1,) + sz[1:], order=self.order)
        return gain, inds, state

    def make_chirp(self):
        if self.cell_size_m > 3e-3:
            # If d>3mm, we will not be able to run an input signal with full
            # 20kHz bandwidth
            n = 10
        else:
            n, _ = cheb2ord(
                20000 / (self.fs / 2),
                0.4 * np.sqrt(3) * self.step_rate,
                1.5,
                100,
            )
        b, a = cheby2(n, 100, 0.4 * np.sqrt(3) * self.step_rate)
        impulse = np.zeros(256, order=self.order)
        impulse[0] = 1.0
        return np.expand_dims(lfilter(b, a, impulse).astype(np.float32), 1)

    def run(
        self,
        in_locs,
        out_locs,
        input_sig: Optional[np.ndarray] = None,
        increment: int = 1,
        ir_tail_countdown_steps: int = 150,
        max_steps: Optional[int] = None,
        trace_interval: Optional[int] = None,
        plot_interval: Optional[int] = None,
        desc: Optional[str] = None,
    ):
        if max_steps is None:
            max_steps = int(
                np.max(self.is_solid.shape) / np.sqrt(self.step_alpha)
            )
        if input_sig is None:
            input_sig = self.make_chirp()

        self.reset_simulation_state(
            input_sig=input_sig,
            ir_tail_countdown_steps=ir_tail_countdown_steps,
            trace_interval=trace_interval,
            plot_interval=plot_interval,
            in_locs=in_locs,
            out_locs=out_locs,
        )
        done = False

        pbar = tqdm(range((max_steps + increment - 1) // increment), desc=desc)
        for _ in pbar:
            done = self.step(increment=increment)
            pbar.set_postfix(
                level=10 * np.log10(self.latest_level),
                peak=10 * np.log10(self.peak_ir_level),
                ir_tail_countdown=self.max_steps_to_go,
            )
            if done:
                break
        pcm = self.output_sig[: self.sample_num, :].copy()
        return pcm, self.fs


def run_test(shape, iloc, rel_olocs):
    nx, ny, nz = [(n - 1) // 2 for n in shape]
    voxels = VoxelGrid(nx=nx, ny=ny, nz=nz)
    grid = FEMGridPML(voxels=voxels)
    ix, iy, iz = iloc
    irs, fs = grid.run(
        [iloc],
        [[ix + x, iy + y, iz + z] for (x, y, z) in rel_olocs] + [[0, 0, 0]],
        plot_interval=10,
        max_steps=1000,
    )
    return grid, irs, fs


def test():
    grid, irs, fs = run_test(
        shape=(109, 141, 139),
        iloc=[90, 90, 90],
        rel_olocs=[
            [20, 0, 0],
            [-20, 0, 0],
            [21, 0, 0],
            [-21, 0, 0],
            [22, 0, 0],
            [-22, 0, 0],
            [20, 40, 0],
            [-20, 40, 0],
        ],
    )
    px.line(irs).show()
    # p = Path(__file__).parent / "python_output"
    # for i in tqdm(range(300), desc="saving trace"):
    #     np.save(p / f"test_pml_{i}.npy", grid.trace[i])
    return irs


def test_C():
    run_test(
        shape=(100, 70, 70),
        iloc=[10, 10, 10],
        rel_olocs=[
            [10, 0, 0],
            [20.5, 0, 0],
            [20, 0, 0],
            [-20.5, 0, 0],
            [30, 0, 0],
            [20.5, 0, 0],
            [50, 0, 0],
            [-20.5, 0, 0],
            [80, 0, 0],
            [20.5, 0, 0],
            [50, 50, 0],
            [-20.5, 0, 0],
            [20, 20, 0],
            [-20.5, 0, 0],
            [50, 50, 50],
            [20.5, 40, 0],
            [20, 20, 20],
            [-20.5, 40, 0],
            [2, 2, 2],
        ],
    )


if __name__ == "__main__":
    test()
