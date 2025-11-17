from ufb_banding.ufb import TransformParams, TransformCoefs

from ufb_banding.banding import (
    BandingParams,
    BandingCoefs,
)
from ufb_banding.banding.spatial import (
    SpatialBandingCoefs,
    SpatialBandingParams,
)
from ufb_banding.backends.torch import dft_mod_matrix
import torch
from torch import nn
import numpy as np

from spatpy.signal_path.primitives import (
    Reblocker,
    HistoryBuffer,
    IIRSmoother,
    apply_along_axis,
)

torch.fx.wrap("apply_along_axis")


from spatpy.signal_path.io import read_wav_file
from spatpy.signal_path import SignalPathConfig

from typing import Optional, Tuple, Union
from enum import Enum


class ForwardTransform(nn.Module):
    def __init__(
        self,
        transform: Union[TransformParams, TransformCoefs],
        sigpath: Optional[SignalPathConfig] = None,
        conv: bool = False,
    ):
        super().__init__()
        if sigpath is None:
            coefs = transform
        else:
            coefs = TransformCoefs(
                params=transform, block_size=sigpath.block_size
            )
        self.coefs = coefs
        self.mod_matrix = dft_mod_matrix(coefs).to(dtype=torch.complex64) \
                                               .refine_names("sample", "bin")
        self.hist = HistoryBuffer(
            "sample",
            coefs.block_size,
            coefs.pad_ana_len,
            conv=conv,
            batch_mode=True if sigpath is None else sigpath.batch_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.mod_matrix.device: 
            self.mod_matrix = self.mod_matrix.to(x.device)
        delay_line = self.hist(x).align_to(..., "sample")
        return torch.matmul(delay_line + torch.tensor(0j), self.mod_matrix)

    @property
    def fbin(self):
        return torch.from_numpy(
            BandingCoefs.get_bin_freq(self.sp.fs, self.tfm_coefs.nb_modulations)
        ).float()

    @property
    def nbin(self):
        return self.tfm_coefs.nb_modulations


class ForwardBanding(nn.Module):
    def __init__(
        self,
        banding: Union[BandingParams, BandingCoefs],
        sigpath: Optional[SignalPathConfig] = None,
        trainable: bool = False,
    ):
        super().__init__()
        if sigpath is None:
            coefs = banding
        else:
            coefs = BandingCoefs(
                banding, dt_ms=sigpath.block_size_ms, fs=sigpath.fs
            )
        band_matrix = (
            torch.from_numpy(coefs.band_matrix.T)
            .float()
            .refine_names("bin", "band")
        )
        self.band_matrix = (
            nn.Parameter(band_matrix) if trainable else band_matrix
        )

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        # put the bin dimension last
        bins = bins.align_to(..., "bin")
        names = bins.names
        # wipe the names of all the dimensions
        bin_power = (bins * bins.rename(None).conj()).rename(None).real.float()
        # create the bands tensor
        bands = torch.matmul(bin_power, self.band_matrix)
        # rename the dimensions to be whatever they were, with 'bin' replaced by 'band'
        return bands.refine_names(*names[:-1], "band")


class BinCovariance(nn.Module):
    def __init__(self, normalise: bool = True):
        super().__init__()
        self.do_normalise = normalise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.align_to(..., "ch", "bin")
        names = x.names
        nch = x.shape[-2]
        x = x.rename(None)
        # compute covariance matrix
        bincov = torch.matmul(x, x.conj().transpose(-1, -2))
        # flatten
        bincov = torch.flatten(bincov, -2, -1)
        # normalise by power in each mic
        if self.do_normalise:
            diag = diag_indices(nch)
            bincov *= 1 / bincov.rename(None).real[..., diag].sum(
                axis=-1, keepdims=True
            ).clamp(min=1e-20)
        bincov.names = names[:-2] + ("cov",)
        return bincov


@torch.fx.wrap
def diag_indices(nch):
    return torch.arange(nch) + torch.arange(nch * nch)[::nch]


@torch.fx.wrap
def copy_diag(a, b, nch):
    diag = diag_indices(nch)
    names = a.names
    a = a.rename(None)
    a[..., diag, :] = b[..., diag, :]
    a.names = names
    return a


class PowerVector(nn.Module):
    def __init__(
        self,
        banding: Union[SpatialBandingParams, SpatialBandingCoefs],
        sigpath: Optional[SignalPathConfig] = None,
        cov_to_pv_trainable: bool = False,
        band_matrix_trainable: bool = False,
        smoothing_trainable: bool = False,
        hz_s_per_band: Optional[float] = 5.0,
        nch: Optional[int] = None,
        normalise: bool = True,
        do_phase_adjust: bool = True
    ):
        super().__init__()
        if sigpath is None:
            coefs = banding
        else:
            coefs = SpatialBandingCoefs(
                banding,
                nch=sigpath.nmic if nch is None else nch,
                dt_ms=sigpath.block_size_ms,
                fs=sigpath.fs,
            )
        self.coefs = coefs

        covmat = torch.view_as_real(torch.from_numpy(self.coefs.cov_to_pv))
        band_matrix = torch.view_as_real(torch.from_numpy(
            self.coefs.band_matrix.T.astype(np.complex64)
        ))
        if cov_to_pv_trainable:
            self.cov_to_pv = nn.Parameter(covmat)
        else:
            self.register_buffer("cov_to_pv", covmat, persistent=False)
        if band_matrix_trainable:
            self.band_matrix = nn.Parameter(band_matrix)
        else:
            self.register_buffer("band_matrix", band_matrix, persistent=False)
        self.do_normalise = normalise
        self.bincov = BinCovariance(self.do_normalise)
        self.do_phase_adjust = do_phase_adjust

        if self.coefs.do_smoothing:
            tau = torch.from_numpy(self.coefs.tau)
            if smoothing_trainable:
                self.tau = nn.Parameter(tau)
            else:
                self.register_buffer("tau", tau, persistent=False)
            self.smooth = IIRSmoother()

    @staticmethod
    def adjust_phase(bincov, nch, n=1):
        # perform phase adjustments (for off-diagonals)
        # goal of this phase adjustment is to compute:
        # Cov[i] = |Cov[i]| * angle(conj(Cov[i + n]) * Cov[i - n])
        # where 'angle(x)' is the unitary angle of the complex number x,
        # and 'conj(x)' is the complex conjuation function.
        # The 'a', 'b', 'c' notation used corresponds to:
        # C = |C| * angle(A * B).

        bincov = bincov.align_to(..., "bin")
        names = bincov.names
        bincov = bincov.rename(None)

        angle = (bincov[..., : -2 * n].conj() * bincov[..., 2 * n :]).angle()
        angle = torch.cat((angle[..., 0:1], angle, angle[..., -1:]), dim=-1)
        angle.names = names
        torch_trace_1j = 1j
        adjusted = bincov.abs() * torch.exp(torch_trace_1j * angle)
        # leave diagonal elements alone
        adjusted = copy_diag(adjusted, bincov, nch)
        adjusted = adjusted.refine_names(..., "cov", "bin")
        return adjusted.align_to(..., "cov").to(dtype=torch.complex64)

    @staticmethod
    def purity(pv: torch.Tensor) -> torch.Tensor:
        pv = pv.align_to(..., "pv")
        nch = pv.shape[-1] ** 0.5
        names = pv.names
        purity = torch.linalg.norm(pv.rename(None), dim=(-1))
        purity = ((nch**0.5) * purity - 1) / ((nch**0.5) - 1)
        purity = purity.clamp(0, 1)
        purity.names = names[:-1]
        return purity

    def forward(self, bins: torch.Tensor) -> torch.Tensor:
        bins = bins.align_to(..., "ch", "bin")
        nch = bins.shape[-2]
        cov = apply_along_axis(self.bincov, "bin", bins)
        # import pdb
        # pdb.set_trace()
        if self.do_phase_adjust:
            cov = self.adjust_phase(cov, nch)

        # # --- START: ADD THESE DEBUGGING LINES ---
        # mat1 = cov.align_to(..., "bin")
        # mat2 = torch.view_as_complex(self.band_matrix)
        # print("\n--- DEBUGGING MATMUL ---")
        # print(f"Shape of cov BEFORE align: {cov.shape}")
        # print(f"Names of cov BEFORE align: {cov.names}")
        # print(f"Shape of mat1 (cov AFTER align): {mat1.shape}")
        # print(f"Shape of mat2 (band_matrix): {mat2.shape}")
        # print("--------------------------\n")
        # # --- END: ADD THESE DEBUGGING LINES ---

        # band_cov = torch.matmul(
        #     cov.align_to(..., "bin"), torch.view_as_complex(self.band_matrix)
        # ).align_to(..., "cov").refine_names(..., "band", "cov")

        # --- START: THIS IS THE FINAL FIX ---
        # Get the dimension names and find the index of the unnamed dimension
        names = list(cov.names)
        try:
            # The dimension we want to multiply is the one that is not named.
            unnamed_dim_index = names.index(None)
        except ValueError:
            # Fallback in case there are no unnamed dimensions, which would be an error.
            raise ValueError("Could not find the dimension to contract in 'cov' tensor.")

        # Create a list of all dimension indices, e.g., [0, 1, 2, 3, 4]
        dim_indices = list(range(cov.dim()))
        # Remove the index of the unnamed dimension from the list.
        dim_indices.pop(unnamed_dim_index)
        # Create the new permutation order by moving the unnamed dimension's index to the end.
        # e.g., if index was 3, the order becomes [0, 1, 2, 4, 3]
        permutation = dim_indices + [unnamed_dim_index]

        # STEP 1: Aggressively drop the names from the tensor.
        cov_unnamed = cov.rename(None)

        print("Names of cov BEFORE align: {cov_unnamed.names}")

        # STEP 2: Now that the tensor is unnamed, permute it. This will now succeed.
        mat1 = cov_unnamed.permute(*permutation)
        mat2 = torch.view_as_complex(self.band_matrix)

        # STEP 3: The multiplication will succeed.
        band_cov = torch.matmul(mat1, mat2)

        # STEP 4: Restore the names for subsequent operations.
        # We build the new name list from the original names, in the new permuted order.
        output_names = [names[i] for i in dim_indices] + ['band']
        band_cov = band_cov.refine_names(*output_names)
        # --- END: FINAL PERMUTE-BASED FIX ---

        # normalise covariance matrix
        if self.do_normalise:
            diag = diag_indices(nch)
            diag_sum = (
                band_cov.rename(None)
                .real[..., diag]
                .sum(axis=-1, keepdims=True)
                .clamp(1e-20)
            )
            band_cov *= 1 / diag_sum
        if self.coefs.do_smoothing:
            band_cov = apply_along_axis(self.smooth, "frame", band_cov, self.tau)

        band_pv = torch.matmul(
            torch.view_as_complex(self.cov_to_pv), band_cov.align_to(..., "cov", "frame")
        ).refine_names(..., "pv", "frame")
        names = band_pv.names
        band_pv = band_pv.rename(None).real
        band_pv.names = names
        return band_pv.align_to(..., "pv")
