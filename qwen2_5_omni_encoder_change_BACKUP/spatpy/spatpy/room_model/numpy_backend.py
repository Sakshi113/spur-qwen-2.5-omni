import math

from spatpy.room_model.common import (
    plot_mat,
    MakeRoomParams,
    ReverbParams,
)
from dataclasses import dataclass
import numpy as np
from spatpy.dsp import convolve_1d


@dataclass
class MakeRoomNumpy(MakeRoomParams):
    def __post_init__(self):
        self.low_lp_kernel = self.get_lp_kernel()
        self.mid_lp_kernel = self.get_mid_kernel()

    def convolve_kernel(self, foa_ir, kernel):
        return np.vstack(
            [
                convolve_1d(foa_ir[i, :], kernel, mode="full")
                for i in range(foa_ir.shape[0])
            ]
        )

    def generate(self, params: ReverbParams):
        src_loc = np.array([params.src_x_m, params.src_y_m, params.src_z_m])
        v = params.room_volume
        c = self.speed_of_sound
        fs = self.fs
        d = (src_loc**2).sum() ** 0.5  # Source distance

        mfp = (2 / 3) * (v ** (1 / 3))
        ir_len = round(fs * params.reverb_time_lf)
        t = np.arange(0, ir_len) / fs + (d / c)
        density = 4 * math.pi * (t**2) * (c**3) / (fs * v)

        echo_pwr_gain = (
            (np.random.rand(ir_len) < density)
            * np.maximum(1, density)
            / (t**2)
        )
        echo_pwr_gain[0 : round(0.001 * fs)] = 0

        diffusion_kernel = np.arange(params.diff_s * fs, 1, -1) ** 2
        diffusion_kernel /= diffusion_kernel.sum()  # normalise the kernel

        diffuse_echo_fraction = 1 - (
            (1 - (params.diff_pct / 100)) ** (t * c / mfp)
        )
        diffusion_mask = np.maximum(
            0,
            convolve_1d(
                echo_pwr_gain * diffuse_echo_fraction,
                diffusion_kernel,
                mode="full",
            ),
        )
        diffusion_mask = diffusion_mask[:ir_len]
        echo_pwr_gain[0] = 1 / (t[0] ** 2)  # Insert hearing the sound dry
        echo_dir = np.random.randn(3, ir_len)
        echo_dir[:, 0] = src_loc
        echo_dir /= np.linalg.norm(echo_dir, axis=0)

        noise_ir = np.random.randn(4, ir_len) * np.array(
            [[1], [math.sqrt(1 / 3)], [math.sqrt(1 / 3)], [math.sqrt(1 / 3)]],
        )
        foa_ir = np.vstack((np.ones((1, ir_len)), echo_dir))
        foa_ir = foa_ir * np.sqrt(echo_pwr_gain) * np.sqrt(
            1 - diffuse_echo_fraction
        ) + noise_ir * np.sqrt(diffusion_mask)

        foa_dc_to_low = self.convolve_kernel(foa_ir, self.low_lp_kernel)
        foa_dc_to_low = foa_dc_to_low[:, :ir_len]
        foa_dc_to_mid = self.convolve_kernel(foa_ir, self.mid_lp_kernel)
        foa_dc_to_mid = foa_dc_to_mid[:, :ir_len]

        foa_mid_to_high = foa_ir - foa_dc_to_mid
        foa_low_to_mid = foa_dc_to_mid - foa_dc_to_low

        foa_ir = foa_mid_to_high * (
            0.001 ** (t / params.reverb_time_hf)
        ) + foa_low_to_mid * (0.001 ** (t / params.reverb_time_lf))
        normalise = np.sqrt((foa_ir[:, 0] ** 2).sum())
        return foa_ir.T / normalise


def main():
    makeroom = MakeRoomNumpy()
    params = ReverbParams()
    plot_mat(makeroom.generate(params))


if __name__ == "__main__":
    main()
