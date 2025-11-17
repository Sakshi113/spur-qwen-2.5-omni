import math
import torch

from spatpy.room_model.common import (
    plot_mat,
    ReverbParams,
    MakeRoomParams,
)
from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class MakeRoomTorch(MakeRoomParams):
    device: Optional[Any] = None

    def __post_init__(self):
        if self.device is None:
            self.device = (
                torch.device("cuda")
                if (torch.cuda.is_available())
                else torch.device("cpu")
            )

        # The flip is necessary due to how torch likes reversed kernels
        self.low_lp_kernel = torch.tensor(self.get_lp_kernel(), device=self.device).float().flip(dims=(0,))
        self.mid_lp_kernel = torch.tensor(self.get_mid_kernel(), device=self.device).float().flip(dims=(0,))

    def generate(self, params: ReverbParams):
        src_loc = torch.tensor((params.src_x_m, params.src_y_m, params.src_z_m), device=self.device)
        v = params.room_volume
        c = self.speed_of_sound
        fs = self.fs
        d = (src_loc**2).sum() ** 0.5  # Source distance

        mfp = (2 / 3) * (v ** (1 / 3))
        ir_len = round(fs * params.reverb_time_lf)
        t = torch.arange(0, ir_len, device=self.device) / fs + (d / c)
        density = 4 * math.pi * (t**2) * (c**3) / (fs * v)

        echo_pwr_gain = (
            (torch.rand((ir_len), device=self.device).lt(density))
            * density.clamp(min=1, max=math.inf)
            / (t**2)
        )
        echo_pwr_gain[0 : round(0.001 * fs)] = 0

        # It looks like conv1d reverses the kernel? So I am creating a ramp up
        diffusion_kernel = (
            torch.arange(1, params.diff_s * fs, device=self.device) ** 2
        )  # torch.arange(diff_s * fs, 1, -1, device=self.device) ** 2
        diffusion_kernel /= diffusion_kernel.sum()  # normalise the kernel

        diffuse_echo_fraction = 1 - (
            (1 - (params.diff_pct / 100)) ** (t * c / mfp)
        )
        diffusion_mask = torch.conv1d(
            (echo_pwr_gain * diffuse_echo_fraction).view(1, 1, -1),
            diffusion_kernel.view(1, 1, -1),
            padding=len(diffusion_kernel),
        ).squeeze()
        diffusion_mask = diffusion_mask[:ir_len]

        echo_pwr_gain[0] = 1 / (t[0] ** 2)  # Insert hearing the sound dry

        echo_dir = torch.randn((ir_len, 3), device=self.device)
        echo_dir[0, :] = src_loc
        echo_dir /= self.broadcast((echo_dir**2).sum(dim=-1).sqrt(), echo_dir)

        noise_ir = torch.randn((ir_len, 4), device=self.device) * torch.tensor(
            (1, math.sqrt(1 / 3), math.sqrt(1 / 3), math.sqrt(1 / 3)),
            device=self.device,
        )
        foa_ir = torch.cat(
            (torch.ones((ir_len, 1), device=self.device), echo_dir), dim=-1
        )
        foa_ir = foa_ir * self.broadcast(
            echo_pwr_gain.sqrt() * (1 - diffuse_echo_fraction).sqrt(), foa_ir
        ) + noise_ir * self.broadcast(diffusion_mask.sqrt(), noise_ir)

        foa_dc_to_low = (
            torch.conv1d(
                foa_ir.transpose(-2, -1).view(4, 1, -1),
                self.low_lp_kernel.view(1, 1, -1),
                padding=len(self.low_lp_kernel),
            )
            .squeeze()
            .transpose(-1, -2)
        )
        foa_dc_to_low = foa_dc_to_low[:ir_len]
        foa_dc_to_mid = (
            torch.conv1d(
                foa_ir.transpose(-2, -1).view(4, 1, -1),
                self.mid_lp_kernel.view(1, 1, -1),
                padding=len(self.mid_lp_kernel),
            )
            .squeeze()
            .transpose(-1, -2)
        )
        foa_dc_to_mid = foa_dc_to_mid[:ir_len]

        foa_mid_to_high = foa_ir - foa_dc_to_mid
        foa_low_to_mid = foa_dc_to_mid - foa_dc_to_low

        foa_ir = foa_mid_to_high * self.broadcast(
            (0.001 ** (t / params.reverb_time_hf)), foa_ir
        ) + foa_low_to_mid * self.broadcast(
            (0.001 ** (t / params.reverb_time_lf)), foa_ir
        )
        normalise = (foa_ir[:, 0] ** 2).sum().sqrt() # Normalise energy
        # normalise = foa_ir[0, 0] # Normalise dry signal
        return foa_ir / normalise
    
    def generate_batch(self, params: List[ReverbParams]):
        src_loc = torch.tensor([[param.src_x_m, param.src_y_m, param.src_z_m] for param in params], dtype=torch.float32, device=self.device)
        vol = torch.tensor([param.room_volume for param in params], device=self.device)
        c = self.speed_of_sound
        fs = self.fs
        d = (src_loc**2).sum(dim=-1) ** 0.5  # Source distance

        mfp = (2 / 3) * (vol ** (1 / 3))
        ir_len = round(fs * max([param.reverb_time_lf for param in params]))
        t = torch.arange(0, ir_len, device=self.device).unsqueeze(0).repeat(len(d), 1) / fs + (d.unsqueeze(-1) / c)
        density = 4 * math.pi * (t**2) * (c**3) / (fs * vol.unsqueeze(-1))

        echo_pwr_gain = (
            (torch.rand((len(params), ir_len), device=self.device).lt(density))
            * density.clamp(min=1, max=math.inf)
            / (t**2)
        )
        echo_pwr_gain[:, 0 : round(0.001 * fs)] = 0

        # It looks like conv1d reverses the kernel? So I am creating a ramp up
        diff_s = torch.tensor([param.diff_s for param in params], device=self.device)
        diffusion_kernel = (
            (torch.arange(1, diff_s.max() * fs, device=self.device) ** 2).unsqueeze(0).repeat(len(d), 1)
        )  # torch.arange(diff_s * fs, 1, -1, device=self.device) ** 2
        for i, v in enumerate(diff_s): diffusion_kernel[i, int(v * fs):] = 0
        diffusion_kernel /= diffusion_kernel.sum(-1).unsqueeze(-1)  # normalise the kernel

        diffuse_echo_fraction = 1 - (
            (1 - (torch.tensor([param.diff_pct for param in params], device=self.device) / 100)).unsqueeze(-1) ** (t * c / mfp.unsqueeze(-1))
        )
        diffusion_mask = torch.conv1d(
            (echo_pwr_gain * diffuse_echo_fraction).unsqueeze(0),
            diffusion_kernel.unsqueeze(1),
            padding=diffusion_kernel.shape[-1],
            groups=len(params)
        ).squeeze()
        diffusion_mask = diffusion_mask[:, :ir_len]

        echo_pwr_gain[:, 0] = 1 / (t[:, 0] ** 2)  # Insert hearing the sound dry

        echo_dir = torch.randn((len(d), ir_len, 3), device=self.device)
        echo_dir[:, 0, :] = src_loc
        echo_dir /= self.broadcast((echo_dir**2).sum(dim=-1).sqrt(), echo_dir)

        noise_ir = torch.randn((len(d), ir_len, 4), device=self.device) * torch.tensor(
            (1, math.sqrt(1 / 3), math.sqrt(1 / 3), math.sqrt(1 / 3)),
            device=self.device,
        )
        foa_ir = torch.cat(
            (torch.ones((len(d), ir_len, 1), device=self.device), echo_dir), dim=-1
        )
        foa_ir = foa_ir * self.broadcast(
            echo_pwr_gain.sqrt() * (1 - diffuse_echo_fraction).sqrt(), foa_ir
        ) + noise_ir * self.broadcast(diffusion_mask.sqrt(), noise_ir)

        foa_dc_to_low = (
            torch.conv1d(
                # Will this bleed the tail of one IR to the next?
                foa_ir.transpose(-2, -1).flatten(0, 1).unsqueeze(1),
                self.low_lp_kernel.view(1, 1, -1),
                padding=len(self.low_lp_kernel),
            )
            .reshape(len(d), 4, -1)
            .transpose(-1, -2)
        )
        foa_dc_to_low = foa_dc_to_low[:, :ir_len]
        foa_dc_to_mid = (
            torch.conv1d(
                foa_ir.transpose(-2, -1).flatten(0, 1).unsqueeze(1),
                self.mid_lp_kernel.view(1, 1, -1),
                padding=len(self.mid_lp_kernel),
            )
            .reshape(len(d), 4, -1)
            .transpose(-1, -2)
        )
        foa_dc_to_mid = foa_dc_to_mid[:, :ir_len]

        foa_mid_to_high = foa_ir - foa_dc_to_mid
        foa_low_to_mid = foa_dc_to_mid - foa_dc_to_low

        rt60_hf = torch.tensor([param.reverb_time_hf for param in params], device=self.device)
        rt60_lf = torch.tensor([param.reverb_time_lf for param in params], device=self.device)
        foa_ir = foa_mid_to_high * self.broadcast(
            (0.001 ** (t / rt60_hf.unsqueeze(-1))), foa_ir
        ) + foa_low_to_mid * self.broadcast(
            (0.001 ** (t / rt60_lf.unsqueeze(-1))), foa_ir
        )
        normalise = (foa_ir[:, :, 0] ** 2).sum(-1).sqrt() # Normalise energy
        # normalise = foa_ir[0, 0] # Normalise dry signal
        ir_lens = (torch.tensor([param.reverb_time_lf for param in params], device=self.device) * fs).round().int()
        return foa_ir / normalise.reshape(-1, 1, 1), ir_lens

    # Used to expand vectors to matrix size for multiply/division operations
    def broadcast(self, vec, mat):
        return vec.unsqueeze(-1).expand_as(mat)


def main():
    makeroom = MakeRoomTorch(fs=48000)
    params = ReverbParams()
    plot_mat(makeroom.generate(params).cpu())


if __name__ == "__main__":
    main()
