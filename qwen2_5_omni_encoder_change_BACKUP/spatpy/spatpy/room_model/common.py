import numpy as np
import plotly.graph_objects as go
from scipy import signal
from dataclasses import dataclass
from plotly import subplots


@dataclass
class MakeRoomParams:
    fs: float = 48000.0
    speed_of_sound: float = 343.3
    lp_kernel_freq_hz: float = 150.0
    mid_lp_kernel_freq_hz: float = 1000.0

    def butter_based_lp_kernel(self, freq_hz):
        impulse = np.zeros(int(2 / (freq_hz / (self.fs / 2))))
        impulse[0] = 1
        b, a = signal.butter(1, freq_hz, fs=self.fs)
        kernel = signal.lfilter(b, a, impulse)
        return kernel

    def get_lp_kernel(self):
        return self.butter_based_lp_kernel(self.lp_kernel_freq_hz)

    def get_mid_kernel(self):
        return self.butter_based_lp_kernel(self.mid_lp_kernel_freq_hz)


@dataclass
class ReverbParams:
    src_x_m: float = 2 
    src_y_m: float = 0 
    src_z_m: float = 0
    room_volume: float = 60  # Cubic meters
    reverb_time_lf: float = 0.24  # Low frequency reverb time in seconds
    reverb_time_hf: float = 0.14
    diff_s: float = 0.002  # Diffusion time in seconds
    diff_pct: float = 30  # Percent of each echo that's diffusion


# Plotting functions used for debugging
def plot_arr(arr):
    fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=np.arange(len(arr)), y=arr), row=1, col=1)
    fig.show()


def plot_mat(mat):
    fig = subplots.make_subplots(rows=1, cols=1, shared_xaxes=True)
    for i in range(mat.shape[-1]):
        arr = mat[:, i]
        fig.add_trace(go.Scatter(x=np.arange(len(arr)), y=arr), row=1, col=1)
    fig.show()
