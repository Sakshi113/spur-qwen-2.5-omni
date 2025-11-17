import numpy as np
from pathlib import Path
from scipy.io import loadmat
import plotly.graph_objects as go
from plotly.subplots import make_subplots

py_dir = Path("python_output")
m_dir = Path("matlab_output")


def plot_step(step):
    py_frame = np.load(py_dir / f"test_pml_{step}.npy").squeeze()
    m_frame = loadmat(m_dir / f"test_pml_{step + 1}.mat", squeeze_me=True)["x"]
    n = py_frame.shape[0] // 2
    py_frame = py_frame[n, :, :]
    n = m_frame.shape[0] // 2
    m_frame = m_frame[n, :, :]

    fig = make_subplots(rows=1, cols=3, shared_xaxes=True, shared_yaxes=True, subplot_titles=['Matlab', 'Python', 'Difference']) 
    fig.add_trace(
        go.Heatmap(z=m_frame),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Heatmap(z=py_frame),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Heatmap(z=(m_frame - py_frame)),
        row=1,
        col=3
    )
    fig.show()
    return fig


if __name__ == '__main__':
    plot_step(200)
# for step in range(1000):
