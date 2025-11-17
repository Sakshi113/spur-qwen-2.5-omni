import torch
from torch import nn
from typing import Tuple
from scipy.io import wavfile
import numpy as np
import plotly.graph_objects as go
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser

import os
import sys

from spatpy.signal_path.primitives import Reblocker, apply_along_axis
from spatpy.signal_path import SignalPathConfig
from plotly import subplots


def read_wav_file(filename: str) -> Tuple[torch.Tensor, float]:
    """
    Read a wave file from disk.

    Args:
        filename: path to file
    Returns:
        A tuple of `(pcm, fs)`.
    """
    fs, raw = wavfile.read(filename)
    if raw.dtype.name.startswith("int"):
        pcm = raw * (1.0 / (1 << (raw.itemsize * 8 - 1)))
    else:
        assert raw.dtype.name == "float32"
        pcm = raw
    if pcm.ndim == 1:
        pcm = np.expand_dims(pcm, 0)

    x = torch.tensor(pcm.astype(np.float32), names=("sample", "ch"))
    return (x, fs)


def write_wav_file(filename, fs, pcm):
    if "frame" in pcm.names:
        pcm = pcm.align_to("frame", "sample", "ch")
        nch = pcm.shape[-1]
        pcm = pcm.rename(None).reshape(-1, nch).refine_names("sample", "ch")
    return wavfile.write(
        filename,
        int(fs),
        (pcm.align_to("sample", "ch").numpy() * (2 ** 16 - 1)).astype("int16"),
    )


def format_z(z, db=False, dynamic_range_db=None, title=None):
    if dynamic_range_db is None:
        dynamic_range_db = 120.0
    if title is None:
        title = ""
    names = None
    if isinstance(z, torch.Tensor):
        names = z.names

        # put frame last so that we don't get rid of it
        # with the loop following
        if "frame" in names:
            z = z.align_to(..., "frame")
            names = z.names

        # reduce dimensionality by picking the first element
        # of each dimension
        while z.dim() > 2:
            z = z[0, ...]
            if len(title) > 0:
                title += ", "
            title += f"{names[0]}[0]"
            names = names[1:]

        # put frame first so it becomes the x axis in the plot
        if "frame" in names:
            z = z.align_to("frame", ...)
            names = z.names
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
    z = z.rename(None)
    if db:
        z = torch.clamp(20 * torch.log10(z), torch.max(z) - dynamic_range_db)
    if z.dim() == 3:
        z = z[0, :, :]
    return z.detach().numpy().T, title, names


def stacked_heatmap(*args, **kwargs):
    """
    Plot heatmaps stacked vertically with a shared x axis.

    Positional arguments are the z-matrices from top to bottom.

    Keyword arguments are the same as for seaborn.heatmap, with numbers (starting from 1).
    e.g. vmin2 sets the minimum z value for the second heatmap from the top.

    Returns a `plotly.graph_objects.Figure <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html>`_.
    """
    show = kwargs.pop("show", False)
    nplot = len(args)
    cmap = []
    xstep = []
    ystep = []
    yticklabels = []
    xlabel = []
    ylabel = []
    zlabel = []
    vmin = []
    vmax = []
    dynamic_range_db = []
    db = []
    center = []
    row_titles = []
    for j in range(nplot):
        i = j + 1
        cmap.append(kwargs.pop(f"cmap{i}", "viridis" if j == 0 else "jet"))
        xstep.append(kwargs.pop(f"xstep{i}", None))
        ystep.append(kwargs.pop(f"ystep{i}", None))
        yticklabels.append(kwargs.pop(f"yticklabels{i}", None))
        ylabel.append(kwargs.pop(f"ylabel{i}", None))
        zlabel.append(kwargs.pop(f"zlabel{i}", None))
        vmin.append(kwargs.pop(f"vmin{i}", None))
        vmax.append(kwargs.pop(f"vmax{i}", None))
        center.append(kwargs.pop(f"center{i}", None))
        db.append(kwargs.pop(f"db{i}", None))
        dynamic_range_db.append(kwargs.pop(f"dynamic_range_db{i}", None))
        row_titles.append(kwargs.pop(f"title{i}", None))
    vertical_spacing = 0.3 / nplot
    height_ratios = kwargs.pop("height_ratios", [1 for arg in args])
    row_heights = [ratio / sum(height_ratios) for ratio in height_ratios]
    fig = subplots.make_subplots(
        rows=nplot,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        row_titles=row_titles,
        x_title=kwargs.pop("xlabel", None),
        vertical_spacing=vertical_spacing,
    )
    for (i, title) in enumerate(ylabel):
        fig.update_yaxes(title_text=title, row=(i + 1), col=1)
    title = kwargs.pop("title", None)
    if title:
        fig.update_layout(title=dict(text=title, xanchor="center", x=0.5))

    for (i, toplot) in enumerate(args):
        z, _, _ = format_z(toplot, db=db[i], dynamic_range_db=dynamic_range_db[i])
        offset = sum(row_heights[:i]) + i * 0.035 - 0.035
        hm = go.Heatmap(
            z=z,
            y=yticklabels[i],
            colorbar=dict(
                len=row_heights[i] - 0.01, yanchor="top", y=1 - offset, title=zlabel[i]
            ),
            colorscale=cmap[i],
            zmin=vmin[i],
            zmax=vmax[i],
            zmid=center[i],
        )
        fig.add_trace(hm, row=i + 1, col=1)
    if show:
        fig.show()
    return fig


def heatmap(
    z,
    xaxis_title=None,
    yaxis_title=None,
    xaxis_type=None,
    yaxis_type=None,
    title=None,
    db=False,
    dynamic_range_db=None,
    **kwargs,
):
    if title is None:
        title = ""

    new_z, title, names = format_z(
        z, db=db, dynamic_range_db=dynamic_range_db, title=title
    )
    if names:
        if xaxis_title is None:
            xaxis_title = names[0]
        if yaxis_title is None:
            yaxis_title = names[1]

    return go.Figure(
        go.Heatmap(z=new_z, **kwargs),
        go.Layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis_type=xaxis_type,
            yaxis_type=yaxis_type,
        ),
    )


# from plotly.io.base_renderers
def open_html_in_browser(html, using=None, new=0, autoraise=True):
    html = html.encode("utf8")

    class OneShotRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            bufferSize = 1024 * 1024
            for i in range(0, len(html), bufferSize):
                self.wfile.write(html[i : i + bufferSize])

        def log_message(self, format, *args):
            # Silence stderr logging
            pass

    server = HTTPServer(("127.0.0.1", 0), OneShotRequestHandler)
    webbrowser.get(using).open(
        "http://127.0.0.1:%s" % server.server_port, new=new, autoraise=autoraise
    )
    server.handle_request()


def mermaid_html(mermaid, filename=None, show=False):
    page = f"""
    <html>
    <body>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true}});</script>
        <div class="mermaid">
            {mermaid}
        </div>
    </body>
    </html>
    """
    if show:
        open_html_in_browser(page)
    if filename:
        with open(filename, "wt") as fobj:
            fobj.write(page)
    return page


def load_file(filename: str, batch_mode: bool = True, block_size_ms: float = 10.0):
    pcm, fs = read_wav_file(filename)
    config = SignalPathConfig(
        fs=fs, nmic=pcm.shape[1], block_size_ms=block_size_ms, batch_mode=batch_mode
    )
    reblock = Reblocker(block_size=config.block_size)
    frames = reblock(pcm)
    return config, frames
