from spatpy.geometry import PointCloud, polarticks
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass, field
from spatpy.speakers import SpeakerChannelFormat
import xarray as xr
from typing import Optional, Dict, List, Tuple, Any, Union
import plotly.express as px
import os

# 12 colors, paired
# https://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
COLOR_PALETTE = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
]


ROTATE_CAMERA_SNIPPET = """p = document.getElementById('{plot_id}'); p.on('plotly_relayout', function(evt) {if (evt.hasOwnProperty('scene.camera')) {Plotly.relayout(p, {'polar.angularaxis.rotation': Math.atan2(evt['scene.camera'].eye.x, evt['scene.camera'].eye.y)*180/Math.PI+180})}})"""


def format_freq(f):
    return f"{f / 1000:.2f}kHz"


def angularticks(angularaxis):
    return polarticks(0, 1, 4) if angularaxis == "el" else polarticks(-1, 1, 2)


def unwrap_phase_with_target(phase, target_phase):
    u = np.unwrap(phase)
    t = np.unwrap(target_phase)
    k = 0
    for i in range(len(target_phase)):

        def cost(n):
            return np.abs(t[i] - (u[i] + 2 * n * np.pi))

        if cost(k + 1) < cost(k):
            k += 1
        elif cost(k - 1) < cost(k):
            k -= 1
        u[i] += 2 * k * np.pi
    return u


@dataclass
class BoundsTracker:
    extrema: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def track(self, **kwargs):
        for (k, vs) in kwargs.items():
            if not isinstance(vs, list):
                vs = [vs]
            for v in vs:
                if not isinstance(v, (np.ndarray, xr.DataArray)):
                    continue
                if k not in self.extrema:
                    self.extrema[k] = (np.min(v), np.max(v))
                else:
                    minv, maxv = self.extrema[k]
                    self.extrema[k] = (
                        min(np.min(v), minv),
                        max(np.max(v), maxv),
                    )


@dataclass
class UpdateTracker:
    update_keys: List[str]
    steps: Optional[Dict] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = {k: [] for k in self.update_keys}

    def push(self, ud):
        for k in self.update_keys:
            v = ud.get(k, None)
            if isinstance(v, xr.DataArray):
                v = v.values
            self.steps[k].append(v)


@dataclass
class ResponsePlotter:
    flat: Optional[bool] = None
    phase_as_intensity: bool = True
    polar_range_db: float = 30
    mesh_colorscale: Any = field(default_factory=(lambda: "spectral_r"))
    phase_colorscale: str = field(default_factory=(lambda: "edge"))
    name_colors: Optional[Dict[str, int]] = field(default_factory=dict)
    color_palette: Optional[List[str]] = field(
        default_factory=(lambda: COLOR_PALETTE)
    )

    def __post_init__(self):
        if self.phase_as_intensity:
            self.mesh_colorscale = self.phase_colorscale

        if isinstance(self.mesh_colorscale, str):
            self.mesh_colorscale = px.colors.get_colorscale(
                self.mesh_colorscale
            )

    def name_color(self, name, is_target=False):
        if name not in self.name_colors:
            self.name_colors[name] = len(self.name_colors) % len(
                self.color_palette
            )
        n = self.name_colors[name]
        i = 0 if is_target else 1
        return COLOR_PALETTE[2 * n + i]

    def add_freq_lines(self, fig, f_sel_hz=None, f_nyq_hz=None):
        if f_nyq_hz is not None:
            fig.add_vline(
                x=f_nyq_hz,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"f_nyq",
                annotation_font_color="gray",
                annotation_yanchor="bottom",
                annotation_y=1.0,
            )
        if f_sel_hz is not None:
            fig.add_vline(
                x=f_sel_hz,
                line_dash="dot",
                line_color="gray",
                annotation_text="f_sel",
                annotation_font_color="gray",
                annotation_yanchor="bottom",
                annotation_y=1.0,
            )

    def freq_suffix(self, f_sel_hz=None, f_nyq_hz=None):
        suff = ""
        if f_nyq_hz is not None:
            suff += f", f_nyq = {format_freq(f_nyq_hz)}"
        if f_sel_hz is not None:
            suff += f", f_sel = {format_freq(f_sel_hz)}"
        return suff

    def get_2d_updates(self, resp, target, angularaxis):
        points = PointCloud.from_response(resp)
        if angularaxis == "az":
            planar = np.all(np.diff(points.el) == 0)
        else:
            planar = np.all(np.diff(points.az) == 0)

        az_plot = resp.az.values
        el_plot = resp.el.values
        if not planar:
            outline = points.projection_outline(angularaxis=angularaxis)
            resp = resp[outline]
            az_plot = az_plot[outline]
            el_plot = el_plot[outline]

        angles = az_plot if angularaxis == "az" else el_plot
        perm = np.argsort(angles)
        source_angle = angles[perm]
        r = 20 * np.log10(np.abs(resp[perm].values))
        r = np.array(r.tolist() + [r[0]])
        theta = np.array(source_angle.tolist() + [source_angle[0]])
        phase = np.angle(resp[perm])
        if target is not None:
            outline = points.projection_outline(angularaxis=angularaxis)
            target_phase = np.angle(target.values[outline[perm]])
            phase = unwrap_phase_with_target(phase, target_phase)
        return dict(r=r, theta=theta), dict(x=source_angle, y=phase)

    def add_response_2d(
        self,
        fig,
        label,
        z,
        target,
        angularaxis,
        visible=True,
        xy_row=1,
        xy_col=1,
        polar_row=1,
        polar_col=2,
    ):
        for (i, resp) in enumerate((target, z)):
            if resp is None:
                continue
            is_target = i == 0

            t1, t2 = self.get_2d_updates(resp, target, angularaxis)

            fig.add_trace(
                go.Scatterpolar(
                    theta=t1["theta"],
                    thetaunit="radians",
                    r=t1["r"],
                    mode="lines",
                    line=dict(
                        color=self.name_color(label, is_target=is_target),
                        dash="dot" if is_target else "solid",
                    ),
                    name=f'|{label}{"-target" if is_target else ""}|',
                    legendgroup=label,
                    showlegend=True,
                    visible=visible,
                ),
                row=polar_row,
                col=polar_col,
            )
            fig.add_trace(
                go.Scatter(
                    x=t2["x"],
                    y=t2["y"],
                    mode="lines",
                    line=dict(
                        color=self.name_color(label, is_target=is_target),
                        dash="dot" if is_target else None,
                    ),
                    name=f'∠{label}{"-target" if is_target else ""}',
                    legendgroup=label,
                    showlegend=False,
                    visible=visible,
                ),
                row=xy_row,
                col=xy_col,
            )
            fig.update_layout(hovermode="x unified")
        return fig

    def plot_response_2d(
        self,
        y,
        y_target=None,
        label_coord=None,
        angularaxis=None,
        title=None,
        for_3d=False,
        write_html=None,
    ):
        if angularaxis is None:
            angularaxis = "az"
        if title is None:
            title = "Response"
        if label_coord is None:
            label_coord = "label"

        angulartitle = "φ" if angularaxis == "az" else "θ"
        if for_3d:
            specs = [
                [{"type": "xy"}, {"type": "scene", "rowspan": 2}],
                [{"type": "polar"}, None],
            ]
            xy_col = 1
            xy_row = 1
            polar_col = 1
            polar_row = 2
            nrow = 2
            row_heights = [0.2, 0.8]
            subplot_titles = (
                f"∠h({angulartitle})",
                f"|h(θ, φ)|",
                f"|h({angulartitle})|",
            )
        else:
            specs = [[{"type": "xy"}, {"type": "polar"}]]
            xy_col = 1
            xy_row = 1
            polar_col = 2
            polar_row = 1
            nrow = 1
            row_heights = [1.0]
            subplot_titles = (f"∠h({angulartitle})", f"|h({angulartitle})|")

        fig = make_subplots(
            rows=nrow,
            row_heights=row_heights,
            cols=2,
            subplot_titles=subplot_titles,
            specs=specs,
        )
        fig.update_xaxes(
            # title="Azimuth (φ)" if angularaxis == "az" else "Elevation (θ)",
            **angularticks(angularaxis),
            range=[0, np.pi] if angularaxis == "el" else [-np.pi, np.pi],
            row=xy_row,
            col=xy_col,
        )
        if not for_3d:
            ann = list(fig.layout.annotations)
            ann[1]["y"] = 1.1
            fig.layout.annotations = ann
        else:
            ann = list(fig.layout.annotations)
            ann[2]["y"] = 0.65
            fig.layout.annotations = ann

        if write_html:
            fig.write_html(write_html)
        return fig

    def add_response_3d(
        self,
        fig,
        label: str,
        h: xr.DataArray,
        target: Optional[xr.DataArray] = None,
        visible=True,
        showlegend=True,
        cmin=None,
        cmid=None,
        cmax=None,
        mic_offset=None,
        with_2d=False,
    ):
        intensity_name = "phase" if self.phase_as_intensity else "gain"
        for (i, resp) in enumerate((target, h)):
            if resp is None:
                continue
            is_target = True if i == 0 else False

            points = PointCloud.from_response(resp, mic_offset=mic_offset)
            triangles = points.mesh3d(
                colorscale=self.mesh_colorscale,
                showlegend=showlegend,
                visible="legendonly" if is_target else True,
                name=f'|{label}{"-target" if is_target else ""}|',
                intensity_fmt=intensity_name + ": %{intensity:.2e}",
                intensity=np.angle(resp.values)
                if self.phase_as_intensity
                else np.real(np.abs(resp.values) * np.sign(resp.values)),
                cmin=cmin,
                cmid=cmid,
                cmax=cmax,
                showscale=False,
            )
            if with_2d:
                fig.add_trace(triangles, row=1, col=2)
            else:
                fig.add_trace(triangles)

    def plot_response(
        self,
        y: xr.DataArray,
        y_target: Optional[xr.DataArray] = None,
        angularaxis=None,
        title=None,
        label_coord=None,
        label=None,
        body_trace=None,
        with_2d=True,
        with_3d=None,
        write_html=None,
    ) -> go.Figure:
        all_same_az = np.all(np.diff(y.az) == 0)
        all_same_el = np.all(np.diff(y.el) == 0)
        if self.flat is not None:
            with_3d = not self.flat
        else:
            with_3d = True

        if all_same_az or all_same_el:
            with_2d = True
            if angularaxis is None:
                angularaxis = "el" if all_same_az else "az"
            with_3d = False

        if title is None:
            title = "Response"
        if angularaxis is None:
            angularaxis = "az"

        label_dim = None
        if label_coord and label_coord in y.coords:
            label_dim = list(y[label_coord].xindexes)[0]
            y = y.set_index({label_dim: label_coord})
            if y_target is not None:
                if label_dim in y_target.dims:
                    y_target = y_target.set_index({label_dim: label_coord})

        if with_2d:
            fig = self.plot_response_2d(
                y,
                y_target,
                angularaxis=angularaxis,
                title=title,
                for_3d=with_3d,
                label_coord=label_coord,
            )
            bounds_2d = BoundsTracker()
        else:
            fig = go.Figure()

        if self.phase_as_intensity:
            intensity = np.angle(y.values)
            cmin = -np.pi
            cmax = np.pi
        else:
            intensity = np.real(np.abs(y.values) * np.sign(y.values))
            cmin = np.min(intensity)
            cmax = np.max(intensity)

        rmax = np.max(np.abs(y.values))

        bounds = BoundsTracker()

        xy_col = 1
        xy_row = 1

        if with_3d:
            polar_col = 1
            polar_row = 2
        else:
            polar_col = 2
            polar_row = 1

        # add an empty trace to show the color scale
        color_scale = dict(
            colorscale=self.mesh_colorscale,
            colorbar=polarticks(-1, 1, 4)
            if self.phase_as_intensity
            else dict(tickformat="^+11.2e"),
            color=[0],
            cmin=cmin,
            cmid=0,
            cmax=cmax,
            size=0,
            showscale=True,
        )

        if with_3d:
            empty_trace = go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                marker=color_scale,
                name="empty",
                visible=True,
                showlegend=False,
            )

            if with_2d:
                fig.add_trace(empty_trace, row=1, col=2)
            else:
                fig.add_trace(empty_trace)

            mic_offset = None
            scale = 2.5 * rmax
            if body_trace is not None:
                body_data = body_trace.to_plotly_json()
                px, py, pz = body_data["x"], body_data["y"], body_data["z"]
                maxv = max(
                    np.max(px) - np.min(px),
                    np.max(py) - np.min(py),
                    np.max(pz) - np.min(pz),
                )
                scale /= maxv
                for ax in ("x", "y", "z"):
                    body_data[ax] *= scale
                mic_offset = scale

            if with_3d:
                sources, _ = PointCloud(points_xr=y.source).scale_maxdim(
                    2.5 * rmax
                )
                source_trace = sources.scatterpolar(
                    name="sources (not to scale)", opacity=0.1, hoverinfo="skip"
                )
                if with_2d:
                    fig.add_trace(source_trace, row=1, col=2)
                else:
                    fig.add_trace(source_trace)
                source_data = source_trace.to_plotly_json()

                if body_trace is not None:
                    body_data.pop("intensity", None)
                    if with_2d:
                        fig.add_trace(body_data, row=1, col=2)
                    else:
                        fig.add_trace(body_data)

        things_to_plot = []
        if label_dim is not None:
            for l in y[label_dim]:
                things_to_plot.append((l.item(), y.sel({label_dim: l})))
        else:
            if label is None:
                label = "y"
            things_to_plot.append((label, y))
        for (label, toplot) in things_to_plot:
            band = y.band[0].item()
            target = None
            if y_target is not None:
                target = y_target.sel({"band": band})
            z = toplot.sel({"band": band})

            if with_2d:
                self.add_response_2d(
                    fig,
                    label,
                    z,
                    target,
                    angularaxis,
                    visible=True,
                    xy_row=xy_row,
                    xy_col=xy_col,
                    polar_row=polar_row,
                    polar_col=polar_col,
                )

            if with_3d:
                self.add_response_3d(
                    fig,
                    label,
                    z,
                    target,
                    visible=True,
                    showlegend=True,
                    cmin=cmin,
                    cmid=0.0,
                    cmax=cmax,
                    mic_offset=mic_offset,
                )

        steps = []
        update_keys = [
            "r",
            "theta",
            "x",
            "y",
            "z",
            "i",
            "j",
            "k",
            "vertexcolor",
            "customdata",
            "marker",
            "color",
        ]
        empty_data = dict(
            x=[0.0], y=[0.0], z=[0.0], intensity=[0.0], marker=color_scale
        )
        for (band, freq) in enumerate(y.freq):
            updates = UpdateTracker(update_keys)
            if with_3d:
                updates.push(empty_data)

            for (i, (label, toplot)) in enumerate(things_to_plot):
                if with_3d and i == 0:
                    updates.push(source_data)
                    bounds.track(**source_data)
                    if body_trace is not None:
                        updates.push(body_data)
                        bounds.track(**body_data)

                z = toplot.isel(band=band)
                target = None
                if y_target is not None:
                    target = y_target.isel(band=band)

                if with_2d:
                    if target is not None:
                        t1, t2 = self.get_2d_updates(target, None, angularaxis)
                        for t in (t1, t2):
                            updates.push(t)
                            bounds_2d.track(**t)

                    t1, t2 = self.get_2d_updates(z, target, angularaxis)
                    for t in (t1, t2):
                        updates.push(t)
                        bounds_2d.track(**t)

                if with_3d and target is not None:
                    target_update = PointCloud.from_response(
                        target, mic_offset=mic_offset
                    ).get_mesh3d_data(
                        intensity=np.angle(target.values)
                        if self.phase_as_intensity
                        else np.real(
                            np.abs(target.values) * np.sign(target.values)
                        ),
                        colorscale=self.mesh_colorscale,
                        cmin=cmin,
                        cmax=cmax,
                    )
                    bounds.track(**target_update)
                    updates.push(target_update)

                if with_3d:
                    z_update = PointCloud.from_response(
                        z, mic_offset=mic_offset
                    ).get_mesh3d_data(
                        intensity=np.angle(z.values)
                        if self.phase_as_intensity
                        else np.real(
                            np.abs(target.values) * np.sign(target.values)
                        ),
                        colorscale=self.mesh_colorscale,
                        cmin=cmin,
                        cmax=cmax,
                    )
                    # if self.phase_as_intensity:
                    #     z_update["intensity"] = np.angle(z.values)

                    bounds.track(**z_update)
                    updates.push(z_update)

            steps.append(
                dict(
                    method="update",
                    label=f"{format_freq(y.freq[band].item())}",
                    args=[updates.steps],
                )
            )

        if with_2d:
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        range=[
                            bounds_2d.extrema["r"][1] - self.polar_range_db,
                            bounds_2d.extrema["r"][1],
                        ],
                        ticksuffix="dB",
                        hoverformat="{%:.1f}",
                    ),
                    angularaxis=dict(thetaunit="radians"),
                    sector=[0, 180] if angularaxis == "el" else [0, 360],
                )
            )
            fig.update_yaxes(
                **polarticks(-4, 4),
                range=[bounds_2d.extrema["y"][0], bounds_2d.extrema["y"][1]],
                row=xy_row,
                col=xy_col,
            )

        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    y=-0.1,
                    yanchor="top",
                    pad={"t": 50},
                    currentvalue={
                        "xanchor": "left",
                    },
                    steps=steps,
                )
            ],
        )

        fig.update_layout(legend=dict(orientation="h", xanchor="left", x=0))
        if with_3d:
            xmin, xmax = bounds.extrema["x"]
            ymin, ymax = bounds.extrema["y"]
            zmin, zmax = bounds.extrema["z"]
            ax_min = min([xmin, ymin, zmin])
            ax_max = max([xmax, ymax, zmax])
            fig.update_scenes(
                dict(
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=1),
                    xaxis=dict(range=[ax_min, ax_max]),
                    yaxis=dict(range=[ax_min, ax_max]),
                    zaxis=dict(range=[ax_min, ax_max]),
                )
            )
        if write_html:
            fig.write_html(write_html, post_script=ROTATE_CAMERA_SNIPPET)
        return fig

    def plot_metrics(
        self,
        h: xr.DataArray,
        y: xr.DataArray,
        y_target: xr.DataArray,
        label_coord=None,
        angularaxis=None,
        title=None,
    ):
        if label_coord is None:
            label_coord = "strat"
        labels = np.atleast_1d(y[label_coord])
        fig = go.Figure()

        err_visible = [False] * 3 * len(labels)
        wng_visible = [False] * 3 * len(labels)
        for (i, label) in enumerate(labels):
            h_strat = h.loc[label]
            y_strat = y.loc[label]
            y_target_strat = y_target
            wng = np.sum(np.abs(y_target_strat) ** 2, axis=1) / np.sum(
                np.real(h_strat * h_strat.conj()), axis=1
            )
            wng_db = 10.0 * np.log10(wng)

            def rms_err_db(x):
                return 10.0 * np.log10(
                    np.mean(
                        (np.abs(np.abs(x) - np.abs(y_target_strat)) ** 2.0),
                        axis=1,
                    )
                )

            def max_err_db(x):
                return 20.0 * np.log10(
                    np.max(np.abs(np.abs(x) - np.abs(y_target_strat)), axis=1)
                )

            label_s = label.item()
            fig.add_trace(
                go.Scatter(
                    x=y.freq,
                    y=wng_db,
                    visible=False,
                    mode="markers+lines",
                    name="wng",
                    legendgroup=label_s,
                    legendgrouptitle_text=label_s,
                    line=dict(color=self.name_color(label_s)),
                )
            )
            wng_visible[3 * i + 0] = True

            fig.add_trace(
                go.Scatter(
                    x=y.freq,
                    y=rms_err_db(y_strat),
                    mode="markers+lines",
                    name="rms",
                    legendgroup=label_s,
                    legendgrouptitle_text=label_s,
                    line=dict(color=self.name_color(label_s)),
                ),
            )
            err_visible[3 * i + 1] = True

            fig.add_trace(
                go.Scatter(
                    x=y.freq,
                    y=max_err_db(y_strat),
                    mode="markers+lines",
                    name="max",
                    legendgroup=label_s,
                    line=dict(color=self.name_color(label_s), dash="dot"),
                ),
            )
            err_visible[3 * i + 2] = True

        fig.update_xaxes(title_text="Source Frequency (Hz)")
        fig.update_layout(hovermode="x unified")
        fig.update_layout(
            legend=dict(yanchor="bottom", y=-0.05, xanchor="left", x=1.0)
        )

        err_title = (
            f"{title} Error (all"
            f' {"elevations" if angularaxis == "el" else "azimuths"})'
        )
        wng_title = (
            f"{title} White Noise Gain (all"
            f' {"elevations" if angularaxis == "el" else "azimuths"})'
        )

        fig.update_layout(title=err_title)
        fig.update_yaxes(title="Error (dB)")

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    showactive=True,
                    xanchor="left",
                    x=0,
                    yanchor="bottom",
                    y=-0.2,
                    buttons=list(
                        [
                            dict(
                                label="Error",
                                method="update",
                                args=[
                                    {"visible": err_visible},
                                    {
                                        "title": err_title,
                                        "yaxis": {"title": "Error (dB)"},
                                    },
                                ],
                            ),
                            dict(
                                label="WNG",
                                method="update",
                                args=[
                                    {"visible": wng_visible},
                                    {
                                        "title": wng_title,
                                        "yaxis": {"title": "Gain (dB)"},
                                    },
                                ],
                            ),
                        ]
                    ),
                )
            ]
        )
        return fig

    def plot_magnitude(self, h, mic=0, in_db=False, angularaxis=None, **kwargs):
        if angularaxis is None:
            angularaxis = "az"
        if in_db:
            max_db = 20 * np.log10(np.max(np.abs(h)) + 1e-40)
            kwargs.update(
                dict(
                    zmax=max_db.item(),
                    zmin=max_db.item() - 30,
                    colorscale="viridis",
                )
            )
        else:
            kwargs.update(dict(zmid=0, colorscale="spectral"))

        if "colorbar" not in kwargs:
            kwargs["colorbar"] = dict()
        if "title" not in kwargs["colorbar"]:
            letter = "φ" if angularaxis == "az" else "θ"
            kwargs["colorbar"]["title"] = f"|h({letter})|" + (
                " (dB)" if in_db else ""
            )

        y = h.az.values if angularaxis == "az" else h.el.values
        z = np.zeros(h.shape) + np.nan
        perm = np.argsort(y)
        for i in h.band:
            hperm = h.isel(band=i)[perm]
            if in_db:
                z[i, :] = 20.0 * np.log10(np.abs(hperm) + 1e-40)
            else:
                z[i, :] = np.abs(hperm) * np.sign(np.real(hperm))
        return go.Heatmap(x=h.freq.values, y=y[perm], z=z.T, **kwargs)

    def plot_phase(self, h, angularaxis=None, **kwargs):
        if angularaxis is None:
            angularaxis = "az"
        if "colorscale" not in kwargs:
            kwargs["colorscale"] = self.phase_colorscale

        if "colorbar" not in kwargs:
            kwargs["colorbar"] = dict()

        if "zmin" not in kwargs and "zmax" not in kwargs:
            kwargs["zmin"] = -np.pi
            kwargs["zmax"] = np.pi

        if kwargs["colorbar"] is not None:
            if "title" not in kwargs["colorbar"]:
                letter = "φ" if angularaxis == "az" else "θ"
                kwargs["colorbar"]["title"] = f"∠h({letter})"

            if all(
                [
                    x not in kwargs["colorbar"]
                    for x in ["tickmode", "tickvals", "ticktext"]
                ]
            ):
                kwargs["colorbar"].update(**polarticks())

        y = h.az.values if angularaxis == "az" else h.el.values
        z = np.zeros(h.shape) + np.nan
        perm = np.argsort(y)
        for (i, f) in enumerate(h.freq):
            z[i, :] = np.angle(h[i, perm])
        return go.Heatmap(x=h.freq.values, y=y[perm], z=z.T, **kwargs)

    def plot_magnitude_and_phase(
        self,
        y,
        show_phase=False,
        label_coord=None,
        label=None,
        in_db=True,
        title=None,
        angularaxis=None,
        f_sel_hz=None,
        f_nyq_hz=None,
    ):
        if angularaxis is None:
            angularaxis = "az"

        things_to_plot = []
        if label_coord is None and "label" not in y.dims:
            things_to_plot.append(y)
            if label is None:
                label = "y"
            labels = [label]
        else:
            label_coord = "label" if label_coord is None else label_coord
            label_dim = y.coords[label_coord].dims[0]
            labels = np.atleast_1d(y[label_coord])
            for i in range(len(labels)):
                things_to_plot.append(y.isel({label_dim: i}))
        height_ratios = [1 for _ in range(len(labels))]
        nplot = len(height_ratios)
        row_heights = [ratio / sum(height_ratios) for ratio in height_ratios]

        fig = make_subplots(
            rows=nplot,
            cols=1,
            row_heights=row_heights,
            shared_xaxes=True,
            subplot_titles=labels,
            x_title="Source frequency (Hz)",
        )

        mag_visible = [False] * 2 * len(labels)
        phase_visible = [False] * 2 * len(labels)
        for (i, toplot) in enumerate(things_to_plot):
            offset = sum(row_heights[:i]) + i * 0.035 - 0.035
            mag_hm = self.plot_magnitude(
                toplot,
                in_db=in_db,
                angularaxis=angularaxis,
                showscale=i == 0,
                colorbar=dict(
                    len=sum(row_heights) - 0.01, yanchor="top", y=1 - offset
                ),
                zmid=0,
                zmax=1,
                visible=not show_phase,
            )
            phase_hm = self.plot_phase(
                toplot,
                angularaxis=angularaxis,
                showscale=i == 0,
                colorbar=dict(
                    len=sum(row_heights) - 0.01, yanchor="top", y=1 - offset
                )
                if i == 0
                else None,
                visible=show_phase,
            )
            fig.add_trace(mag_hm, row=i + 1, col=1)
            mag_visible[2 * i] = True
            fig.add_trace(phase_hm, row=i + 1, col=1)
            phase_visible[2 * i + 1] = True

        self.add_freq_lines(fig, f_sel_hz=f_sel_hz, f_nyq_hz=f_nyq_hz)

        mag_title = f"{title} Magnitude" + self.freq_suffix(
            f_sel_hz=f_sel_hz, f_nyq_hz=f_nyq_hz
        )
        phase_title = f"{title} Phase" + self.freq_suffix(
            f_sel_hz=f_sel_hz, f_nyq_hz=f_nyq_hz
        )

        fig.update_xaxes(
            range=[np.log10(y.freq[0].item()), np.log10(y.freq[-1].item())],
            type="log",
        )
        fig.update_yaxes(
            title_text="θ" if angularaxis == "el" else "φ",
            **angularticks(angularaxis),
            range=[-np.pi, np.pi - np.min(np.abs(np.diff(y.az)))]
            if angularaxis == "az"
            else [0, np.pi],
        )
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=1 if show_phase else 0,
                    showactive=True,
                    xanchor="left",
                    x=0,
                    yanchor="bottom",
                    y=-0.2,
                    buttons=list(
                        [
                            dict(
                                label="Magnitude",
                                method="update",
                                args=[
                                    {"visible": mag_visible, "title": mag_title}
                                ],
                            ),
                            dict(
                                label="Phase",
                                method="update",
                                args=[
                                    {
                                        "visible": phase_visible,
                                        "title": phase_title,
                                    }
                                ],
                            ),
                        ]
                    ),
                )
            ]
        )
        return fig


@dataclass
class PolarPlotter:
    trace_dim: str = field(default_factory=lambda: "trace")
    slider_dim: Optional[str] = None
    range_db: Optional[float] = None
    colorscale: str = field(default_factory=lambda: "Plasma")
    angularaxis: str = field(default_factory=lambda: "az")
    angularunit: str = field(default_factory=lambda: "radians")
    step_interval: int = 1

    @staticmethod
    def label_str(y, dim, i):
        val = y[dim][i].item()
        if isinstance(val, str):
            label = val
        elif isinstance(val, float):
            label = f"{val:.1f}"
            if dim == "freq":
                label = format_freq(val)
            elif dim.endswith("_deg"):
                label += "°"
        else:
            label = str(val)
        return label

    @classmethod
    def trace_kwargs(cls, y, dim, i, line_colors):
        name = f"{dim} = " + cls.label_str(y, dim, i)
        kwargs = dict(name=name)
        val = y[dim][i].item()
        # if it's a string, treat as categorical and don't apply colorscale
        if not isinstance(val, str):
            kwargs["line"] = dict(color=line_colors[i])
        return kwargs

    def make_figure(self, y: xr.DataArray, marker=None, **kwargs):
        fig = go.Figure(**kwargs)
        slider_axis = None
        if self.slider_dim:
            slider_axis = y.coords[self.slider_dim].dims[0]
        trace_axis = y.coords[self.trace_dim].dims[0]
        ntrace = y.sizes[trace_axis]
        line_colors = px.colors.sample_colorscale(
            self.colorscale, np.linspace(0, 1, ntrace, endpoint=False)
        )
        for i in range(y.sizes[trace_axis]):
            if self.slider_dim:
                z = y.isel({slider_axis: 0, trace_axis: i})
            else:
                z = y.isel({trace_axis: i})
            t = self.trace_data(z)
            angulartitle = "φ" if self.angularaxis == "az" else "θ"
            fig.add_trace(
                go.Scatterpolar(
                    theta=t["theta"],
                    thetaunit=self.angularunit,
                    r=t["r"],
                    mode="lines" if marker is None else "markers+lines",
                    marker=marker,
                    hovertemplate="r: %{r:.1f}<br>"
                    + angulartitle
                    + ": %{theta:.1f}"
                    + ("°" if self.angularunit == "degrees" else ""),
                    showlegend=True,
                    visible=True,
                    **self.trace_kwargs(y, self.trace_dim, i, line_colors),
                )
            )

        polar_tick_suffix = None
        range_kwargs = dict()
        if self.range_db is not None:
            if self.slider_dim is None:
                range_kwargs = dict(
                    range=[
                        np.max(y) - self.range_db,
                        np.max(y),
                    ]
                )
            polar_tick_suffix = "dB"
        fig.update_layout(
            polar=dict(
                radialaxis=dict(ticksuffix=polar_tick_suffix, **range_kwargs),
                angularaxis=dict(thetaunit=self.angularunit),
                sector=[0, 180] if self.angularaxis == "el" else [0, 360],
            ),
        )
        return fig

    def trace_data(self, y: xr.DataArray):
        axis = self.angularaxis + (
            "_deg" if self.angularunit == "degrees" else ""
        )
        if axis in y.coords:
            theta = np.atleast_1d(y[axis].values).flatten()
            r = np.atleast_1d(y.values).flatten()
        else:
            theta = np.atleast_1d(y.values).flatten()
            r = np.ones(len(theta))
        perm = np.argsort(theta)
        r = r[perm].tolist()
        theta = theta[perm].tolist()
        return dict(
            r=np.array(r + [r[0] if len(r) >= 1 else 0.0]),
            theta=np.array(theta + [theta[0] if len(theta) >= 1 else 0.0]),
        )

    def make_updates(self, y: xr.DataArray):
        bounds = BoundsTracker()
        steps = []
        slider_axis = y.coords[self.slider_dim].dims[0]
        trace_axis = y.coords[self.trace_dim].dims[0]
        for i in range(0, y.sizes[slider_axis], self.step_interval):
            updates = UpdateTracker(["r", "theta"])
            for j in range(y.sizes[trace_axis]):
                z = y.isel({slider_axis: i, trace_axis: j})
                t = self.trace_data(z)
                bounds.track(**t)
                updates.push(t)

            steps.append(
                dict(
                    method="update",
                    label=self.label_str(y, self.slider_dim, i),
                    args=[updates.steps],
                )
            )
        sliders = [
            dict(
                active=0,
                y=-0.1,
                yanchor="top",
                pad={"t": 50},
                currentvalue=dict(
                    xanchor="left", prefix=f"{self.slider_dim} = "
                ),
                steps=steps,
            )
        ]
        return bounds, sliders

    def apply_updates(self, fig, updates):
        bounds, sliders = updates
        polar_range = bounds.extrema["r"]
        if self.range_db is not None:
            polar_range = [
                bounds.extrema["r"][1] - self.range_db,
                bounds.extrema["r"][1],
            ]
        fig.update_layout(
            sliders=sliders,
            polar=dict(radialaxis=dict(range=polar_range)),
        )

    def plot(self, y: Union[xr.DataArray, np.ndarray], **kwargs) -> go.Figure:
        marker = None
        if isinstance(y, np.ndarray):
            if self.slider_dim and y.ndim == 1:
                y = np.expand_dims(y, axis=1)
                marker = dict(symbol=["x-thin-open", "circle"])

            if self.slider_dim:
                dims = (self.slider_dim, self.trace_dim)
                coords = {
                    self.slider_dim: range(y.shape[0]),
                    self.trace_dim: range(y.shape[1]),
                }
            else:
                dims = (self.trace_dim,)
                coords = {self.trace_dim: range(y.shape[0])}

            y = xr.DataArray(
                y, name=kwargs.get("name", "y"), dims=dims, coords=coords
            )
        fig = self.make_figure(y, marker=marker, **kwargs)
        if self.slider_dim is not None:
            updates = self.make_updates(y)
            self.apply_updates(fig, updates)
        return fig


def plot_power_vector_difference(y, slide_angle=True, angularaxis=None):
    angularaxis = "az" if angularaxis is None else angularaxis
    nangle = 60 if slide_angle else 12
    angles_deg = np.linspace(
        0, 360 if angularaxis == "az" else 180, nangle, endpoint=False
    ).tolist()

    if angularaxis == "el":
        angles_deg.append(180)

    points = PointCloud(points_xr=y)
    outline = points.projection_outline(angularaxis=angularaxis)
    y = y[outline]
    ref_dim = f"ref_{angularaxis}_deg"
    x = y.source.expand_dims({ref_dim: angles_deg})
    ref_sources = ((x.az_deg - x[ref_dim] + 360) % 360).reduce(
        np.argmin, "source"
    )
    y_ref = y.isel(source=ref_sources)
    diff = (y - y_ref).reduce(np.linalg.norm, "pv")
    # diff = y.dot(y_ref, 'pv')
    diff = diff.assign_coords(source=y.source)

    if slide_angle:
        trace_dim = "freq"
        slider_dim = ref_dim
        diff = diff.where(diff.freq < 5000, drop=True)
        colorscale = "Plasma"
    else:
        trace_dim = ref_dim
        slider_dim = "freq"
        colorscale = "HSV"

    return PolarPlotter(
        trace_dim=trace_dim,
        slider_dim=slider_dim,
        angularaxis=angularaxis,
        colorscale=colorscale,
    ).plot(diff)


@dataclass
class HeatmapSlider:
    z_sel: Dict
    slider_dim: str
    range_db: Optional[float] = None
    colorscale: str = field(default_factory=lambda: "Spectral")
    step_interval: int = 1

    @classmethod
    def trace_kwargs(cls, dim, val):
        name = f"{dim} = {val}"
        kwargs = dict(name=name)
        return kwargs

    @staticmethod
    def label_str(y, dim, i):
        val = y[dim][i].item()
        if isinstance(val, str):
            label = val
        elif isinstance(val, float):
            label = f"{val:.1f}"
            if dim == "freq":
                label = format_freq(val)
            elif dim.endswith("_deg"):
                label += "°"
        else:
            label = str(val)
        return label

    def make_figure(self, z: xr.DataArray, **kwargs):
        fig = make_subplots(
            subplot_titles=[
                f"{z_dim} = {z_val}" for (z_dim, z_val) in self.z_sel.items()
            ],
            cols=len(self.z_sel),
            **kwargs,
        )
        col = 1
        slider_axis = z.coords[self.slider_dim].dims[0]
        for z_dim, z_val in self.z_sel.items():
            zi = z.sel(
                {slider_axis: z.coords[self.slider_dim][0], z_dim: z_val}
            )
            t = self.trace_data(zi)
            fig.add_trace(
                go.Heatmap(
                    z=t["z"],
                    colorbar=dict(tickformat=" < 11.2~e"),
                    colorscale=self.colorscale,
                    zmid=0.0,
                    **self.trace_kwargs(z_dim, z_val),
                ),
                row=1,
                col=col,
            )
            col += 1
        return fig

    def trace_data(self, z: xr.DataArray):
        return dict(z=z.values)

    def make_updates(self, z: xr.DataArray):
        bounds = BoundsTracker()
        steps = []
        slider_axis = z.coords[self.slider_dim].dims[0]
        for i in range(0, z.sizes[slider_axis], self.step_interval):
            updates = UpdateTracker(["z"])
            for z_dim, z_val in self.z_sel.items():
                zi = z.sel({slider_axis: i, z_dim: z_val})
                t = self.trace_data(zi)
                bounds.track(**t)
                updates.push(t)

            steps.append(
                dict(
                    method="update",
                    label=self.label_str(z, self.slider_dim, i),
                    args=[updates.steps],
                )
            )
        sliders = [
            dict(
                active=0,
                y=-0.1,
                yanchor="top",
                pad={"t": 50},
                currentvalue=dict(
                    xanchor="left", prefix=f"{self.slider_dim} = "
                ),
                steps=steps,
            )
        ]
        return bounds, sliders

    def apply_updates(self, fig, updates):
        bounds, sliders = updates
        fig.update_layout(
            sliders=sliders,
        )

    def plot(self, z: xr.DataArray, **kwargs) -> go.Figure:
        fig = self.make_figure(z, **kwargs)
        updates = self.make_updates(z)
        self.apply_updates(fig, updates)
        return fig


@dataclass
class VolumeSlider:
    colorscale: str = field(default_factory=lambda: "Plasma")
    slider_dim: str = field(default_factory=lambda: "frame")
    step_interval: int = 1
    decimate_by: int = 5
    surface_count: int = 7
    opacityscale: str = field(default_factory=lambda: "extremes")
    opacity: float = 0.2
    caps: dict = field(
        default_factory=lambda: dict(x_show=False, y_show=False, z_show=False)
    )

    @staticmethod
    def label_str(y, dim, i):
        val = y[dim][i].item()
        if isinstance(val, str):
            label = val
        elif isinstance(val, float):
            label = f"{val:.1f}"
            if dim == "freq":
                label = format_freq(val)
            elif dim.endswith("_deg"):
                label += "°"
        else:
            label = str(val)
        return label

    def make_figure(self, v: xr.DataArray, **kwargs):
        t = self.trace_data(v.isel({self.slider_dim: 0}))
        fig = go.Figure()
        fig.add_trace(
            go.Volume(
                colorbar=dict(tickformat="^+11.2e"),
                colorscale=self.colorscale,
                opacity=self.opacity,
                opacityscale=self.opacityscale,
                surface_count=self.surface_count,
                caps=self.caps,
                **t,
                **kwargs,
            ),
        )
        return fig

    def trace_data(self, v: xr.DataArray):
        v = v.isel(
            x=range(0, len(v.x), self.decimate_by),
            y=range(0, len(v.y), self.decimate_by),
            z=range(0, len(v.z), self.decimate_by),
        )
        x, y, z = np.mgrid[0 : len(v.x), 0 : len(v.y), 0 : len(v.z)]
        return dict(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=v.values.flatten(),
        )

    def make_updates(self, v: xr.DataArray):
        bounds = BoundsTracker()
        steps = []
        slider_axis = v.coords[self.slider_dim].dims[0]
        for i in range(0, v.sizes[slider_axis], self.step_interval):
            updates = UpdateTracker(["value"])
            vi = v.sel({slider_axis: i})
            t = self.trace_data(vi)
            bounds.track(**t)
            updates.push(t)
            steps.append(
                dict(
                    method="restyle",
                    label=self.label_str(v, self.slider_dim, i),
                    args=[updates.steps],
                )
            )
        sliders = [
            dict(
                active=0,
                y=-0.1,
                yanchor="top",
                pad={"t": 50},
                currentvalue=dict(
                    xanchor="left", prefix=f"{self.slider_dim} = "
                ),
                steps=steps,
            )
        ]
        return bounds, sliders

    def apply_updates(self, fig, updates):
        bounds, sliders = updates
        # cmin, cmax = bounds.extrema["value"]
        # fig.update_traces(cmin=cmin, cmax=cmax, overwrite=True)
        fig.update_layout(
            sliders=sliders,
        )

    def plot(self, v: xr.DataArray, **kwargs) -> go.Figure:
        fig = self.make_figure(v, **kwargs)
        updates = self.make_updates(v)
        self.apply_updates(fig, updates)
        return fig


if __name__ == "__main__":
    from spatpy.ambisonics import AmbisonicChannelFormat
    from spatpy import placement

    sources = placement.circular_array(1000)
    # y = AmbisonicChannelFormat("BF2H").azel_to_pan(sources.az, sources.el)
    # fig = PolarPlotter(trace_dim="dlb_name").plot(y)
    # fig.show()
    y = SpeakerChannelFormat("5.0").azel_to_pan(sources.az, sources.el)
    fig = PolarPlotter(trace_dim="dlb_name").plot(np.abs(y))
    fig.show()
    print()
