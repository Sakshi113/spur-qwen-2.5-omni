from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import re
from typing import List, Dict
import numpy as np
from dataclasses import dataclass

from spatpy.device_model import DeviceModel

COMSOL_DEVICE_DEFAULT_VERSIONS = {
    "find_x2": "2021-08-11",
    "find_x3_pro": "2021-11-29",
}

COMSOL_FIND_X2_MIC_ORDER = ["top", "bottom", "back"]

COMSOL_FIND_X3_PRO_MIC_ORDER = ["back", "top", "bottom"]

COMSOL_SIMULATIONS = {
    "find_x2": {
        "2021-08-11": {
            "mic_file_patterns": [f"p_m{i + 1}_*r_10*.txt" for i in range(3)],
        }
    },
    "find_x3_pro": {
        "2021-11-29": {
            "mic_file_patterns": [
                f"mic_{COMSOL_FIND_X3_PRO_MIC_ORDER[i]}_oppo_find_x3_pro.txt"
                for i in range(3)
            ]
        },
    },
}


# note: simulations have y/x coordinates flipped
COMSOL_DEVICE_GEOMETRY = {
    "find_x2": DeviceModel(
        name="find_x2",
        height=164.9,
        width=74.5,
        depth=8.0,
        mic_order=COMSOL_FIND_X2_MIC_ORDER,
        mic_position=dict(
            top=(20.8, 0.0, 4.0),
            bottom=(27.3, 164.9, 4.6),
            back=(53.4, 24.7, 10.0),
        ),
    ),
    "find_x3_pro": DeviceModel(
        name="find_x3_pro",
        height=163.6,
        width=74.0,
        depth=8.26,
        mic_order=COMSOL_FIND_X3_PRO_MIC_ORDER,
        mic_position=dict(
            back=(47.5, 23.7, 9.7),
            top=(62.2, 0.0, 3.6),
            bottom=(25.0, 163.6, 4.1),
        ),
    ),
}

COMSOL_DEVICE_NAMES = list(COMSOL_DEVICE_GEOMETRY.keys())


@dataclass
class ComsolTable:
    name: str
    metadata: Dict
    df: pd.DataFrame

    @classmethod
    def from_file(cls, filename):
        with open(filename, "rt") as fobj:
            lines = fobj.readlines()
        i = 0
        metadata = {}
        while i < len(lines) and lines[i].startswith("% "):
            key, value = lines[i][2:].strip().split(":", 1)
            if key == "Table":
                i += 1
                columns = re.split("\s{2,}", lines[i][2:].strip())
                scalars = []
                units = {}
                for c in columns:
                    m = re.match("^(.*?), Point: ", c)
                    if m:
                        coords = []
                        for s in c.split(m.group(0))[1:]:
                            xs = []
                            for x in s.lstrip("(").rstrip(")").split(", "):
                                v = float(x)
                                if abs(v) < 1e-8:
                                    v = 0.0
                                xs.append(v)
                            coords.append(tuple(xs))
                        units["pressure"] = m.group(1)
                    else:
                        name = c
                        unit = "unknown"
                        if " " in c:
                            name, unit = c.split(" ")
                            unit = unit.lstrip("(").rstrip(")")
                        scalars.append(name)
                        units[name] = unit
            metadata[key] = value.lstrip()
            i += 1

        metadata["Units"] = units
        points = []
        while i < len(lines):
            values = re.split("\s{2,}", lines[i].strip())
            pressures = [
                np.complex128(s.replace("i", "j"))
                for s in values[len(scalars) :]
            ]
            for (coord, press) in zip(coords, pressures):
                (x, y, z) = coord
                # transpose x and y to match our convention
                points.append(
                    (
                        y,
                        x,
                        z,
                        *[float(x) for x in values[: len(scalars)]],
                        press,
                    )
                )
            i += 1

        df = pd.DataFrame.from_records(
            points, columns=tuple(["x", "y", "z"] + scalars + ["pressure"])
        )

        return cls(filename.stem, metadata, df)


@dataclass
class ComsolModel:
    device: str
    version: str
    mics: List[ComsolTable]
    geometry: DeviceModel

    @classmethod
    def from_simulations_dir(
        cls, device, simulations_dir, device_version=None
    ):
        if device_version is None:
            device_version = COMSOL_DEVICE_DEFAULT_VERSIONS.get(device)

        p = Path(simulations_dir) / device / device_version
        assert p.exists()
        mics = []
        simulation = COMSOL_SIMULATIONS[device][device_version]
        for mic_pattern in simulation["mic_file_patterns"]:
            mic_file = list(p.glob(mic_pattern))[0]
            mics.append(ComsolTable.from_file(mic_file))
        geometry = COMSOL_DEVICE_GEOMETRY.get(device)
        return cls(device, device_version, mics, geometry)

    @property
    def body_trace(self):
        return self.geometry.body_trace
