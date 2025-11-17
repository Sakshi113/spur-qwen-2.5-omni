import os
from urllib.request import urlretrieve
from spatpy.io import read_wav_file, write_wav_file
from zipfile import ZipFile
from pathlib import Path
import tempfile
import shutil
from subprocess import check_call
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
from enum import IntEnum, Enum

LC3_SOURCE_ZIP_URL = "https://www.etsi.org/deliver/etsi_ts/103600_103699/103634/01.03.01_60/ts_103634v010301p0.zip"

SPATPY_DATA_PATH = Path(__file__).parent / "data"
LC3_SOURCE_LOCAL_PATH = SPATPY_DATA_PATH / "lc3"

LC3_FIXED_POINT_BINARY = (
    LC3_SOURCE_LOCAL_PATH / "src" / "fixed_point" / "LC3plus"
)
LC3_FLOATING_POINT_BINARY = (
    LC3_SOURCE_LOCAL_PATH / "src" / "floating_point" / "LC3plus"
)


def download_lc3(url=None):
    if url is None:
        url = LC3_SOURCE_ZIP_URL

    zip_local_path = SPATPY_DATA_PATH / Path(url).name
    tmp_dir = tempfile.mkdtemp()
    try:
        if not zip_local_path.exists():
            urlretrieve(url, zip_local_path)
        with ZipFile(zip_local_path, "r") as zf:
            source_dir = [
                zi.filename
                for zi in zf.infolist()
                if zi.is_dir() and len(zi.filename.split("/")) == 3
            ][0]
            zf.extractall(
                path=tmp_dir,
                members=[
                    name
                    for name in zf.namelist()
                    if name.startswith(source_dir)
                ],
            )
            if LC3_SOURCE_LOCAL_PATH.exists():
                shutil.rmtree(LC3_SOURCE_LOCAL_PATH)
            shutil.move(
                os.path.join(tmp_dir, source_dir), LC3_SOURCE_LOCAL_PATH
            )
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def build_lc3(fixed_point: bool = False):
    if not LC3_SOURCE_LOCAL_PATH.exists():
        download_lc3()
    bin_path = (
        LC3_FIXED_POINT_BINARY if fixed_point else LC3_FLOATING_POINT_BINARY
    )
    if not bin_path.exists():
        check_call(["make"], cwd=bin_path.parent)
    return bin_path


class LC3DelayCompensation(IntEnum):
    DISABLED = 0
    """Don't use delay compensation"""

    DECODER = 1
    """Compensate delay in decoder"""

    SPLIT = 2
    """Split delay equally in encoder and decoder"""


class LC3ErrorProtection(IntEnum):
    DISABLED = 0
    """Error protection disabled"""

    MINIMUM = 1
    """Minimum error protection, detection only"""

    MODERATE = 2
    """Moderate error protection"""

    STRONG = 3
    """Strong error protection"""

    MAXIMUM = 4
    """Maximum error protection"""


@dataclass
class LC3Frontend:
    bitrate: Union[int, str]
    fixed_point: bool = False
    bps: Optional[int] = None
    swf: Optional[str] = None
    dc: Optional[LC3DelayCompensation] = None
    frame_ms: Optional[float] = None
    bandwidth: Optional[Union[int, str]] = None
    frame_counter_disable: bool = True
    verbose: bool = False
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    format_g192: bool = False
    cfg_g192: Optional[str] = None
    epf: Optional[str] = None
    ept: bool = False
    edf: Optional[str] = None
    epmode: Optional[Union[LC3ErrorProtection, str]] = None
    ep_dbg: Optional[str] = None
    hrmode: bool = False
    lfe: bool = False

    @property
    def configuration_name(self):
        ms = 10.0 if self.frame_ms is None else self.frame_ms
        return f"lc3-{ms:.1f}ms-{self.bitrate // 1000}kbps-per-ch"

    @property
    def bin_path(self):
        return build_lc3(fixed_point=self.fixed_point)

    @property
    def flags(self):
        args = []
        if self.bps:
            args.extend(["-bps", str(self.bps)])
        if self.swf:
            args.extend(["-swf", self.swf])
        if self.dc:
            args.extend(["-dc", str(self.dc.value)])
        if self.frame_ms:
            args.extend(["-frame_ms", f"{self.frame_ms:.1f}"])
        if self.frame_counter_disable:
            args.extend(["-q"])
        if self.verbose:
            args.extend(["-v"])
        if self.start_frame:
            args.extend(["-y", str(self.start_frame)])
        if self.end_frame:
            args.extend(["-z", str(self.end_frame)])
        if self.bandwidth:
            args.extend(["-bandwidth", str(self.bandwidth)])
        if self.format_g192:
            args.extend(["-formatG192"])
        if self.cfg_g192:
            args.extend(["-cfgG192", self.cfg_g192])
        if self.epf:
            args.extend(["-epf", self.epf])
        if self.ept:
            args.extend(["-ept"])
        if self.edf:
            args.extend(["-edf", self.edf])
        if self.epmode:
            if isinstance(self.epmode, str):
                args.extend(["-epmode", self.epmode])
            else:
                assert isinstance(self.epmode, LC3ErrorProtection)
                args.extend(["-epmode", str(self.epmode.value)])
        if self.ep_dbg:
            args.extend(["-ep_dbg", self.ep_dbg])
        if self.hrmode:
            args.extend(["-hrmode"])
        if self.lfe:
            args.extend(["-lfe"])
        return args

    def __call__(self, in_filename: str, out_filename: str):
        cmd = [str(self.bin_path)] + self.flags + [
            in_filename,
            out_filename,
            str(self.bitrate),
        ]
        check_call(cmd)

    def mono_roundtrip(self, x: np.array, fs: float):
        _, tmp_in = tempfile.mkstemp(suffix=".wav")
        _, tmp_out = tempfile.mkstemp(suffix=".wav")
        write_wav_file(tmp_in, x, fs)
        try:
            self(tmp_in, tmp_out)
            y, _ = read_wav_file(tmp_out)
        finally:
            os.remove(tmp_in)
            os.remove(tmp_out)
        return y

    def roundtrip(
        self, x: np.ndarray, fs: float, channel_axis: Optional[int] = None
    ):
        if channel_axis is None:
            channel_axis = -1
        if channel_axis < 0:
            channel_axis += x.ndim
        x = np.moveaxis(x, channel_axis, 1)
        ys = []
        for ch in range(x.shape[-1]):
            ys.append(self.mono_roundtrip(x[:, ch], fs))
        y = np.vstack(ys).T
        if channel_axis is not None:
            y = np.moveaxis(y, 1, channel_axis)
        return y.astype(x.dtype)
