from spatpy.speakers import ChannelFormat, SpeakerChannelFormat
from spatpy.ambisonics import AmbisonicChannelFormat, PlanarChannelFormat
from spatpy.beehive import BeehiveChannelFormat
from typing import Union
import numpy as np
import xarray as xr
from enum import Enum


class ChannelFormatKind(Enum):
    BEEHIVE = BeehiveChannelFormat
    SPEAKER = SpeakerChannelFormat
    AMBISONIC = AmbisonicChannelFormat
    PLANAR = PlanarChannelFormat


def parse_channel_format(s):
    fmt_name = None
    kwargs = None
    for (name, fmt) in ChannelFormatKind.__members__.items():
        kwargs = fmt.value.parse(s)
        if kwargs:
            fmt_name = name
            break
    return (fmt_name, kwargs)


def string_to_channel_format(s):
    fmt, kwargs = parse_channel_format(s)
    return ChannelFormatKind[fmt].value(**kwargs)


def mix_matrix(
    src: Union[str, ChannelFormat], dst: Union[str, ChannelFormat], **kwargs
):
    if isinstance(src, str):
        src = string_to_channel_format(src)
    if isinstance(dst, str):
        dst = string_to_channel_format(dst)
    if src.canonical_name == dst.canonical_name:
        return np.eye(src.nchannel)
    for (name, cls) in ChannelFormatKind.__members__.items():
        if isinstance(dst, cls.value):
            fn_name = f"{name.lower()}_mix_matrix"
            if not hasattr(src, fn_name):
                return None
            fn = getattr(src, fn_name)
            return fn(dst, **kwargs)
    return None


def apply_mix(y, target_format, mix=None):
    if isinstance(target_format, str):
        target_format = string_to_channel_format(target_format)
    src_format = string_to_channel_format(y.format)
    if mix is None:
        mix = mix_matrix(src_format, target_format)
    extra_dims = tuple([d for d in y.dims if d != y.format])
    new_dims = (y.format,) + extra_dims
    y = y.transpose(*new_dims)
    x = mix @ y.values
    coords = target_format.coords
    coords.update(y.source.coords)
    return xr.DataArray(
        x,
        dims=(target_format.dim_name,) + extra_dims,
        coords=coords,
        attrs=dict(format=target_format.short_name),
    )


if __name__ == "__main__":
    A = mix_matrix("HOA1S", "5.1L")
    print(A)
