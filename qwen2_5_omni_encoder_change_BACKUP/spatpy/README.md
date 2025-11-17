# spatpy

**[View the documentation](https://devpi.dolby.net/capture/main/spatpy/latest/+doc/index.html)**

To install from devpi:
```
pip3 install spatpy[extras,ufb_banding] --extra-index https://devpi.dolby.net/capture/main
```

`extras` and `ufb_banding` install optional dependencies and may be omitted. See `pyproject.toml` in the top-level directory for details.

To develop:
```
pip3 install -e git+ssh://git@gitlab-sfo.dolby.net/capture/spatpy.git@main#egg=spatpy --src .
```

## LC3 source code
Available from [here](https://www.etsi.org/deliver/etsi_ts/103600_103699/103634/01.03.01_60/ts_103634v010301p0.zip).
