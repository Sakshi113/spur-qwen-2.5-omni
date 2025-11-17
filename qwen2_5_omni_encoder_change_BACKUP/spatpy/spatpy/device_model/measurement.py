import xarray as xr
import numpy as np
from pathlib import Path
import soundfile as sf


class DeviceMeasurementDir():
    def __init__(self, measurement_dir):
        self.measurement_dir = Path(measurement_dir)
        assert self.measurement_dir.exists(), f'"{self.measurement_dir}" not found'
        self.files = list(self.measurement_dir.glob('*.wav'))
        #self._parse()

    """ Must accept a filename and return a dictionary representing the metadata
        for that file. The keys returned from this must be the same for all filenames

        'device' is a special key on return. Data with the same 'device' will be loaded
        into a single xarray.DataArray
    """
    def _dimensions_from_filename(self, filename):
        raise NotImplementedError

    def _get_dim_key(self, dims):
        return tuple([dims[key] if key in dims else None for key in self.dimension_keys])

    """ Must be a stable ordered set of dimension keys """
    @property
    def dimension_keys(self):
        return list(self.dimensions.keys())

    def get_coords(self, **selector_dict):
        coords = {}
        for filename in self.files:
            dims = self._dimensions_from_filename(filename)
            if not self._dims_match_selector(dims, selector_dict):
                continue
            for (dim_name, dim_val) in dims.items():
                coords[dim_name] = coords.get(dim_name, [])
                if dim_val not in coords[dim_name]:
                    coords[dim_name].append(dim_val)
        coords = {key: sorted(val) for key, val in coords.items()}
        return coords

    def get_file_list(self, **selector_dict):
        files = [
            filename for filename in self.files
            if self._dims_match_selector(self._dimensions_from_filename(filename), selector_dict)]
        return files

    def _dims_match_selector(self, dims, selector_dict):
        match = True
        for (dim_name, matches) in selector_dict.items():
            matches = matches if isinstance(matches, list) else [matches]
            match &= dim_name in dims and dims[dim_name] in matches
        return match

    def load_irs(self, **selector_dict):
        selector_dict.update(dict(type='ir'))
        return self.load(**selector_dict)
    
    def load_recordings(self, **selector_dict):
        selector_dict.update(dict(type='recording'))
        return self.load(**selector_dict)

    def load(self, **selector_dict) -> xr.DataArray:
        coords = self.get_coords(**selector_dict)
        if len(coords) == 0:
            return {}
        devices = coords['device']
        
        dset = {}
        for device in devices:
            selector_dict.update(dict(device=device))
            # Get max length, samplerate and nchannels while checking for consistency
            max_length = 0
            fs = None
            nchannels = None
            for filename in self.get_file_list(**selector_dict):
                info = sf.info(filename)
                max_length = max(max_length, info.frames)
                assert fs == None or fs == info.samplerate
                assert nchannels == None or nchannels == info.channels
                fs = info.samplerate
                nchannels = info.channels

            coords = self.get_coords(**selector_dict)
            shape = [len(coords[dim]) for dim in coords] + [max_length, nchannels]
            coords['time'] = np.arange(max_length)/fs
            if len(self.mic_order) == nchannels:
                coords['channel'] = self.mic_order
            else:
                coords['channel'] = np.arange(nchannels)
            xdata = xr.DataArray(
                np.zeros(shape=shape, dtype=np.float32),
                coords=coords
            )

            for filename in self.get_file_list(**selector_dict):
                dims = self._dimensions_from_filename(filename)
                info = sf.info(filename)
                data = sf.read(filename, always_2d=True)[0]
                xdata.isel(time=slice(0, info.frames)).isel(channel=slice(0, info.channels)).loc[dims] = data
            
            if 'angle_deg' in xdata.coords:
                xdata = xdata.sortby('angle_deg')

            # Trim dimensions that were filtered for only one item (not in a list)
            xdata = xdata.sel(selector_dict, drop=True)

            dset[device] = xdata
        
        return dset


class CaptureShareMeasurementDir(DeviceMeasurementDir):
    mic_order = ['bottom', 'top', 'back']
    
    def _dimensions_from_filename(self, filename):
        """ Must accept a filename and return a dictionary representing the metadata
        for that file. The keys returned from this must be the same for all filenames

        'device' is a special key on return. Data with the same 'device' will be loaded
        into a single xarray.DataArray
        """
        dims = {}
        parts = filename.stem.split('_')
        if parts[-1].lower() == 'ir':
            dims['type'] = parts.pop(-1).lower()
        else:
            dims['type'] = 'recording'
        
        if parts[-1] in ['occluded']:
            dims['occlusion'] = parts.pop(-1)
        else:
            dims['occlusion'] = 'none'

        if parts[-1] in ['portrait', 'landscape']:
            dims['orientation'] = parts.pop(-1)
        
        if 'deg' in parts[-1]:
            angle = float(parts.pop(-1)[:-3])
            direction = parts.pop(-1)
            angle = -angle if direction == 'right' else angle
            angle = ((angle + 180) % 360) - 180
            dims['angle_deg'] = angle

        if parts[-1] == 'handheld':
            dims['condition'] = parts.pop(-1)
        else:
            dims['condition'] = 'free'

        dims['device'] = '_'.join(parts)

        return dims