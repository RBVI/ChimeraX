from typing import Optional

import numpy as np

from chimerax.map.volume import (
    set_initial_region_and_style, show_volume_dialog
    , MultiChannelSeries, MapChannelsModel, default_settings, set_data_cache
    , data_already_opened, any_volume_open, _reset_color_sequence
    , set_initial_volume_color, Volume
)

from ..types import Axis

from . import modality
from .dicom_models import DicomGrid

class DICOMVolume(Volume):
    def __init__(self, session, grid_data, rendering_options = None):
        Volume.__init__(self, session, grid_data, rendering_options=rendering_options)
        self.active = False

    def is_segmentation(self):
        try:
            return self.data.dicom_data.modality == modality.Segmentation
        except AttributeError:
            # TODO: This is a hack to get around the fact that there is no DICOM data
            # in a segmentation
            return True
        
    def copy(self):
        v = dicom_volume_from_grid_data(self.data, self.session, open_model = False, style = None, show_dialog = False)
        v.copy_settings_from(self)
        return v

    def set_2d_segment_data(self, axis: Axis, slice: int, positions, value: int, min_threshold: Optional[float] = None, max_threshold: Optional[float] = None) -> None:
        for position in positions:
            center_x, center_y, radius = position
            self.set_data_in_puck(axis, slice, round(center_x), round(center_y), radius, value, min_threshold, max_threshold)
        self.data.values_changed()

    def _sphere_grid_bounds(self, center, radius):
        ijk_center = self.data.xyz_to_ijk(center)
        spacings = self.data.plane_spacings()
        ijk_size = [radius/s for s in spacings]
        from math import floor, ceil
        ijk_min = [max(int(floor(c-s)), 0) for c,s in zip(ijk_center,ijk_size)]
        ijk_max = [min(int(ceil(c+s)), m-1) for c, s, m in zip(ijk_center, ijk_size, self.data.size)]
        return ijk_min, ijk_max

    def set_sphere_data(self, center: tuple, radius: int, value: int, min_threshold: Optional[float] = None, max_threshold: Optional[float] = None) -> None:
        # Optimization: Mask only subregion containing sphere.
        ijk_min, ijk_max = self._sphere_grid_bounds(center, radius)
        from chimerax.map_data import GridSubregion, zone_mask
        subgrid = GridSubregion(self.data, ijk_min, ijk_max)
        reference_subgrid = GridSubregion(self.data.reference_data, ijk_min, ijk_max)

        if min_threshold and max_threshold:
            mask = zone_mask_clamped_by_referenced_grid(subgrid, reference_subgrid, [center], radius, min_threshold, max_threshold)
        else:
            mask = zone_mask(subgrid, [center], radius)

        dmatrix = subgrid.full_matrix()

        from numpy import putmask
        putmask(dmatrix, mask, value)

        self.data.values_changed()

    def set_data_in_puck(self, axis: Axis, slice_number: int, left_offset: int, bottom_offset: int, radius: int, value: int, min_threshold: Optional[float] = None, max_threshold: Optional[float] = None) -> None:
        # TODO: if not segmentation, refuse
        # TODO: Preserve the happiest path. If the radius of the segmentation overlay is
        #  less than the radius of one voxel, there's no need to go through all the rigamarole.
        #  grid.data.segment_array[slice][left_offset][bottom_offset] = 1
        x_max, y_max, z_max = self.data.size
        x_step, y_step, z_step = self.data.step
        if not min_threshold:
            min_threshold = float('-inf')
        if not max_threshold:
            max_threshold = float('inf')
        if axis == Axis.AXIAL:
            slice = self.data.pixel_array[slice_number]
            reference_slice = self.data.reference_data.pixel_array[slice_number]
            vertical_max = y_max - 1
            vertical_step = y_step
            horizontal_max = x_max - 1
            horizontal_step = x_step
        elif axis == Axis.CORONAL:
            slice = self.data.pixel_array[:, slice_number, :]
            reference_slice = self.data.reference_data.pixel_array[:, slice_number, :]
            vertical_max = z_max - 1
            vertical_step = z_step
            horizontal_max = x_max - 1
            horizontal_step = x_step
        else:
            slice = self.data.pixel_array[:, :, slice_number]
            reference_slice = self.data.reference_data.pixel_array[:, :, slice_number]
            vertical_max = z_max - 1
            vertical_step = z_step
            horizontal_max = y_max - 1
            horizontal_step = y_step
        scaled_radius = round(radius / horizontal_step)
        x = 0
        y = round(radius)
        d = 1 - y
        while y > x:
            if d < 0:
                d += 2 * x + 3
            else:
                d += 2 * (x - y) + 5
                y -= 1
            x += 1
            scaled_horiz_x = round(x / horizontal_step)
            scaled_vert_x = round(x / vertical_step)
            scaled_horiz_y = round(y / horizontal_step)
            scaled_vert_y = round(y / vertical_step)
            x_start = round(max(left_offset - scaled_horiz_x, 0))
            x_end = round(min(left_offset + scaled_horiz_x, horizontal_max - 1))
            y_start = round(max(bottom_offset - scaled_vert_y, 0))
            y_end = round(min(bottom_offset + scaled_vert_y, vertical_max))
            mask = np.where(reference_slice[y_start, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_start, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_start][x_start:x_end][np.where(mask == 1)] = value
            mask = np.where(reference_slice[y_start+1, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_start+1, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_start+1][x_start:x_end][np.where(mask == 1)] = value
            mask = np.where(reference_slice[y_end, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_end, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_end][x_start:x_end][np.where(mask == 1)] = value
            mask = np.where(reference_slice[y_end-1, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_end-1, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_end-1][x_start:x_end][np.where(mask == 1)] = value
               #slice[y_end][x_start:x_end][np.where(min_threshold <= reference_slice <= max_threshold)] = value
            #for i in range(x_start, x_end):
            ##    if min_threshold <= reference_slice[y_start][i] <= max_threshold:
            ##        slice[y_start][i] = value
            #    if min_threshold <= reference_slice[y_end][i] <= max_threshold:
            #        slice[y_end][i] = value
            #    # Try to account for the fact that with spacings < 1 some lines get skipped, even if it
            #    # causes redundant writes
            #    if min_threshold <= reference_slice[y_start+1][i] <= max_threshold:
            #        slice[y_start+1][i] = value
            #    if min_threshold <= reference_slice[y_end-1][i] <= max_threshold:
            #        slice[y_end-1][i] = value
            x_start = round(max(left_offset - scaled_horiz_y, 0))
            x_end = round(min(left_offset + scaled_horiz_y, horizontal_max))
            y_start = round(max(bottom_offset - scaled_vert_x, 0))
            y_end = round(min(bottom_offset + scaled_vert_x, vertical_max))
            mask = np.where(reference_slice[y_start, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_start, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_start][x_start:x_end][np.where(mask == 1)] = value
            mask = np.where(reference_slice[y_start+1, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_start+1, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_start+1][x_start:x_end][np.where(mask == 1)] = value
            mask = np.where(reference_slice[y_end, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_end, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_end][x_start:x_end][np.where(mask == 1)] = value
            mask = np.where(reference_slice[y_end-1, x_start:x_end] <= max_threshold, 1, 0)
            mask &= np.where(reference_slice[y_end-1, x_start:x_end] >= min_threshold, 1, 0)
            slice[y_end-1][x_start:x_end][np.where(mask == 1)] = value

            #for i in range(x_start, x_end):
            #    if min_threshold <= reference_slice[y_start][i] <= max_threshold:
            #        slice[y_start][i] = value
            #    if min_threshold <= reference_slice[y_end][i] <= max_threshold:
            #        slice[y_end][i] = value
            #    # Try to account for the fact that with spacings < 1 some lines get skipped, even if it
            #    # causes redundant writes
            #    if min_threshold <= reference_slice[y_start+1][i] <= max_threshold:
            #        slice[y_start+1][i] = value
            #    if min_threshold <= reference_slice[y_end-1][i] <= max_threshold:
            #        slice[y_end-1][i] = value
            #slice[y_start + 1][x_start:x_end] = value
            #slice[y_end - 1][x_start:x_end] = value
        mask = np.where(reference_slice[bottom_offset, left_offset - scaled_radius:left_offset + scaled_radius] <= max_threshold, 1, 0)
        mask &= np.where(reference_slice[bottom_offset, left_offset - scaled_radius:left_offset + scaled_radius] >= min_threshold, 1, 0)
        #for i in range(left_offset - scaled_radius, left_offset + scaled_radius):
        #    if min_threshold <= reference_slice[bottom_offset][i] <= max_threshold:
        #        slice[bottom_offset][i] = value
        slice[bottom_offset][left_offset - scaled_radius:left_offset + scaled_radius][np.where(mask == 1)] = value

    def set_step(self, step: int) -> None:
        ijk_min = self.region[0]
        ijk_max = self.region[1]
        ijk_step = [step, step, step]
        self.new_region(ijk_min, ijk_max, ijk_step, adjust_step = False)

    def segment(self, number):
        new_grid = DicomGrid(
            None, self.data.size, 'uint8'
            , self.data.origin, self.data.step, self.data.rotation
            , "", name = "segmentation %d" % number, time = None, channel = None
        )
        new_grid.reference_data = self.data
        new_grid.initial_plane_display = False
        new_seg_model = open_dicom_grids(self.session, [new_grid], name = "new segmentation")[0]
        self.session.models.add(new_seg_model)
        return new_seg_model[0]

def dicom_volume_from_grid_data(grid_data, session, style = 'auto',
                          open_model = True, model_id = None, show_dialog = True):
    '''
    Supported API.
    Create a new :class:`.Volume` model from a :class:`~.data.GridData` instance and set its initial
    display style and color and add it to the session open models.

    Parameters
    ----------
    grid_data : :class:`~.data.GridData`
      Use this GridData to create the Volume.
    session : :class:`~chimerax.core.session.Session`
      The session that the Volume will belong to.
    style : 'auto', 'surface', 'mesh' or 'image'
      The initial display style.
    open_model : bool
      Whether to add the Volume to the session open models.
    model_id : tuple of integers
      Model id for the newly created Volume.
      It is an error if the specifid id equals the id of an existing model.
    show_dialog : bool
      Whether to show the Volume Viewer user interface panel.

    Returns
    -------
    volume : the created :class:`.Volume`
    '''

    set_data_cache(grid_data, session)

    ds = default_settings(session)
    ro = ds.rendering_option_defaults()
    if getattr(grid_data, 'polar_values', None):
      ro.flip_normals = True
      ro.cap_faces = False
    if hasattr(grid_data, 'initial_rendering_options'):
      for oname, ovalue in grid_data.initial_rendering_options.items():
        setattr(ro, oname, ovalue)

    # Create volume model
    d = data_already_opened(grid_data.path, grid_data.grid_id, session)
    if d:
      grid_data = d

    v = DICOMVolume(session, grid_data, rendering_options = ro)

    # Set display style
    if style == 'auto':
      # Show single plane data in image style.
      single_plane = [s for s in grid_data.size if s == 1]
      style = 'image' if single_plane else 'surface'
    if style is not None:
      v._style_when_shown = style

    if grid_data.rgba is None:
      if not any_volume_open(session):
        _reset_color_sequence(session)
      set_initial_volume_color(v, session)

    if not model_id is None:
      if session.models.have_id(model_id):
        from chimerax.core.errors import UserError
        raise UserError('Tried to create model #%s which already exists'
                        % '.'.join('%d'%i for i in model_id))

      v.id = model_id

    if open_model:
      session.models.add([v])

    if show_dialog:
      show_volume_dialog(session)

    return v


def open_dicom_grids(session, grids, name, **kw):
    if kw.get('polar_values', False):
        for g in grids:
            g.polar_values = True
        if g.rgba is None:
            g.rgba = (0,1,0,1) # Green

    channel = kw.get('channel', None)
    if channel is not None:
        for g in grids:
            g.channel = channel

    series = kw.get('vseries', None)
    if series is not None:
        if series:
            for i,g in enumerate(grids):
                if tuple(g.size) != tuple(grids[0].size):
                    gsizes = '\n'.join((g.name + (' %d %d %d' % g.size)) for g in grids)
                    from chimerax.core.errors import UserError
                    raise UserError('Cannot make series from volumes with different sizes:\n%s' % gsizes)
                g.series_index = i
        else:
            for g in grids:
                if hasattr(g, 'series_index'):
                    delattr(g, 'series_index')

    maps = []
    if 'show' in kw:
        show = kw['show']
    else:
        show = (len(grids) >= 1 and getattr(grids[0], 'show_on_open', True))
    si = [d.series_index for d in grids if hasattr(d, 'series_index')]
    is_series = (len(si) == len(grids) and len(set(si)) > 1)
    cn = [d.channel for d in grids if d.channel is not None]
    is_multichannel = (len(cn) == len(grids) and len(set(cn)) > 1)
    for d in grids:
        show_data = show
        if is_series or is_multichannel:
            show_data = False	# MapSeries or MapChannelsModel classes will decide which to show
        vkw = {'show_dialog': False}
        if hasattr(d, 'initial_style') and d.initial_style in ('surface', 'mesh', 'image'):
            vkw['style'] = d.initial_style
        v = dicom_volume_from_grid_data(d, session, open_model = False, **vkw)
        maps.append(v)
        if not show_data:
            v.display = False
        set_initial_region_and_style(v)

    show_dialog = kw.get('show_dialog', True)
    if maps and show_dialog:
        show_volume_dialog(session)

    msg = ''
    if is_series and is_multichannel:
        cmaps = {}
        for m in maps:
            cmaps.setdefault(m.data.channel,[]).append(m)
        if len(set(len(cm) for cm in cmaps.values())) > 1:
            session.logger.warning('Map channels have differing numbers of series maps: %s'
                                   % ', '.join('%d (%d)' % (c,cm) for c, cm in cmaps.items()))
        from chimerax.map_series import MapSeries
        ms = [MapSeries('channel %d' % c, cm, session) for c, cm in cmaps.items()]
        mc = MultiChannelSeries(name, ms, session)
        models = [mc]
    elif is_series:
        from chimerax.map_series import MapSeries
        ms = MapSeries(name, maps, session)
        ms.display = show
        models = [ms]
    elif is_multichannel:
        mc = MapChannelsModel(name, maps, session)
        mc.display = show
        mc.show_n_channels(3)
        models = [mc]
    elif len(maps) == 0:
        msg = 'No map data opened'
        session.logger.warning(msg)
        models = maps
    else:
        models = maps

    # Create surfaces before adding to session so that initial view can use corrrect bounds.
    for v in maps:
        if v.display:
            v.update_drawings()

    return models, msg

def zone_mask_clamped_by_referenced_grid(grid_data, referenced_grid, zone_points, zone_radius,
              min_value, max_value,
              invert_mask = False):
    """Like zone_mask from the map_data bundle, but with min/max value filtering"""
    from numpy import single as floatc, array, ndarray, zeros, int8, intc, where

    if not isinstance(zone_points, ndarray):
        zone_points = array(zone_points, floatc)

    shape = tuple(reversed(grid_data.size))
    mask_3d = zeros(shape, int8)
    mask_1d = mask_3d.ravel()

    if invert_mask:
        mask_value = 0
        mask_1d[:] = 1
    else:
        mask_value = 1

    from chimerax.map_data import grid_indices
    from chimerax.geometry import find_closest_points

    size_limit = 2 ** 22          # 4 Mvoxels
    if mask_3d.size > size_limit:
    # Calculate plane by plane to save memory with grid point array
        xsize, ysize, zsize = grid_data.size
        grid_points = grid_indices((xsize,ysize,1), floatc)
        grid_data.ijk_to_xyz_transform.transform_points(grid_points, in_place = True)
        flat_reference = referenced_grid.full_matrix().ravel()
        zstep = [grid_data.ijk_to_xyz_transform.matrix[a][2] for a in range(3)]
        for z in range(zsize):
            i1, i2, n1 = find_closest_points(grid_points, zone_points, zone_radius)
            offset = xsize*ysize*z
            if min_value and max_value:
                mask = where((min_value <= flat_reference[i1 + offset]) & (flat_reference[i1 + offset] <= max_value), 1, 0)
                mask_1d[i1 + offset] = mask
            else:
                mask_1d[i1 + offset] = mask_value
            grid_points[:,:] += zstep
    else:
        grid_points = grid_indices(grid_data.size, floatc)
        grid_data.ijk_to_xyz_transform.transform_points(grid_points, in_place = True)
        flat_reference = referenced_grid.full_matrix().ravel()
        i1, _, n1 = find_closest_points(grid_points, zone_points, zone_radius)
        if min_value and max_value:
            mask = where((min_value <= flat_reference[i1]) & (flat_reference[i1] <= max_value), 1, 0)
            mask_1d[i1] = mask
        else:
            mask_1d[i1] = mask_value
    return mask_3d