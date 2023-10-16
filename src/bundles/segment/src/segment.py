# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Command to assign segmentation colors and create segmentation surfaces.
#

# -----------------------------------------------------------------------------
#
def segmentation_colors(session, segmentations, color = None,
                        map = None, surfaces = None,
                        by_attribute = None, outside_color = None,
                        step = None,  # Step is just used for surface coloring interpolation.
                        max_segment_id = None):

    if len(segmentations) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No segmentations specified')

    if max_segment_id is not None:
        for seg in segmentations:
            seg._max_segment_id = max_segment_id

    if color is not None:
        color = color.uint8x4()
    if outside_color is not None:
        outside_color = outside_color.uint8x4()
        
    if map is None and surfaces is None:
        for seg in segmentations:
            _color_segmentation(seg, by_attribute, color, outside_color)

    if map is not None:
        if len(segmentations) != 1:
            from chimerax.core.errors import UserError
            raise UserError('segmentation colors: Can only specify one segmentation'
                            ' when coloring a map, got %d' % len(segmentations))
        seg = segmentations[0]
        if tuple(map.data.size) != tuple(seg.data.size):
            from chimerax.core.errors import UserError
            raise UserError('segmentation colors: Volume size %d,%d,%d' % tuple(map.data.size) +
                            ' does not match segmentation size %d,%d,%d' % tuple(seg.data.size))

        _color_map(map, seg, by_attribute, color, outside_color)

    if surfaces is not None:
        if len(segmentations) != 1:
            from chimerax.core.errors import UserError
            raise UserError('segmentation colors: Can only specify one segmentation'
                            ' when coloring a surface, got %d' % len(segmentations))
        seg = segmentations[0]
        for surface in surfaces:
            _color_surface(surface, seg, by_attribute,
                           color=color, outside_color=outside_color, step=step)
        
# -----------------------------------------------------------------------------
#
def _color_segmentation(segmentation, attribute_name, color = None, outside_color = None):
    seg = segmentation
    ac = _attribute_colors(seg, attribute_name)
    zc = (0,0,0,0) if outside_color is None else outside_color
    seg_colors = ac.segment_colors(color, zc)
    if outside_color is None and seg.segment_colors is not None:
        seg_mask = (ac._segment_attribute_values == 0)
        seg_colors[seg_mask] = seg.segment_colors[seg_mask]
    seg.segment_colors = seg_colors

# -----------------------------------------------------------------------------
#
def _color_map(map, segmentation, attribute_name, color = None, outside_color = None):
    ac = _attribute_colors(segmentation, attribute_name)
    zc = (255,255,255,255) if outside_color is None else outside_color
    seg_rgba = ac.segment_colors(color, zc)
    if outside_color is None and isinstance(map.mask_colors, SegmentationMapColor):
         # Blend with existing colors
        seg_mask = (ac._segment_attribute_values == 0)
        cur_rgba = map.mask_colors._segment_rgba
        seg_rgba[seg_mask] = cur_rgba[seg_mask]
    map.mask_colors = SegmentationMapColor(segmentation, seg_rgba)

# -----------------------------------------------------------------------------
#
class SegmentationMapColor:
    def __init__(self, segmentation, segment_rgba):
        self._segmentation = segmentation
        self._segment_rgba = segment_rgba
        self._segment_rgb = segment_rgba[:,:3].copy()	# Make contiguous

    def __call__(self, color_plane, region):
        seg = self._segmentation
        seg_matrix = seg.region_matrix(region)
        shape = color_plane.shape[:-1]
        segment_ids = seg_matrix.reshape(shape) # Squeeze out single plane dimension
        nc = color_plane.shape[-1] # Number of color components, 4 for rgba, 3 for rgb
        segment_colors = self._segment_rgba if nc == 4 else self._segment_rgb
        from chimerax.map import indices_to_colors
        indices_to_colors(segment_ids, segment_colors, color_plane, modulate = True)
        
# -----------------------------------------------------------------------------
#
def _color_surface(surface, segmentation, attribute_name,
                   color = None, outside_color = None, step = None):
    sstep = _voxel_limit_step(segmentation, subregion = 'all') if step is None else step
    vs = _surface_vertex_segments(surface, segmentation, step=sstep)
    c = _attribute_colors(segmentation, attribute_name)
    if outside_color is None:
        # Preserve current vertex colors outside where attribute value is 0.
        vc = surface.get_vertex_colors(create = True)
        c.modify_segment_colors(color, vs, vc)
    else:
        sc = c.segment_colors(color, outside_color)
        vc = sc[vs]
    surface.vertex_colors = vc

    vcount = c.unique_values_count(vs)
    what = 'segments' if attribute_name is None else ('%s values' % attribute_name)
    msg = 'Colored surface %s (#%s) with %d %s' % (surface.name, surface.id_string, vcount, what)
    surface.session.logger.info(msg)
    
# -----------------------------------------------------------------------------
#
def _surface_vertex_segments(surface, segmentation, step = (1,1,1)):
    # TODO: This can and does give incorrect segment ids for some vertices, getting the
    # segment id for a neighboring voxel instead of the segment id actually used to
    # compute the surface.  This happens when a two segment voxels are diagonally adjacent.
    # Would be much better to record the exact segment ids with each vertex when
    # the surface is computed.  Still need the method used below for surfaces that
    # were not computed from the segmentation.

    v = surface.vertices
    n = surface.normals
    offset = 0.5 * min(segmentation.data.step)
    points = v - offset*n
    seg_ids_float = segmentation.interpolated_values(points, surface.scene_position,
                                               step = step, method = 'nearest')
    # TODO: Interpolation is giving float32 which can only handle integers up to
    # 2**24 = 16 million.
    from numpy import int32
    seg_ids = seg_ids_float.astype(int32)
    return seg_ids
    
# -----------------------------------------------------------------------------
#
def _attribute_colors(seg, attribute_name):
    if not hasattr(seg, '_segment_attribute_colors'):
        seg._segment_attribute_colors = {}	# Map attribute name to AttributeColors
    if attribute_name in seg._segment_attribute_colors:
        c = seg._segment_attribute_colors[attribute_name]
    else:
        c = AttributeColors(seg, attribute_name)
        seg._segment_attribute_colors[attribute_name] = c
    return c

# -----------------------------------------------------------------------------
#
class AttributeColors:
    def __init__(self, segmentation, attribute_name = None, zero_color = (150,150,150,255)):
        self._attribute_name = attribute_name

        seg = segmentation
        if attribute_name is None:
            av = None
            mi = _maximum_segment_id(seg)
            nc = mi + 1
        else:
            av = _attribute_values(seg, attribute_name)
            mi = len(av)-1
            nc = av.max()+1

        if not hasattr(seg, '_max_segment_id'):
            seg._max_segment_id = mi

        self._max_segment_id = mi
        self._segment_attribute_values = av
        self._zero_color = zero_color
        self.attribute_rgba = _random_colors(nc, seed = (attribute_name or 'all'))

    def segment_colors(self, color = None, outside_color = None):
        zc = self._zero_color if outside_color is None else outside_color
        av = self._segment_attribute_values
        if color is None:
            ac = self.attribute_rgba.copy()
            ac[0,:] = zc
            c = ac if av is None else ac[av]
        else:
            from numpy import empty, uint8
            c = empty((self._max_segment_id + 1, 4), uint8)
            if av is None:
                c[:] = color
                c[0,:] = zc
            else:
                c[:] = zc
                c[av.nonzero()] = color
        return c

    def modify_segment_colors(self, color, segment_ids, colors):
        '''
        Change colors array for segments with nonzero attribute value.
        The segment_ids array gives the segment id for each element of the colors array.
        '''
        av = self._segment_attribute_values
        ea = segment_ids if av is None else av[segment_ids]
        nz = ea.nonzero()
        colors[nz] = self.attribute_rgba[ea][nz] if color is None else color

    def unique_values_count(self, segment_ids):
        av = self._segment_attribute_values
        if av is None:
            v = set(segment_ids)
        else:
            v = set(av[segment_ids])
        v.discard(0)
        vl = list(v)
        vl.sort()
        return len(v)
    
# -----------------------------------------------------------------------------
#
def _random_colors(count, cmin=50, cmax=255, opaque = True, seed = None):
    from numpy import random, uint8
    if seed is not None:
        from zlib import adler32
        random.seed(adler32(seed.encode('utf-8')))	# Ensure reproducible colors.
    c = random.randint(cmin, high = cmax, size = (count, 4), dtype = uint8)
    if opaque:
        c[:,3] = 255
    return c

# -----------------------------------------------------------------------------
#
def _which_segments(segmentation, conditions):

    attribute_name = 'segment'
    max_seg_id = _maximum_segment_id(segmentation)
    from numpy import arange, int32, logical_and
    group = arange(max_seg_id+1, dtype = int32)
    
    for condition in conditions:
        if '=' in condition:
            # All segments with attribute with specified value ("neuron_id=1")
            attribute_name, vals = condition.split('=', maxsplit = 1)
            av = _attribute_values(segmentation, attribute_name)
            values = _parse_integers(vals)
            if values is None:
                from chimerax.core.errors import UserError
                raise UserError('Require integer values "%s"' % condition)
            from numpy import isin
            mask = isin(av, values)
            logical_and(mask, group, mask)
            group[:] = 0
            group[mask] = av[mask]
        elif condition == 'segment':
            pass
        else:
            seg_ids = _parse_integers(condition)
            if seg_ids is None:
                # Condition is just an attribute name, e.g. "neuron_id".
                # Take all segments with non-zero attribute value.
                attribute_name = condition
                av = _attribute_values(segmentation, attribute_name)
                mask = (group != 0)
                group[mask] = av[mask]
            else:
                # One or more segments, e.g. "5-12,23,70-102"
                attribute_name = 'segment'
                try:
                    keep_seg_ids = seg_ids[group[seg_ids] != 0]
                except IndexError:
                    from chimerax.core.errors import UserError
                    raise UserError('Segment ids (%d - %d) out of range (0 - %d)'
                                    % (seg_ids.min(), seg_ids.max(), max_seg_id))
                group[:] = 0
                group[keep_seg_ids] = keep_seg_ids

    return group, attribute_name

# -----------------------------------------------------------------------------
#
def _parse_integers(ids_string):
    '''
    Parse integers, that can be comma-separated numbers and ranges
    such as 5,9,15-32,7.  Return an array of integer values.
    If the string does not contain integers return None.
    '''
    ids = []
    groups = ids_string.split(',')
    for g in groups:
        r = g.split('-')
        if len(r) == 1:
            try:
                r0 = int(r[0])
            except ValueError:
                return None
            ids.append(r0)
        elif len(r) == 2:
            try:
                r0,r1 = int(r[0]), int(r[1])
            except ValueError:
                return None
            ids.extend(range(r0,r1+1))
        else:
            return None
    from numpy import array, int32
    idsa = array(ids, int32)
    return idsa
    
# -----------------------------------------------------------------------------
#
def segmentation_surfaces(session, segmentations,
                          where = None, each = None, region = 'all', step = None,
                          color = None, smooth = False, smoothing_iterations = 10,
                          smoothing_factor = 1.0):

    if len(segmentations) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No segmentations specified')

    if color is not None:
        color = color.uint8x4()

    surfaces = []
    for seg in segmentations:
        sstep = _voxel_limit_step(seg, region) if step is None else step
        segsurfs = calculate_segmentation_surfaces(seg, where=where, each=each,
                                                   region=region, step=sstep, color=color,
                                                   smooth=smooth, smoothing_iterations=smoothing_iterations,
                                                   smoothing_factor=smoothing_factor)
        surfaces.extend(segsurfs)

        nsurf = len(segsurfs)
        if nsurf > 1:
            models_name = ('%s %d surfaces' % (seg.name, nsurf))
            session.models.add_group(segsurfs, name = models_name)
        elif nsurf == 1:
            session.models.add(segsurfs)

        tcount = sum([len(s.triangles) for s in segsurfs])
        session.logger.info('Created %d segmentation surfaces, %d triangles, subsampled %s'
                            % (nsurf, tcount, _step_string(sstep)))
    return surfaces

# -----------------------------------------------------------------------------
#
def calculate_segmentation_surfaces(seg, where = None, each = None,
                                    region = 'all', step = None, color = None,
                                    smooth = False, smoothing_iterations = 10,
                                    smoothing_factor = 1.0):
    # Warn if number of surfaces is large.
    if where is None and each is None:
        max_seg_id = _maximum_segment_id(seg)
        if max_seg_id > 100:
            from chimerax.core.errors import UserError
            raise UserError('Segmentation %s (#%s) has %d segments (> 100).'
                            ' To create surface for each segment use "each segment" option.'
                            % (seg.name, seg.id_string, max_seg_id))

    # Compute surfaces
    conditions = (where if where else []) + ([each] if each else [])
    group, attribute_name = _which_segments(seg, conditions)
    matrix = seg.matrix(step = step, subregion = region)
    from . import segmentation_surfaces
    surfs = segmentation_surfaces(matrix, group)

    # Transform vertices from index to scene units and compute normals.
    geom = []
    tf = seg.matrix_indices_to_xyz_transform(step = step, subregion = region)
    for region_id, va, ta in surfs:
        tf.transform_points(va, in_place = True)
        from chimerax.surface import calculate_vertex_normals
        na = calculate_vertex_normals(va, ta)
        if smooth:
            sf, si = smoothing_factor, smoothing_iterations
            from chimerax.surface import smooth_vertex_positions
            smooth_vertex_positions(va, ta, sf, si)
            smooth_vertex_positions(na, ta, sf, si)
        geom.append((region_id, va, na, ta))

    # Determine surface coloring.
    if color is None:
        attr = None if attribute_name == 'segment' else attribute_name
        colors = _attribute_colors(seg, attr).attribute_rgba
        
    # Create one or more surface models
    from chimerax.core.models import Surface
    from chimerax.surface import combine_geometry_xvnt
    segsurfs = []
    if each is None and len(geom) > 1:
        # Combine multiple surfaces into one.
        va, na, ta = combine_geometry_xvnt(geom)
        name = '%s %d %ss' % (seg.name, len(geom), attribute_name)
        s = Surface(name, seg.session)
        s.clip_cap = True  # Cap surface when clipped
        s.set_geometry(va, na, ta)
        if color is None:
            color_counts = [(colors[region_id], len(sva)) for region_id, sva, sna, sta in geom]
            s.vertex_colors = _vertex_colors(len(va), color_counts)
        else:
            s.color = color
        segsurfs.append(s)
    else:
        # Create multiple surface models
        for region_id, va, na, ta in geom:
            name = '%s %s %d' % (seg.name, attribute_name, region_id)
            s = Surface(name, seg.session)
            s.clip_cap = True  # Cap surface when clipped
            s.set_geometry(va, na, ta)
            s.color = colors[region_id] if color is None else color
            segsurfs.append(s)

    return segsurfs

# -----------------------------------------------------------------------------
#
def _vertex_colors(n, color_counts):
    from numpy import empty, uint8
    vertex_colors = empty((n,4), uint8)
    voffset = 0
    for color, nc in color_counts:
        vertex_colors[voffset:voffset+nc,:] = color
        voffset += nc
    return vertex_colors

# -----------------------------------------------------------------------------
#
def _voxel_limit_step(seg, subregion, increase_limit = 10):
    ijk_min, ijk_max = seg.subregion(subregion = subregion)[:2]
    si,sj,sk = [(i1-i0+1) for i0,i1 in zip(ijk_min, ijk_max)]
    vcount = si*sj*sk
    vlimit = seg.rendering_options.voxel_limit * 2**20 * increase_limit
    step = (1,1,1)
    while vcount / (step[0] * step[1] * step[2]) > vlimit:
        step = tuple(2*s for s in step)
    return step

# -----------------------------------------------------------------------------
#
def _step_string(step):
    si,sj,sk = step
    return '%d' % si if si == sj and si == sk else '%d,%d,%d' % (si,sj,sk)

# -----------------------------------------------------------------------------
#
def _maximum_segment_id(segmentation):
    seg = segmentation
    if hasattr(seg, '_max_segment_id'):
        max_seg_id = seg._max_segment_id
    else:
        try:
            max_seg_id = seg.data.find_attribute('maximum_segment_id')
        except Exception:
            max_seg_id = None
        if max_seg_id is None:
            from numpy import issubdtype, integer
            if not issubdtype(seg.data.value_type, integer):
                from chimerax.core.errors import UserError
                raise UserError('Model %s (#%s) is not a segmentation, has non-integer values (%s)'
                                % (seg.name, seg.id_string, str(seg.data.value_type)))
            max_seg_id = seg.full_matrix().max()
        seg._max_segment_id = max_seg_id

    return max_seg_id

# -----------------------------------------------------------------------------
#
def _attribute_values(seg, attribute_name):
    from chimerax.core.errors import UserError
    if hasattr(seg, attribute_name):
        g = getattr(seg, attribute_name)
    else:
        g = None
        if hasattr(seg.data, 'find_attribute'):
            g = seg.data.find_attribute(attribute_name)
        if g is None:
            raise UserError('Segmentation %s (#%s) has no attribute %s'
                            % (seg.name, seg.id_string, attribute_name))

    if isinstance(g, (tuple, list)):
        for i,e in enumerate(g):
            if not isinstance(e, int):
                raise UserError('Segmentation %s (#%s) attribute array %s'
                                ' has non-integer value %s at index %d'
                                % (seg.name, seg.id_string, attribute_name, type(g), i))
    else:
        from numpy import ndarray, int32
        if not isinstance(g, ndarray):
            raise UserError('Segmentation %s (#%s) attribute array %s'
                            ' must be a 1-D array (numpy, tuple, or list), got %s'
                            % (seg.name, seg.id_string, attribute_name, type(g)))
            
        if g.dtype != int32:
            from numpy import uint8, int8, uint16, int16, uint32
            if g.dtype in (uint8, int8, uint16, int16, uint32):
                g = g.astype(int32)
            else:
                raise UserError('Segmentation %s (#%s) attribute array %s'
                                ' has type %s, require int32'
                                % (seg.name, seg.id_string, attribute_name, g.dtype))
            
        if len(g.shape) != 1:
            raise UserError('Segmentation %s (#%s) attribute array %s'
                            ' is %d dimensional, require 1 dimensional'
                            % (seg.name, seg.id_string, attribute_name, len(g.shape)))
    return g

# -----------------------------------------------------------------------------
#
def register_segmentation_command(logger):
    from chimerax.core.commands import CmdDesc, register, IntArg, BoolArg, FloatArg, StringArg
    from chimerax.core.commands import SurfacesArg, ColorArg, RepeatOf
    from chimerax.map import MapsArg, MapArg, MapRegionArg, MapStepArg

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        optional = [('color', ColorArg)],
        keyword = [('map', MapArg),
                   ('surfaces', SurfacesArg),
                   ('by_attribute', StringArg),
                   ('outside_color', ColorArg),
                   ('max_segment_id', IntArg),
                   ('step', MapStepArg),],
        synopsis = 'Set segmentation to use random colors, or apply segmentation coloring to a volume'
    )
    register('segmentation colors', desc, segmentation_colors, logger=logger)

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        keyword = [('where', RepeatOf(StringArg)),
                   ('each', StringArg),
                   ('region', MapRegionArg),
                   ('step', MapStepArg),
                   ('color', ColorArg),
                   ('smooth', BoolArg),
                   ('smoothing_iterations', IntArg),
                   ('smoothing_factor', FloatArg)],
        synopsis = 'Create surfaces for a segmentation regions.'
    )
    register('segmentation surfaces', desc, segmentation_surfaces, logger=logger)
