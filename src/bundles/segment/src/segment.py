# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Command to assign segmentation colors and create segmentation surfaces.
#

# -----------------------------------------------------------------------------
#
def segmentation_colors(session, segmentations, color = None, map = None, surface = None,
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
        
    if map is None and surface is None:
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
            raise UserError('segmentation colors: Volume size %s' % tuple(map.data.size) +
                            ' does not match segmentation size %s' % tuple(seg.data.size))

        _color_map(map, seg, by_attribute, color, outside_color)

    if surface is not None:
        if len(segmentations) != 1:
            from chimerax.core.errors import UserError
            raise UserError('segmentation colors: Can only specify one segmentation'
                            ' when coloring a surface, got %d' % len(segmentations))
        seg = segmentations[0]
        _color_surface(surface, seg, by_attribute,
                       color=color, outside_color=outside_color, step=step)
        
# -----------------------------------------------------------------------------
#
def _color_segmentation(segmentation, attribute_name, color = None, outside_color = None):
    seg = segmentation
    c = _attribute_colors(seg, attribute_name)
    zc = (0,0,0,0) if outside_color is None else outside_color
    seg_colors = c.segment_colors(color, zc)
    seg.segment_colors = seg_colors
    i = seg._image 
    if i:
        i.segment_colors = seg_colors
        i._need_color_update()
        
# -----------------------------------------------------------------------------
#
def _color_map(map, segmentation, attribute_name, color = None, outside_color = None):
    ac = _attribute_colors(segmentation, attribute_name)
    zc = (255,255,255,255) if outside_color is None else outside_color
    seg_rgba = ac.segment_colors(color, zc)
    seg_rgb = seg_rgba[:3].copy()	# Make contiguous
    def seg_color(color_plane, region, seg=segmentation,
                  segment_rgba = seg_rgba, segment_rgb = seg_rgb):
        seg_matrix = seg.region_matrix(region)
        segment_ids = seg_matrix.reshape(color_plane.shape[:-1]) # Squeeze out single plane dimension
        nc = color_plane.shape[-1] # Number of color components, 4 for rgba, 3 for rgb
        segment_colors = segment_rgba if nc == 4 else segment_rgb
        from chimerax.map import indices_to_colors
        indices_to_colors(segment_ids, segment_colors, color_plane, modulate = True)

    map.mask_colors = seg_color

    i = map._image 
    if i:
        i.mask_colors = map.mask_colors
        i._need_color_update()
        
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

# -----------------------------------------------------------------------------
#
def _surface_vertex_segments(surface, segmentation, step = (1,1,1)):
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
            mi = seg._max_segment_id if hasattr(seg, '_max_segment_id') else seg.full_matrix().max()
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
        self.attribute_rgba = _random_colors(nc)

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
        
# -----------------------------------------------------------------------------
#
def _random_colors(count, opaque = True):
    from numpy import random, uint8
    c = random.randint(128, high = 255, size = (count, 4), dtype = uint8)
    if opaque:
        c[:,3] = 255
    return c

# -----------------------------------------------------------------------------
#
def segmentation_surfaces(session, segmentations, region = 'all', step = None,
                          value = None, zero = False, by_attribute = None,
                          color = None):

    if len(segmentations) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No segmentations specified')

    if color is not None:
        color = color.uint8x4()

    surfaces = []
    tcount = 0
    from ._segment import segment_surface, segment_surfaces, segment_group_surfaces
    for seg in segmentations:
        sstep = _voxel_limit_step(seg, region) if step is None else step
        matrix = seg.matrix(step = sstep, subregion = region)
        if by_attribute is not None:
            g = _attribute_values(seg, by_attribute)
            if value is not None:
                g = (g == value)
            surfs = segment_group_surfaces(matrix, g, zero=zero)
            models_name = ('%s %d %s surfaces' % (seg.name, len(surfs), by_attribute))
        elif value is not None:
            va, ta = segment_surface(matrix, value)
            surfs = [(value, va, ta)]
        else:
            surfs = segment_surfaces(matrix, zero=zero)
            models_name = ('%s %d surfaces' % (seg.name, len(surfs)))
        tcount += sum([len(ta) for value, va, ta in surfs])
        segsurfs = []
        tf = seg.matrix_indices_to_xyz_transform(step = sstep, subregion = region)
        if color is None:
            colors = _attribute_colors(seg, by_attribute).attribute_rgba
        for surf in surfs:
            region_id, va, ta = surf
            tf.transform_points(va, in_place = True)
            from chimerax.surface import calculate_vertex_normals
            na = calculate_vertex_normals(va, ta)
            from chimerax.core.models import Surface
            s = Surface('%s %d' % (seg.name, region_id), session)
            s.clip_cap = True  # Cap surface when clipped
            s.color = colors[region_id] if color is None else color
            s.set_geometry(va, na, ta)
            segsurfs.append(s)
        surfaces.extend(segsurfs)

        nsurf = len(segsurfs)
        if nsurf > 1:
            session.models.add_group(segsurfs, name = models_name)
        elif nsurf == 1:
            session.models.add(segsurfs)

        session.logger.info('Created %d segmentation surfaces, %d triangles, subsampled %s'
                            % (nsurf, tcount, _step_string(sstep)))
    return surfaces

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
def _attribute_values(seg, attribute_name):
    from chimerax.core.errors import UserError
    if hasattr(seg, attribute_name):
        g = getattr(seg, attribute_name)
    else:
        try:
            g = seg.data.find_attribute(attribute_name)
        except:
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
                            ' must be a 1-D array (numpy, tuple, or list), got %d'
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
    from chimerax.core.commands import CmdDesc, register, IntArg, BoolArg, StringArg, SurfaceArg, ColorArg
    from chimerax.map import MapsArg, MapArg, MapRegionArg, MapStepArg

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        optional = [('color', ColorArg)],
        keyword = [('map', MapArg),
                   ('surface', SurfaceArg),
                   ('by_attribute', StringArg),
                   ('outside_color', ColorArg),
                   ('max_segment_id', IntArg),
                   ('step', MapStepArg),],
        synopsis = 'Set segmentation to use random colors, or apply segmentation coloring to a volume'
    )
    register('segmentation colors', desc, segmentation_colors, logger=logger)

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        keyword = [('value', IntArg),
                   ('zero', BoolArg),
                   ('by_attribute', StringArg),
                   ('color', ColorArg),
                   ('region', MapRegionArg),
                   ('step', MapStepArg),],
        synopsis = 'Create surfaces for a segmentation regions.'
    )
    register('segmentation surfaces', desc, segmentation_surfaces, logger=logger)
