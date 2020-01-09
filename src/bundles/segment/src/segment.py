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
def segmentation_colors(session, segmentations, map = None,
                        by_attribute = None, max_segment_id = None):

    if len(segmentations) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No segmentations specified')

    if max_segment_id is not None:
        for seg in segmentations:
            seg._max_segment_id = max_segment_id
            
    if map is None:
        for seg in segmentations:
            _color_segmentation(seg, by_attribute)

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

        _color_map(map, seg, by_attribute)
        
# -----------------------------------------------------------------------------
#
def _color_segmentation(segmentation, attribute_name):
    seg = segmentation
    c = _attribute_colors(seg, attribute_name)
    seg.segment_colors = c.rgba
    i = seg._image 
    if i:
        i.segment_colors = c.rgba
        i._need_color_update()
        
# -----------------------------------------------------------------------------
#
def _color_map(map, segmentation, attribute_name):
    def seg_color(color_plane, region, seg=segmentation, attribute_name=attribute_name):
        mask = seg.region_matrix(region)
        indices = mask.reshape(color_plane.shape[:-1])	# Squeeze out single plane dimension
        nc = color_plane.shape[-1] # Number of color components, 4 for rgba, 3 for rgb
        c = _attribute_colors(seg, attribute_name)
        seg_colors = c.rgba if nc == 2 or nc == 4 else c.rgb
        from chimerax.map import indices_to_colors
        indices_to_colors(indices, seg_colors, color_plane, modulate = True)

    map.mask_colors = seg_color

    i = map._image 
    if i:
        i.mask_colors = map.mask_colors
        i._need_color_update()
        
# -----------------------------------------------------------------------------
#
def _attribute_colors(seg, attribute_name):
    if not hasattr(seg, '_segment_attribute_colors'):
        seg._segment_attribute_colors = {}	# Map attribute name to SegmentationColors
    if attribute_name in seg._segment_attribute_colors:
        c = seg._segment_attribute_colors[attribute_name]
    else:
        c = SegmentationColors(seg, attribute_name)
        seg._segment_attribute_colors[attribute_name] = c
    return c

# -----------------------------------------------------------------------------
#
class SegmentationColors:
    def __init__(self, segmentation, attribute_name = None):
        self._attribute_name = attribute_name

        seg = segmentation
        if attribute_name is None:
            mi = seg._max_segment_id if hasattr(seg, '_max_segment_id') else seg.full_matrix().max()
            g = None
        else:
            g = _attribute_values(seg, attribute_name)
            mi = len(g)-1
        if not hasattr(seg, '_max_segment_id'):
            seg._max_segment_id = mi

        if g is not None:
            nc = g.max() + 1
            gc = _random_colors(nc)
            gc[0,:] = 0	# Attribute value 0 is transparent black
            c = gc[g].copy()
        else:
            c = gc = _random_colors(mi)

        self.rgba = c
        self.rgb = c[:,:3].copy()
        self.attribute_rgba = gc
        
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
def segmentation_surfaces(session, segmentations, region = None, step = None,
                          value = None, zero = False, by_attribute = None):

    if len(segmentations) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No segmentations specified')

    surfaces = []
    tcount = 0
    from ._segment import segment_surface, segment_surfaces, segment_group_surfaces
    for seg in segmentations:
        matrix = seg.matrix(step = step, subregion = region)
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
        tf = seg.matrix_indices_to_xyz_transform(step = step, subregion = region)
        colors = _attribute_colors(seg, by_attribute).attribute_rgba
        for surf in surfs:
            region_id, va, ta = surf
            tf.transform_points(va, in_place = True)
            from chimerax.surface import calculate_vertex_normals
            na = calculate_vertex_normals(va, ta)
            from chimerax.core.models import Surface
            s = Surface('%s %d' % (seg.name, region_id), session)
            s.clip_cap = True  # Cap surface when clipped
            s.color = colors[region_id]
            s.set_geometry(va, na, ta)
            segsurfs.append(s)
        surfaces.extend(segsurfs)

        nsurf = len(segsurfs)
        if nsurf > 1:
            session.models.add_group(segsurfs, name = models_name)
        elif nsurf == 1:
            session.models.add(segsurfs)

        session.logger.info('Created %d segmentation surfaces, %d triangles'
                            % (nsurf, tcount))
    return surfaces

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
    from chimerax.core.commands import CmdDesc, register, IntArg, BoolArg, StringArg
    from chimerax.map import MapsArg, MapArg, MapRegionArg, MapStepArg

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        keyword = [('map', MapArg),
                   ('by_attribute', StringArg),
                   ('max_index', IntArg)],
        synopsis = 'Set segmentation to use random colors, or apply segmentation coloring to a volume'
    )
    register('segmentation colors', desc, segmentation_colors, logger=logger)

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        keyword = [('value', IntArg),
                   ('zero', BoolArg),
                   ('by_attribute', StringArg),
                   ('region', MapRegionArg),
                   ('step', MapStepArg),],
        synopsis = 'Create surfaces for a segmentation regions.'
    )
    register('segmentation surfaces', desc, segmentation_surfaces, logger=logger)
