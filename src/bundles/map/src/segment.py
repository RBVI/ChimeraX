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
def segmentation_colors(session, segmentations, max_index = None, map = None):

    if len(segmentations) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No segmentations specified')

    for seg in segmentations:
        if hasattr(seg, 'segment_colors') and map is not None:
            continue
        mi = seg.full_matrix().max() if max_index is None else max_index
        seg.segment_colors = c = _random_colors(mi)
        seg.segment_rgb = c[:,:3].copy()	# RGB as contiguous array.
        i = seg._image 
        if i:
            i.segment_colors = c
            i._need_color_update()

    if map is not None:
        if len(segmentations) != 1:
            from chimerax.core.errors import UserError
            raise UserError('segmentation colors: Can only specify one segmentation when coloring a map, got %d'
                            % len(segmentations))
        seg = segmentations[0]
        if tuple(map.data.size) != tuple(seg.data.size):
            from chimerax.core.errors import UserError
            raise UserError('segmentation colors: Volume size %s' % tuple(map.data.size) +
                            ' does not match segmentation size %s' % tuple(seg.data.size))
            
        def seg_color(color_plane, region, seg=seg):
            mask = seg.region_matrix(region)
            indices = mask.reshape(color_plane.shape[:-1])	# Squeeze out single plane dimension
            nc = color_plane.shape[-1] # Number of color components, 4 for rgba, 3 for rgb
            seg_colors = seg.segment_rgb if nc == 3 else seg.segment_colors
            from ._map import indices_to_colors
            indices_to_colors(indices, seg_colors, color_plane, modulate = True)

        map.mask_colors = seg_color

        i = map._image 
        if i:
            i.mask_colors = map.mask_colors
            i._need_color_update()
        

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
                          value = None, zero = False, group = None):

    if len(segmentations) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No segmentations specified')

    surfaces = []
    tcount = 0
    from ._map import segment_surface, segment_surfaces, segment_group_surfaces
    from chimerax.core.colors import random_colors
    for seg in segmentations:
        matrix = seg.matrix(step = step, subregion = region)
        if group is not None:
            g = _group_attribute(seg, group)
            if value is not None:
                g = (g == value)
            surfs = segment_group_surfaces(matrix, g, zero=zero)
            group_name = ('%s %d %s surfaces' % (seg.name, len(surfs), group))
        elif value is not None:
            va, ta = segment_surface(matrix, value)
            surfs = [(value, va, ta)]
        else:
            surfs = segment_surfaces(matrix, zero=zero)
            group_name = ('%s %d surfaces' % (seg.name, len(surfs)))
        tcount += sum([len(ta) for value, va, ta in surfs])
        segsurfs = []
        tf = seg.matrix_indices_to_xyz_transform(step = step, subregion = region)
        for surf in surfs:
            v, va, ta = surf
            tf.transform_points(va, in_place = True)
            from chimerax.surface import calculate_vertex_normals
            na = calculate_vertex_normals(va, ta)
            from chimerax.core.models import Surface
            s = Surface('%s %d' % (seg.name, v), session)
            s.clip_cap = True  # Cap surface when clipped
            s.color = random_colors(1)[0]
            s.set_geometry(va, na, ta)
            segsurfs.append(s)
        surfaces.extend(segsurfs)

        nsurf = len(segsurfs)
        if nsurf > 1:
            session.models.add_group(segsurfs, name = group_name)
        elif nsurf == 1:
            session.models.add(segsurfs)

        session.logger.info('Created %d segmentation surfaces, %d triangles'
                            % (nsurf, tcount))
    return surfaces

# -----------------------------------------------------------------------------
#
def _group_attribute(seg, group):
    from chimerax.core.errors import UserError
    if not hasattr(seg, group):
        raise UserError('Segmentation %s (#%s) has no group attribute %s'
                        % (seg.name, seg.id_string, group))
    g = getattr(seg, group)
    if isinstance(g, (tuple, list)):
        for i,e in enumerate(g):
            if not isinstance(e, int):
                raise UserError('Segmentation %s (#%s) group array %s has non-integer value %s at index %d'
                                % (seg.name, seg.id_string, group, type(g), i))
    else:
        from numpy import ndarray, int32
        if not isinstance(g, ndarray):
            raise UserError('Segmentation %s (#%s) group array %s must be a 1-D array (numpy, tuple, or list), got %d'
                            % (seg.name, seg.id_string, group, type(g)))
            
        if g.dtype != int32:
            raise UserError('Segmentation %s (#%s) group array %s has type %s, require int32'
                            % (seg.name, seg.id_string, group, g.dtype))
            
        if len(g.shape) != 1:
            raise UserError('Segmentation %s (#%s) group array %s is %d dimensional, require 1 dimensional'
                            % (seg.name, seg.id_string, group, len(g.shape)))
    return g

# -----------------------------------------------------------------------------
#
def register_segmentation_command(logger):
    from chimerax.core.commands import CmdDesc, register, IntArg, BoolArg, StringArg
    from chimerax.map import MapsArg, MapArg
    from .mapargs import MapRegionArg, MapStepArg

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        keyword = [('max_index', IntArg),
                   ('map', MapArg)],
        synopsis = 'Set segmentation to use random colors, or apply segmentation coloring to a volume'
    )
    register('segmentation colors', desc, segmentation_colors, logger=logger)

    desc = CmdDesc(
        required = [('segmentations', MapsArg)],
        keyword = [('value', IntArg),
                   ('zero', BoolArg),
                   ('group', StringArg),
                   ('region', MapRegionArg),
                   ('step', MapStepArg),],
        synopsis = 'Create surfaces for a segmentation regions.'
    )
    register('segmentation surfaces', desc, segmentation_surfaces, logger=logger)
