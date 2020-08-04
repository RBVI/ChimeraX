# -----------------------------------------------------------------------------
# Command to extract a piece of a volume data set within a surface.
#
#   mask <volumes> <surfaces>
#        [axis <x,y,z>]
#        [fullMap true|false]
#        [pad <d>]
#        [slab <width>|<d1,d2>]
#        [sandwich true|false]
#
#   mask #0 #1 axis 0,0,1 fullmap true pad 10
#
# masks volume #0 using surface #1 via projection along the z axis, and
# makes a full copy of the map (rather than a minimal subregion) and expands
# the surface along its per-vertex normal vectors before masking.
#

# -----------------------------------------------------------------------------
#
def mask(session, volumes, surfaces, axis = None, full_map = False,
         extend = 0, pad = 0, slab = None, sandwich = True, invert_mask = False,
         fill_overlap = False, model_id = None):
    '''Create a new volume where values outside specified surfaces are set to zero.'''
    
    if len(volumes) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No volumes specified')

    if len(surfaces) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No surfaces specified')

    axis = (0,1,0) if axis is None else axis.scene_coordinates()
    if isinstance(slab, float):
        pad = (-0.5*slab, 0.5*slab)
    elif slab is not None:
        pad = slab

    mvlist = []
    from .depthmask import surface_geometry, masked_volume
    for v in volumes:
        surfs = surface_geometry(surfaces, v.position.inverse(), pad)
        if len(surfs) == 0:
            from chimerax.core.errors import UserError
            raise UserError('Only empty surfaces specified')
        mv = masked_volume(v, surfs, axis, full_map, sandwich, invert_mask,
                           fill_overlap, extend, model_id)
        mvlist.append(mv)

    return mvlist

# -----------------------------------------------------------------------------
#
def ones_mask(session, surfaces, on_grid = None, spacing = None, border = 0, axis = None,
              extend = 0, pad = 0, full_map = True,
              slab = None, sandwich = True, invert_mask = False,
              fill_overlap = False, value_type = None, model_id = None):
    '''Create a volume which has value 1 inside surfaces and 0 outside.'''
    
    if len(surfaces) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No surfaces specified')

    if isinstance(slab, float):
        vpad = abs(slab)
    elif slab is not None:
        vpad = max(abs(p) for p in pad)
    else:
        vpad = pad

    from numpy import int8
    vtype = int8 if value_type is None else value_type
    
    if on_grid is None:
        v = ones_volume(surfaces, vpad, spacing, border, value_type = vtype)
    else:
        v = on_grid.writable_copy(require_copy = True, open_model = False, value_type = vtype)
        v.name = on_grid.name
        v.full_matrix()[:] = 1
    volumes = [v]

    mvlist = mask(session, volumes, surfaces, axis = axis, full_map = full_map,
                  extend = extend, pad = pad, slab = slab, sandwich = sandwich,
                  invert_mask = invert_mask, fill_overlap = fill_overlap,
                  model_id = model_id)
    mv = mvlist[0]
    mv.set_parameters(surface_levels = [0.5])
    
    return mvlist

# -----------------------------------------------------------------------------
#
def ones_volume(surfaces, pad, spacing, border, default_size = 100,
                value_type = None):

    # Figure out array size
    bounds = scene_bounds(surfaces)
    bsize = [s + 2*pad + 2*border for s in bounds.size()]
    if spacing is None:
        s = max(bsize)/default_size
        spacing = (s,s,s)
    from math import ceil
    size = [1 + int(ceil(s/sp)) for s,sp in zip(bsize,spacing)]
    origin = [x - (pad+border) for x in bounds.xyz_min]

    # Create ones array
    from numpy import ones, int8
    vtype = int8 if value_type is None else value_type
    varray = ones(size[::-1], vtype)
    from chimerax.map_data import ArrayGridData
    g = ArrayGridData(varray, origin, spacing, name = 'mask')

    # Create Volume model
    from chimerax.map import volume_from_grid_data
    v = volume_from_grid_data(g, surfaces[0].session,
                              open_model = False, show_dialog = False)

    return v

# -----------------------------------------------------------------------------
#
def scene_bounds(models, displayed_only = True):
    from chimerax.geometry import union_bounds
    b = union_bounds([m.bounds() for m in models])
    return b

# -----------------------------------------------------------------------------
#
def register_mask_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, IntArg, FloatArg
    from chimerax.core.commands import AxisArg, Float2Arg, Or, ModelIdArg, SurfacesArg
    from chimerax.map import MapsArg, Float1or3Arg, MapArg, ValueTypeArg
    mask_kw = [('pad', FloatArg),
               ('extend', IntArg),
               ('full_map', BoolArg),
               ('slab', Or(FloatArg, Float2Arg)),
               ('invert_mask', BoolArg),
               ('axis', AxisArg),
               ('sandwich', BoolArg),
               ('fill_overlap', BoolArg),
               ('model_id', ModelIdArg)]
    desc = CmdDesc(
        required = [('volumes', MapsArg)],
        keyword = [('surfaces', SurfacesArg)] + mask_kw,
        required_arguments = ['surfaces'],
        synopsis = 'Mask a map to a surface'
    )
    register('volume mask', desc, mask, logger=logger)

    desc = CmdDesc(
        required = [('surfaces', SurfacesArg)],
        keyword = [('on_grid', MapArg),
                   ('spacing', Float1or3Arg),
                   ('border', FloatArg),
                   ('value_type', ValueTypeArg)] + mask_kw,
        synopsis = 'Make a mask of 1 values for a surface'
    )
    register('volume onesmask', desc, ones_mask, logger=logger)
