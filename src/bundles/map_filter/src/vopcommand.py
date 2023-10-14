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
# Command to perform volume operations that create a new volume, such as
# erasing an octant, Gaussian filtering, adding maps, ....
#
#   Syntax: volume <operation> <volumeSpec>
#               [radius <r>]
#               [center <x,y,z>]
#               [i_center <i,j,k>]
#               [fill_value <v>]
#               [s_dev <d>]
#               [on_grid <id>]
#               [bounding_grid true|false>]
#               [axis_order xyz|yxz|zxy|zyx|yzx|xzy]
#               [model_id <n>]
#               [in_place true|false]
#               [subregion all|<i1,j1,k1,i2,j2,k2>]
#               [step <i>|<i,j,k>]
#               [grid_subregion all|<i1,j1,k1,i2,j2,k2>]
#               [grid_step <i>|<i,j,k>]
#               [frames <n>]
#               [start <f0>]
#               [play_step <fstep>]
#               [play_direction 1|-1]
#               [play_range <fmin,fmax>]
#               [add_mode true|false]
#               [constant_volume true|false]
#               [scale_factors "fromlevels"|<f1,f2,...>]
#
# where op is one of octant, ~octant, resample, add, zFlip, subtract, fourier,
# laplacian, gaussian, permuteAxes, bin, median, scale, boxes, morph, cover,
# flatten, unbend, tile.
#
from chimerax.core.errors import UserError as CommandError

def register_volume_filtering_subcommands(logger):

    from chimerax.core.commands import CmdDesc, register, BoolArg, StringArg, EnumOf, IntArg, Int3Arg
    from chimerax.core.commands import FloatArg, Float3Arg, FloatsArg, ModelIdArg
    from chimerax.core.commands import AxisArg, CenterArg, CoordSysArg
    from chimerax.core.commands import Or, EnumOf
    from chimerax.atomic import AtomsArg
    from chimerax.map.mapargs import MapsArg, MapStepArg, MapRegionArg, Int1or3Arg, Float1or3Arg, ValueTypeArg
    from chimerax.map.mapargs import BoxArg, Float2Arg

    ScaleFactorsArg = Or(EnumOf(['fromlevels']), FloatsArg)

    varg = [('volumes', MapsArg)]
    ssm_kw = [
        ('subregion', MapRegionArg),
        ('step', MapStepArg),
        ('model_id', ModelIdArg),
    ]
    resample_kw = [
        ('on_grid', MapsArg),
        ('bounding_grid', BoolArg),
        ('grid_subregion', MapRegionArg),
        ('grid_step', MapStepArg),
        ('spacing', Float1or3Arg),
        ('value_type', ValueTypeArg),
        ('hide_maps', BoolArg),
    ] + ssm_kw
    add_kw = resample_kw + [
        ('in_place', BoolArg),
        ('scale_factors', ScaleFactorsArg),
    ]
    add_desc = CmdDesc(required = varg, keyword = add_kw,
                       synopsis = 'Add two or more maps')
    register('volume add', add_desc, volume_add, logger=logger)

    bin_desc = CmdDesc(required = varg,
                       keyword = [('bin_size', MapStepArg)] + ssm_kw,
                       synopsis = 'Average map values over bins to reduce map size'
    )
    register('volume bin', bin_desc, volume_bin, logger=logger)

    boxes_desc = CmdDesc(required = varg,
                         keyword = [('centers', AtomsArg),
                                    ('size', FloatArg),
                                    ('isize', IntArg),
                                    ('use_marker_size', BoolArg)] + ssm_kw,
                         required_arguments = ['centers'],
                         synopsis = 'Extract maps for boxes around selected atoms')
    register('volume boxes', boxes_desc, volume_boxes, logger=logger)

    copy_desc = CmdDesc(required = varg,
                         keyword = [('value_type', ValueTypeArg)] + ssm_kw,
                         synopsis = 'Copy a map or a map subregion')
    register('volume copy', copy_desc, volume_copy, logger=logger)

    cover_desc = CmdDesc(required = varg,
        keyword = [('atom_box', AtomsArg),
                   ('pad', FloatArg),
                   ('box', BoxArg), ('x', Float2Arg), ('y', Float2Arg), ('z', Float2Arg),
                   ('f_box', BoxArg), ('fx', Float2Arg), ('fy', Float2Arg), ('fz', Float2Arg),
                   ('i_box', BoxArg), ('ix', Float2Arg), ('iy', Float2Arg), ('iz', Float2Arg),
                   ('use_symmetry', BoolArg),
                   ('cell_size', Int3Arg),
                   ('step', MapStepArg),
                   ('model_id', ModelIdArg)],
                         synopsis = 'Use symmetry to extend a map to cover a region'
    )
    register('volume cover', cover_desc, volume_cover, logger=logger)

    falloff_desc = CmdDesc(required = varg,
                           keyword = [('iterations', IntArg), ('in_place', BoolArg)] + ssm_kw,
                           synopsis = 'Smooth map values where map falls to zero'
    )
    register('volume falloff', falloff_desc, volume_falloff, logger=logger)

    flatten_desc = CmdDesc(required = varg,
                           keyword = [('method', EnumOf(('multiplyLinear', 'divideLinear'))),
                                      ('fitregion', MapRegionArg)] + ssm_kw,
                           synopsis = 'Flatten map gradients')
    register('volume flatten', flatten_desc, volume_flatten, logger=logger)

    flip_desc = CmdDesc(required = varg,
                        keyword = [('axis', EnumOf(('x','y','z','xy', 'yz','xyz'))),
                                   ('in_place', BoolArg)] + ssm_kw,
                        synopsis = 'Reverse axis directions')
    register('volume flip', flip_desc, volume_flip, logger=logger)

    fourier_desc = CmdDesc(required = varg,
                           keyword = [('phase', BoolArg)] + ssm_kw,
                           synopsis = 'Fourier transform a map')
    register('volume fourier', fourier_desc, volume_fourier, logger=logger)

    gauss_kw = [('s_dev', Float1or3Arg),
                 ('bfactor', FloatArg),
                 ('value_type', ValueTypeArg),
                 ('invert', BoolArg)] + ssm_kw
    gaussian_desc = CmdDesc(required = varg,
                            keyword = gauss_kw,
                            synopsis = 'Convolve map with a Gaussian for smoothing'
    )
    register('volume gaussian', gaussian_desc, volume_gaussian, logger=logger)

    laplacian_desc = CmdDesc(required = varg, keyword = ssm_kw,
                             synopsis = 'Laplace filter a map to enhance edges')
    register('volume laplacian', laplacian_desc, volume_laplacian, logger=logger)

    localcorr_desc = CmdDesc(required = varg,
                             keyword = [('window_size', IntArg),
                                        ('subtract_mean', BoolArg),
                                        ('model_id', ModelIdArg)],
                             synopsis = 'Compute correlation between maps over a sliding window')
    register('volume localCorrelation', localcorr_desc, volume_local_correlation, logger=logger)

    maximum_desc = CmdDesc(required = varg, keyword = add_kw,
                           synopsis = 'Maximum pointwise values of 2 or more maps')
    register('volume maximum', maximum_desc, volume_maximum, logger=logger)

    minimum_desc = CmdDesc(required = varg, keyword = add_kw,
                           synopsis = 'Minimum pointwise values of 2 or more maps')
    register('volume minimum', minimum_desc, volume_minimum, logger=logger)

    median_desc = CmdDesc(required = varg,
                          keyword = [('bin_size', MapStepArg),
                                     ('iterations', IntArg)] + ssm_kw,
                          synopsis = 'Median map value over a sliding window')
    register('volume median', median_desc, volume_median, logger=logger)

    morph_desc = CmdDesc(required = varg,
                         keyword = [('frames', IntArg),
                                    ('start', FloatArg),
                                    ('play_step', FloatArg),
                                    ('play_direction', IntArg),
                                    ('play_range', Float2Arg),
                                    ('slider', BoolArg),
                                    ('add_mode', BoolArg),
                                    ('constant_volume', BoolArg),
                                    ('scale_factors', ScaleFactorsArg),
                                    ('hide_original_maps', BoolArg),
                                    ('interpolate_colors', BoolArg)] + ssm_kw,
                         synopsis = 'Linearly interpolate maps')
    register('volume morph', morph_desc, volume_morph, logger=logger)

    multiply_desc = CmdDesc(required = varg, keyword = add_kw,
                            synopsis = 'Multiply maps pointwise')
    register('volume multiply', multiply_desc, volume_multiply, logger=logger)

    new_opt = [('name', StringArg),]
    new_kw = [('size', Int1or3Arg),
              ('grid_spacing', Float1or3Arg),
              ('origin', Float3Arg),
              ('cell_angles', Float1or3Arg),
              ('value_type', ValueTypeArg),
              ('model_id', ModelIdArg)]
    new_desc = CmdDesc(optional = new_opt, keyword = new_kw,
                       synopsis = 'Create a new map with zero values')
    register('volume new', new_desc, volume_new, logger=logger)

    oct_kw = [('center', Float3Arg),
              ('i_center', Int3Arg),
              ('fill_value', FloatArg),
              ('in_place', BoolArg)]
    octant_desc = CmdDesc(required = varg,
                          keyword = oct_kw + ssm_kw,
                          synopsis = 'Zero all but an octant of a map')
    register('volume octant', octant_desc, volume_octant, logger=logger)

    unoctant_desc = CmdDesc(required = varg,
                            keyword = oct_kw + ssm_kw,
                            synopsis = 'Zero an octant of a map')
    register('volume ~octant', unoctant_desc, volume_octant_complement, logger=logger)

    aoarg = [('axis_order', EnumOf(('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')))]
    permuteaxes_desc = CmdDesc(required = varg + aoarg,
                               keyword = ssm_kw,
                               synopsis = 'Permute map axes')
    register('volume permuteAxes', permuteaxes_desc, volume_permute_axes, logger=logger)

    resample_desc = CmdDesc(required = varg, keyword = resample_kw,
                            synopsis = 'Resample a map on the grid of another map')
    register('volume resample', resample_desc, volume_resample, logger=logger)

    ridges_desc = CmdDesc(required = varg,
                          keyword = [('level', FloatArg)] + ssm_kw,
                          synopsis = 'Filter to detect map ridges')
    register('volume ridges', ridges_desc, volume_ridges, logger=logger)

    scale_desc = CmdDesc(required = varg,
                         keyword = [('shift', FloatArg),
                                    ('factor', FloatArg),
                                    ('sd', FloatArg),
                                    ('rms', FloatArg),
                                    ('value_type', ValueTypeArg),
                                    ] + ssm_kw,
                         synopsis = 'Scale and shift map values')
    register('volume scale', scale_desc, volume_scale, logger=logger)

    sharpen_desc = CmdDesc(required = varg,
                           keyword = gauss_kw,
                           synopsis = 'Sharpen map by amplifying high-frequencies using bfactor'
    )
    register('volume sharpen', sharpen_desc, volume_sharpen, logger=logger)

    subtract_desc = CmdDesc(required = varg,
                            keyword = add_kw + [('min_rms', BoolArg)],
                            synopsis = 'Subtract maps pointwise')
    register('volume subtract', subtract_desc, volume_subtract, logger=logger)

    threshold_desc = CmdDesc(required = varg,
                             keyword = [('minimum', FloatArg),
                                        ('set', FloatArg),
                                        ('maximum', FloatArg),
                                        ('set_maximum', FloatArg)] + ssm_kw,
                             synopsis = 'Set map values below a threshold to zero'
    )
    register('volume threshold', threshold_desc, volume_threshold, logger=logger)

    orders = ('ulh', 'ulv', 'urh', 'urv', 'llh', 'llv', 'lrh', 'lrv',
              'ulhr', 'ulvr', 'urhr', 'urvr', 'llhr', 'llvr', 'lrhr', 'lrvr')
    tile_desc = CmdDesc(required = varg,
                        keyword = [('axis', EnumOf(('x','y','z'))),
                                   ('pstep', IntArg),
                                   ('trim', IntArg),
                                   ('columns', IntArg),
                                   ('rows', IntArg),
                                   ('fill_order', EnumOf(orders))] + ssm_kw,
                        synopsis = 'Create a single plane map by tiling planes of a map')
    register('volume tile', tile_desc, volume_tile, logger=logger)

    unbend_desc = CmdDesc(required = varg,
                          keyword = [('path', AtomsArg),
                                     ('yaxis', AxisArg),
                                     ('xsize', FloatArg),
                                     ('ysize', FloatArg),
                                     ('grid_spacing', FloatArg)] + ssm_kw,
                          required_arguments = ['path'],
                          synopsis = 'Unwarp a map along a curved rectangular tube'
    )
    register('volume unbend', unbend_desc, volume_unbend, logger=logger)

    unroll_desc = CmdDesc(required = varg,
                          keyword = [('inner_radius', FloatArg),
                                     ('outer_radius', FloatArg),
                                     ('length', FloatArg),
                                     ('grid_spacing', FloatArg),
                                     ('axis', AxisArg),
                                     ('center', CenterArg),
                                     ('coordinate_system', CoordSysArg)] + ssm_kw,
                          synopsis = 'Unroll a cylindrical shell to form a new map')
    register('volume unroll', unroll_desc, volume_unroll, logger=logger)

    zone_desc = CmdDesc(required = varg,
                        keyword = [('near_atoms', AtomsArg),
                                   ('range', FloatArg),
                                   ('bond_point_spacing', FloatArg),
                                   ('minimal_bounds', BoolArg),
                                   ('new_map', BoolArg),
                                   ('invert', BoolArg)] + ssm_kw,
                        required_arguments=('near_atoms',),
                        synopsis = 'Zero map values beyond a distance range from atoms')
    register('volume zone', zone_desc, volume_zone, logger=logger)

    unzone_desc = CmdDesc(optional = varg,
                          synopsis = 'Show full map with no surface masking.')
    register('volume unzone', unzone_desc, volume_unzone, logger=logger)

    from chimerax.core.commands import create_alias
    create_alias('vop', 'volume $*', logger=logger,
            url="help:user/commands/volume.html#vop")

# -----------------------------------------------------------------------------
#
def volume_add(session, volumes, on_grid = None, bounding_grid = None,
               subregion = 'all', step = 1,
               grid_subregion = 'all', grid_step = 1, spacing = None, value_type = None,
               in_place = False, scale_factors = None, model_id = None,
               hide_maps = True):
    '''Add maps.'''
    rv = combine_op(volumes, 'add', on_grid, bounding_grid, subregion, step,
                    grid_subregion, grid_step, spacing, value_type,
                    in_place, scale_factors, model_id, session,
                    hide_maps = hide_maps)
    return rv

# -----------------------------------------------------------------------------
#
def volume_maximum(session, volumes, on_grid = None, bounding_grid = None,
                   subregion = 'all', step = 1,
                   grid_subregion = 'all', grid_step = 1, spacing = None, value_type = None,
                   in_place = False, scale_factors = None, model_id = None,
                   hide_maps = True):
    '''Pointwise maximum of maps.'''
    rv = combine_op(volumes, 'maximum', on_grid, bounding_grid, subregion, step,
                    grid_subregion, grid_step, spacing, value_type,
                    in_place, scale_factors, model_id, session,
                    hide_maps = hide_maps)
    return rv

# -----------------------------------------------------------------------------
#
def volume_minimum(session, volumes, on_grid = None, bounding_grid = None,
                   subregion = 'all', step = 1,
                   grid_subregion = 'all', grid_step = 1, spacing = None, value_type = None,
                   in_place = False, scale_factors = None, model_id = None,
                   hide_maps = True):
    '''Pointwise minimum of maps.'''
    rv = combine_op(volumes, 'minimum', on_grid, bounding_grid, subregion, step,
                    grid_subregion, grid_step, spacing, value_type,
                    in_place, scale_factors, model_id, session,
                    hide_maps = hide_maps)
    return rv

# -----------------------------------------------------------------------------
#
def volume_multiply(session, volumes, on_grid = None, bounding_grid = None,
                    subregion = 'all', step = 1,
                    grid_subregion = 'all', grid_step = 1, spacing = None, value_type = None,
                    in_place = False, scale_factors = None, model_id = None,
                    hide_maps = True):
    '''Pointwise multiply maps.'''
    rv = combine_op(volumes, 'multiply', on_grid, bounding_grid, subregion, step,
                    grid_subregion, grid_step, spacing, value_type,
                    in_place, scale_factors, model_id, session,
                    hide_maps = hide_maps)
    return rv

# -----------------------------------------------------------------------------
#
def combine_op(volumes, operation = 'add', on_grid = None, bounding_grid = None,
               subregion = 'all', step = 1,
               grid_subregion = 'all', grid_step = 1, spacing = None, value_type = None,
               in_place = False, scale_factors = None, model_id = None, session = None,
               hide_maps = True, open_model = True):

    if bounding_grid is None and not in_place:
        bounding_grid = (on_grid is None)
    if on_grid is None:
        on_grid = volumes[:1]
    if in_place:
        if bounding_grid or grid_step != 1 or grid_subregion != 'all' or spacing is not None:
            raise CommandError("Can't use in_place option with bounding_grid or grid_step or grid_subregion or spacing options")
        for gv in on_grid:
            if not gv.data.writable:
                raise CommandError("Can't modify volume in place: %s" % gv.name)
            if not gv in volumes:
                raise CommandError("Can't change grid in place")
    scale_factors = _check_scale_factors(scale_factors, volumes)

    cv = [combine_operation(volumes, operation, subregion, step,
                            gv, grid_subregion, grid_step, spacing, value_type,
                            bounding_grid, in_place, scale_factors, model_id, session,
                            hide_maps = hide_maps, open_model = open_model)
          for gv in on_grid]

    return _volume_or_list(cv)

# -----------------------------------------------------------------------------
#
def combine_operation(volumes, operation, subregion, step,
                      gv, grid_subregion, grid_step, spacing, value_type,
                      bounding_grid, in_place, scale, model_id, session,
                      hide_maps = True, open_model = True):

    if scale is None:
        scale = [1]*len(volumes)
    if in_place:
        rv = gv
        for i, v in enumerate(volumes):
            s = (scale[i] if v != rv else scale[i]-1)
            op = 'add' if i == 0 else operation
            rv.combine_interpolated_values(v, op, subregion = subregion,
                                           step = step, scale = s)
    else:
        gr = gv.subregion(step = grid_step, subregion = grid_subregion)
        if bounding_grid:
            if same_grids(volumes, subregion, step, gv, gr):
                # Avoid extending grid due to round-off errors.
                r = gr
            else:
                corners = volume_corners(volumes, subregion, step,
                                         gv.model_transform())
                r = gv.bounding_region(corners, step = grid_step, clamp = False)
        else:
            r = gr
        v0 = volumes[0] if volumes else None
        if value_type is None:
            value_type = v0.data.value_type if volumes else gv.data.value_type
        rg = gv.region_grid(r, value_type = value_type, new_spacing = spacing, clamp = False)
        if len(volumes) == 1:
            rg.name = v0.name + ' resampled'
        elif operation == 'subtract':
            rg.name = 'volume difference'
            rg.polar_values = True
        elif operation == 'maximum':
            rg.name = 'volume maximum'
        elif operation == 'minimum':
            rg.name = 'volume minimum'
        elif operation == 'multiply':
            rg.name = 'volume product'
        else:
            rg.name = 'volume sum'
        from chimerax.map import volume_from_grid_data
        rv = volume_from_grid_data(rg, session, model_id = model_id,
                                   show_dialog = False, open_model = open_model)
        rv.position = gv.position
        for i,v in enumerate(volumes):
            op = 'add' if i == 0 else operation
            rv.combine_interpolated_values(v, op, subregion = subregion, step = step,
                                           scale = scale[i])
    rv.data.values_changed()
    if volumes and not in_place:
        rv.copy_settings_from(v0, copy_region = False, copy_xform = False, copy_colors = False)
        if rv.data.name.endswith('difference'):
            rv.set_parameters(cap_faces = False)

    if hide_maps:
        for v in volumes:
            if not v is rv:
                v.display = False

    return rv

# -----------------------------------------------------------------------------
#
def _check_scale_factors(scale_factors, volumes):
    if scale_factors is None:
        return scale_factors

    if scale_factors == 'fromlevels':
        for v in volumes:
            lev = v.maximum_surface_level
            if lev is None:
                raise CommandError('Scale factors "fromlevels" requires all maps have a surface level.  Map "%s" has no surface.' % v.name_with_id())
            if lev == 0:
                raise CommandError('Scale factors "fromlevels" requires all maps have a non-zero surface level.  Map "%s" has surface level 0.' % v.name_with_id())
        sf = tuple(1/v.maximum_surface_level for v in volumes)
        return sf
        
    if len(scale_factors) != len(volumes):
        raise CommandError('Number of scale factors does not match number of volumes')

    return scale_factors

# -----------------------------------------------------------------------------
#
def same_grids(volumes, subregion, step, gv, gr):

    from chimerax.map.volume import same_grid
    for v in volumes:
        if not same_grid(v, v.subregion(step, subregion), gv, gr):
            return False
    return True

# -----------------------------------------------------------------------------
#
def volume_corners(volumes, subregion, step, place):

    from chimerax.map_data import box_corners
    corners = []
    for v in volumes:
        xyz_min, xyz_max = v.xyz_bounds(step = step, subregion = subregion)
        vc = box_corners(xyz_min, xyz_max)
        c = place.inverse() * v.position * vc
        corners.extend(c)
    return corners

# -----------------------------------------------------------------------------
#
def volume_bin(session, volumes, subregion = 'all', step = 1,
           bin_size = (2,2,2), model_id = None):
    '''Reduce map by averaging over rectangular bins.'''
    from .bin import bin
    bv = [bin(v, bin_size, step, subregion, model_id, session)
          for v in volumes]
    return _volume_or_list(bv)

# -----------------------------------------------------------------------------
#
def volume_boxes(session, volumes, centers, size = 0, isize = None, use_marker_size = False,
             subregion = 'all', step = 1, model_id = None):
    '''Extract boxes centered at marker positions.'''
    if size <= 0 and isize is None and not use_marker_size:
        raise CommandError('Must specify size or isize or enable use_marker_size')

    vlist = []
    from .boxes import boxes
    for v in volumes:
        bv = boxes(session, v, centers, size, isize, use_marker_size, step, subregion, model_id)
        vlist.extend(bv)
    return vlist

# -----------------------------------------------------------------------------
#
def volume_copy(session, volumes, value_type = None, subregion = 'all', step = 1, model_id = None):
    '''Copy a map or map subregion.'''
    copies = [v.writable_copy(require_copy = True,
                              subregion = subregion, step = step,
                              value_type = value_type, model_id = model_id)
              for v in volumes]
    return _volume_or_list(copies)

# -----------------------------------------------------------------------------
#
def volume_cover(session, volumes, atom_box = None, pad = 5.0,
             box = None, x = None, y = None, z = None,
             f_box = None, fx = None, fy = None, fz = None,
             i_box = None, ix = None, iy = None, iz = None,
             use_symmetry = True, cell_size = None,
             step = (1,1,1), model_id = None):
    '''Extend a map using symmetry to cover a specified region.'''
    if not atom_box is None and len(atom_box) == 0:
        raise CommandError('No atoms specified')
    bc = len([b for b in (box, f_box, i_box, atom_box) if b])
    if bc == 0:
        raise CommandError('Must specify box to cover')
    if bc > 1:
        raise CommandError('Specify covering box in one way')

    from chimerax.map import volume_from_grid_data
    from .cover import cover_box_bounds, map_covering_box
    cvlist = []
    for v in volumes:
        ijk_min, ijk_max = cover_box_bounds(v, step,
                                            atom_box, pad, box, f_box, i_box)
        ijk_cell_size = (getattr(v.data, 'unit_cell_size', v.data.size)
                         if cell_size is None else cell_size)
        syms = v.data.symmetries if use_symmetry else ()
        cg = map_covering_box(v, ijk_min, ijk_max, ijk_cell_size, syms, step)

        cv = volume_from_grid_data(cg, session, model_id = model_id)
        cv.copy_settings_from(v, copy_region = False, copy_colors = False,
                              copy_zone = False)
        cvlist.append(cv)

    return _volume_or_list(cvlist)

# -----------------------------------------------------------------------------
#
def parse_box(box, x, y, z, bname, xname, yname, zname):

    if box is None and x is None and y is None and z is None:
        return None
    if box:
        return box
    box = ([None,None,None], [None,None,None])
    for a,x,name in zip((0,1,2),(x,y,z),(xname,yname,zname)):
        if x:
            box[0][a], box[1][a] = parse_floats(x, name, 2)
    return box

# -----------------------------------------------------------------------------
#
def volume_falloff(session, volumes, iterations = 10, in_place = False,
               subregion = 'all', step = 1, model_id = None):
    '''Smooth edges of a masked map.'''
    if in_place:
        ro = [v for v in volumes if not v.data.writable]
        if ro:
            raise CommandError("Can't use in_place with read-only volumes "
                               + ', '.join(v.name for v in ro))
        if step != 1:
            raise CommandError('Step must be 1 to modify data in place')
        if subregion != 'all':
            raise CommandError('Require subregion "all" to modify data in place')

    from .falloff import falloff
    fv = [falloff(v, iterations, in_place, step, subregion, model_id, session)
          for v in volumes]
    return _volume_or_list(fv)

# -----------------------------------------------------------------------------
#
def volume_flatten(session, volumes, method = 'multiplyLinear',
               fitregion = None, subregion = 'all', step = 1, model_id = None):
    '''Make map background flat.'''
    if fitregion is None:
        fitregion = subregion

    method = {'multiplyLinear': 'multiply linear',
              'divideLinear': 'divide linear'}[method]
    from .flatten import flatten
    fv = [flatten(v, method, step, subregion, fitregion, model_id)
          for v in volumes]
    return _volume_or_list(fv)

# -----------------------------------------------------------------------------
#
def volume_fourier(session, volumes, subregion = 'all', step = 1, model_id = None, phase = False):
    '''Fourier transform a map'''
    from .fourier import fourier_transform
    fv = [fourier_transform(v, step, subregion, model_id, phase)
          for v in volumes]
    return _volume_or_list(fv)

# -----------------------------------------------------------------------------
#
def volume_gaussian(session, volumes, s_dev = (1.0,1.0,1.0), bfactor = None,
                 subregion = 'all', step = 1, value_type = None, invert = False,
                 model_id = None):
    '''Smooth maps by Gaussian convolution.'''
    if bfactor is not None:
        if bfactor < 0:
            invert = not invert
        from math import pi, sqrt
        # Calculates sd according to https://www3.cmbi.umcn.nl/bdb/theory/
        sd = sqrt(abs(bfactor)/(3*8*pi**2))
        s_dev = (sd,sd,sd)

    from .gaussian import gaussian_convolve
    gv = [gaussian_convolve(v, s_dev, step, subregion, value_type, invert, model_id, session = session)
          for v in volumes]
    return _volume_or_list(gv)
                   
# -----------------------------------------------------------------------------
#
def volume_sharpen(session, volumes, s_dev = (1.0,1.0,1.0), bfactor = None,
                   subregion = 'all', step = 1, value_type = None, invert = False,
                   model_id = None):
    '''Sharpen map by amplifying high-frequencies using bfactor.'''
    if bfactor is not None:
        bfactor = -bfactor
    else:
        invert = not invert	# s_dev specified
    return volume_gaussian(session, volumes, s_dev = s_dev, bfactor = bfactor,
                           subregion = subregion, step = step, value_type = value_type,
                           invert = invert, model_id = model_id)
                   
# -----------------------------------------------------------------------------
#
def volume_laplacian(session, volumes, subregion = 'all', step = 1, model_id = None):
    '''Detect map edges with Laplacian filter.'''
    from .laplace import laplacian
    lv = [laplacian(v, step, subregion, model_id)
          for v in volumes]
    return _volume_or_list(lv)

# -----------------------------------------------------------------------------
#
def volume_local_correlation(session, volumes, window_size = 5, subtract_mean = False, model_id = None):
    '''Compute correlation between two maps over a sliding window.'''
    if len(volumes) != 2:
        raise CommandError('volume local_correlation operation requires '
                           'exactly two map arguments')
    v1, v2 = volumes
    if window_size < 2:
        raise CommandError('volume local_correlation window_size must be '
                           'an integer >= 2')
    if window_size > min(v1.data.size):
        raise CommandError('volume local_correlation window_size must be '
                           'smaller than map size')

    from .localcorr import local_correlation
    mapc = local_correlation(v1, v2, window_size, subtract_mean, model_id)
    return mapc

# -----------------------------------------------------------------------------
#
def volume_median(session, volumes, bin_size = (3,3,3), iterations = 1,
              subregion = 'all', step = 1, model_id = None):
    '''Replace map values with median of neighboring values.'''
    for b in bin_size:
        if b <= 0 or b % 2 == 0:
            raise CommandError('Bin size must be positive odd integer, got %d' % b)

    from .median import median_filter
    mv = [median_filter(v, bin_size, iterations, step, subregion, model_id)
          for v in volumes]
    return _volume_or_list(mv)

# -----------------------------------------------------------------------------
#
def volume_morph(session, volumes, frames = 25, start = 0, play_step = 0.04,
                 play_direction = 1, play_range = None, slider = True,
                 add_mode = False, constant_volume = False, scale_factors = None,
                 hide_original_maps = True, interpolate_colors = True,
                 subregion = 'all', step = 1, model_id = None):
    '''Linearly interpolate pointwise between maps.'''
    if len(volumes) < 2:
        raise CommandError('volume morph requires 2 or more volumes, got %d' % len(volumes))
    if play_range is None:
        if add_mode:
            prange = (-1.0,1.0)
        else:
            prange = (0.0,1.0)
    else:
        prange = play_range

    scale_factors = _check_scale_factors(scale_factors, volumes)
    vs = [tuple(v.matrix_size(step = step, subregion = subregion))
          for v in volumes]
    if len(set(vs)) > 1:
        sizes = ' and '.join([str(s) for s in vs])
        raise CommandError("Volume grid sizes don't match: %s" % sizes)
    v0 = volumes[0]
    for v in volumes[1:]:
        if (not v.scene_position.same(v0.scene_position) or
            not v.data.ijk_to_xyz_transform.same(v0.data.ijk_to_xyz_transform)):
            raise CommandError('Map positions are not the same, %s and %s.'
                               % (v.name_with_id(), v0.name_with_id()) +
                               '  Use the "volume resample" command to make a copy of one map with the same grid as the other map.')

    from .morph import morph_maps
    im = morph_maps(volumes, frames, start, play_step, play_direction, prange,
                    add_mode, constant_volume, scale_factors,
                    hide_original_maps, interpolate_colors, subregion, step, model_id)

    if slider and session.ui.is_gui:
        from .morph_gui import MorphMapSlider
        MorphMapSlider(session, im)

    return im

# -----------------------------------------------------------------------------
#
def volume_new(session, name = 'new', size = (100,100,100), grid_spacing = (1.0,1.0,1.0),
            origin = (0.0,0.0,0.0), cell_angles = (90,90,90),
            value_type = None, model_id = None):
    '''Create a new volume with specified bounds filled with zeros.'''

    from numpy import zeros
    shape = list(size)
    shape.reverse()
    if value_type is None:
        from numpy import float32
        value_type = float32
    a = zeros(shape, dtype = value_type)
    from chimerax.map_data import ArrayGridData
    grid = ArrayGridData(a, origin = origin, step = grid_spacing,
                         cell_angles = cell_angles, name = name)
    from chimerax.map import volume_from_grid_data
    v = volume_from_grid_data(grid, session, model_id = model_id)
    return v

# -----------------------------------------------------------------------------
#
def check_in_place(in_place, volumes):

    if not in_place:
        return
    nwv = [v for v in volumes if not v.data.writable]
    if nwv:
        names = ', '.join([v.name for v in nwv])
        raise CommandError("Can't modify volume in place: %s" % names)

# -----------------------------------------------------------------------------
#
def volume_octant(session, volumes, center = None, i_center = None,
              subregion = 'all', step = 1, in_place = False,
              fill_value = 0, model_id = None):
    '''Extract an octant from a map.'''
    check_in_place(in_place, volumes)
    outside = True
    ov = [octant_operation(v, outside, center, i_center, subregion, step,
                           in_place, fill_value, model_id)
          for v in volumes]
    return _volume_or_list(ov)

# -----------------------------------------------------------------------------
#
def volume_octant_complement(session, volumes, center = None, i_center = None,
                         subregion = 'all', step = 1, in_place = False,
                         fill_value = 0, model_id = None):
    '''Zero an octant of a map.'''
    check_in_place(in_place, volumes)
    outside = False
    ov = [octant_operation(v, outside, center, i_center, subregion, step,
                           in_place, fill_value, model_id)
          for v in volumes]
    return _volume_or_list(ov)

# -----------------------------------------------------------------------------
#
def octant_operation(v, outside, center, i_center,
                     subregion, step, in_place, fill_value, model_id):

    vc = v.writable_copy(require_copy = not in_place,
                         subregion = subregion, step = step,
                         model_id = model_id)
    ic = submatrix_center(v, center, i_center, subregion, step)
    ijk_max = [i-1 for i in vc.data.size]
    set_box_value(vc.data, fill_value, ic, ijk_max, outside)
    vc.data.values_changed()
    return vc

# -----------------------------------------------------------------------------
#
def set_box_value(data, value, ijk_min, ijk_max, outside = False):

    if outside:
        set_value_outside_box(data, value, ijk_min, ijk_max)
        return

    from math import floor, ceil
    ijk_origin = [max(0, int(floor(i))) for i in ijk_min]
    ijk_last = [min(s-1, int(ceil(i))) for i,s in zip(ijk_max, data.size)]
    ijk_size = [b-a+1 for a,b in zip(ijk_origin, ijk_last)]
    if len([i for i in ijk_size if i > 0]) < 3:
        return

    m = data.matrix(ijk_origin, ijk_size)
    m[:,:,:] = value

# -----------------------------------------------------------------------------
#
def set_value_outside_box(data, value, ijk_min, ijk_max):

    i0,j0,k0 = [i-1 for i in ijk_min]
    i1,j1,k1 = [i+1 for i in ijk_max]
    im,jm,km = [s-1 for s in data.size]
    set_box_value(data, value, (0,0,0), (im,jm,k0))
    set_box_value(data, value, (0,0,k1), (im,jm,km))
    set_box_value(data, value, (0,0,k0), (i0,jm,k1))
    set_box_value(data, value, (i1,0,k0), (im,jm,k1))
    set_box_value(data, value, (i0,0,k0), (i1,j0,k1))
    set_box_value(data, value, (i0,j1,k0), (i1,jm,k1))

# -----------------------------------------------------------------------------
#
def volume_permute_axes(session, volumes, axis_order = 'xyz',
                    subregion = 'all', step = 1, model_id = None):
    '''Permute map axes.'''
    ao = {'xyz':(0,1,2), 'xzy':(0,2,1), 'yxz':(1,0,2),
          'yzx':(1,2,0), 'zxy':(2,0,1), 'zyx':(2,1,0)}
    from .permute import permute_axes
    pv = [permute_axes(v, ao[axis_order], step, subregion, model_id)
          for v in volumes]
    return _volume_or_list(pv)

# -----------------------------------------------------------------------------
#
def volume_resample(session, volumes, on_grid = None, bounding_grid = False,
                    subregion = 'all', step = 1,
                    grid_subregion = 'all', grid_step = 1, spacing = None,
                    value_type = None, model_id = None, hide_maps = True):
    '''Interoplate a map on a new grid.'''
    if on_grid is None and spacing is None:
            raise CommandError('volume resample must specify onGrid option or spacing option')
    rv = []
    for v in volumes:
        if on_grid is None:
            on_grid = [v]
        for gv in on_grid:
            cv = combine_operation([v], 'add', subregion, step,
                                   gv, grid_subregion, grid_step, spacing, value_type,
                                   bounding_grid, False, None, model_id, session,
                                   hide_maps = hide_maps)
            rv.append(cv)
    return _volume_or_list(rv)

# -----------------------------------------------------------------------------
#
def volume_ridges(session, volumes, level = None, subregion = 'all', step = 1, model_id = None):
    '''Find ridges in a map.'''
    from .ridges import ridges
    rv = [ridges(v, level, step, subregion, model_id)
          for v in volumes]
    return _volume_or_list(rv)

# -----------------------------------------------------------------------------
#
def volume_scale(session, volumes, shift = 0, factor = 1, sd = None, rms = None,
                 value_type = None, subregion = 'all', step = 1, model_id = None):
    '''Scale, shift and convert number type of map values.'''
    if not sd is None and not rms is None:
        raise CommandError('volume scale: Cannot specify both sd and rms options')

    from .scale import scaled_volume
    sv = [scaled_volume(v, factor, sd, rms, shift, value_type, step, subregion, model_id,
                        session = session)
          for v in volumes]
    return _volume_or_list(sv)

# -----------------------------------------------------------------------------
#
def volume_subtract(session, volumes, on_grid = None, bounding_grid = False,
                    subregion = 'all', step = 1,
                    grid_subregion = 'all', grid_step = 1, spacing = None, value_type = None,
                    in_place = False, scale_factors = None, min_rms = False,
                    model_id = None, hide_maps = True, open_model = True):
    '''Subtract two maps.'''
    if len(volumes) != 2:
        raise CommandError('volume subtract operation requires exactly two volumes')
    if min_rms and scale_factors:
        raise CommandError('volume subtract cannot specify both minRMS and scaleFactors options.')
    mult = (1,'minrms') if min_rms else scale_factors

    sv = combine_op(volumes, 'subtract', on_grid, bounding_grid, subregion, step,
                    grid_subregion, grid_step, spacing, value_type,
                    in_place, mult, model_id, session,
                    hide_maps = hide_maps, open_model=open_model)
    return sv

# -----------------------------------------------------------------------------
#
def volume_threshold(session, volumes, minimum = None, set = None,
                 maximum = None, set_maximum = None,
                 subregion = 'all', step = 1, model_id = None):
    '''Set map values below or above a threshold to a constant.'''
    from .threshold import threshold
    tv = [threshold(v, minimum, set, maximum, set_maximum,
                    step, subregion, model_id, session)
          for v in volumes]
    return _volume_or_list(tv)

# -----------------------------------------------------------------------------
#
def volume_tile(session, volumes, axis = 'z', pstep = 1, trim = 0,
            columns = None, rows = None, fill_order = 'ulh',
            subregion = 'shown', step = 1, model_id = None):
    '''Concatenate maps along an axis.'''
    from .tile import tile_planes
    tv = []
    for v in volumes:
        t = tile_planes(v, axis, pstep, trim, rows, columns, fill_order,
                        step, subregion, model_id)
        if t is None:
            raise CommandError('volume tile: no planes')
        tv.append(t)
    return _volume_or_list(tv)

# -----------------------------------------------------------------------------
#
def volume_unbend(session, volumes, path, yaxis = None, xsize = None, ysize = None, grid_spacing = None,
              subregion = 'all', step = 1, model_id = None):
    '''Unbend a map near a smooth splined path.'''
    if len(path) < 2:
        raise CommandError('volume unbend path must have 2 or more nodes')
    if yaxis is None:
        from chimerax.core.commands import Axis
        yaxis = Axis((0,1,0))
    from .unbend import atom_path, unbend_volume
    p = atom_path(path)
    rv = []
    for v in volumes:
        gs = min(v.data.step) if grid_spacing is None else grid_spacing
        yax = yaxis.scene_coordinates(v.position)
        xs = 10*gs if xsize is None else xsize
        ys = 10*gs if ysize is None else ysize
        uv = unbend_volume(v, p, yax, xs, ys, gs, subregion, step, model_id)
        rv.append(uv)
    return _volume_or_list(rv)

# -----------------------------------------------------------------------------
#
def volume_unroll(session, volumes, inner_radius = None, outer_radius = None, length = None,
              grid_spacing = None, axis = None, center = None, coordinate_system = None,
              subregion = 'all', step = (1,1,1), model_id = None):
    '''Flatten a cylindrical shell within a map.'''
    rv = []
    for v in volumes:
        a, c = axis_and_center(axis, center, coordinate_system, v.position)
        r0, r1, h = parse_cylinder_size(inner_radius, outer_radius, length,
                                        c, a, v, subregion, step)
        gsp = parse_grid_spacing(grid_spacing, v, step)
        from . import unroll
        uv = unroll.unroll_operation(v, r0, r1, h, c, a, gsp,
                                     subregion, step, model_id)
        rv.append(uv)
    return _volume_or_list(rv)

# -----------------------------------------------------------------------------
#
def axis_and_center(axis, center, coordinate_system, to_coords):
    a = (0,0,1)
    c = (0,0,0)
    if axis:
        asc = axis.scene_coordinates(coordinate_system or to_coords)
        a = to_coords.inverse().transform_vector(asc)
    if center:
        csc = center.scene_coordinates(coordinate_system or to_coords)
        c = to_coords.inverse() * csc
    return a, c

# -----------------------------------------------------------------------------
#
def parse_cylinder_size(inner_radius, outer_radius, length, center, axis,
                        v, subregion, step):

    if length is None:
        import numpy
        a = numpy.argmax(axis)
        xyz_min, xyz_max = v.xyz_bounds(step, subregion)
        h = xyz_max[a] - xyz_min[a]
    elif isinstance(length, (float, int)):
        h = length
    else:
        raise CommandError('length must be a number')
    if inner_radius is None or outer_radius is None:
        from . import unroll
        rmin, rmax = unroll.cylinder_radii(v, center, axis)
        pad = 0.10
        r0 = rmin * (1 - pad) if inner_radius is None else inner_radius
        r1 = rmax * (1 + pad) if outer_radius is None else outer_radius
    else:
        r0, r1 = inner_radius, outer_radius
    if not isinstance(r0, (float, int)):
        raise CommandError('inner_radius must be a number')
    if not isinstance(r1, (float, int)):
        raise CommandError('outer_radius must be a number')

    return r0, r1, h

# -----------------------------------------------------------------------------
#
def parse_grid_spacing(grid_spacing, v, step):

    if grid_spacing is None:
        gsz = min([s*st for s, st in zip(v.data.step, step)])
    elif isinstance(grid_spacing, (int, float)):
        gsz = grid_spacing
    else:
        raise CommandError('grid_spacing must be a number')
    return gsz

# -----------------------------------------------------------------------------
#
def volume_flip(session, volumes, axis = 'z', subregion = 'all', step = 1,
            in_place = False, model_id = None):
    '''Flip a map axis reversing the hand.'''
    fv = [flip_operation(v, axis, subregion, step, in_place, model_id)
          for v in volumes]
    return _volume_or_list(fv)

# -----------------------------------------------------------------------------
#
def flip_operation(v, axes, subregion, step, in_place, model_id):

    from . import flip
    if in_place:
        if not v.data.writable:
            raise CommandError("Can't flip volume opened from a file in-place: %s" % v.name)
        if subregion != 'all' or step != 1:
            raise CommandError("Can't flip a subregion of a volume in-place: %s" % v.name)
        m = v.data.full_matrix()
        flip.flip_in_place(m, axes)
        v.data.values_changed()
        return v
    else:
        g = v.grid_data(subregion = subregion, step = step, mask_zone = False)
        fg = flip.FlipGrid(g, axes)
        from chimerax.map import volume_from_grid_data
        fv = volume_from_grid_data(fg, v.session, model_id = model_id)
        fv.copy_settings_from(v, copy_region = False)
        v.display = False
        return fv

# -----------------------------------------------------------------------------
# Return center in submatrix index units.
#
def submatrix_center(v, xyz_center, index_center, subregion, step):

    ijk_min, ijk_max, step = v.subregion(step, subregion)

    if index_center is None:
        if xyz_center is None:
            index_center = [0.5*(ijk_min[a] + ijk_max[a]) for a in range(3)]
        else:
            index_center = v.data.xyz_to_ijk(xyz_center)

    ioffset = map(lambda a,s: ((a+s-1)/s)*s, ijk_min, step)

    sic = tuple(map(lambda a,b,s: (a-b)/s, index_center, ioffset, step))
    return sic

# -----------------------------------------------------------------------------
#
def volume_zone(session, volumes, near_atoms, range = 3, bond_point_spacing = None,
                minimal_bounds = False, new_map = False, invert = False,
                subregion = 'all', step = 1, model_id = None):

    '''Mask a map keeping only parts close to specified atoms.'''
    if len(near_atoms) == 0:
        raise CommandError('no atoms specified for zone')

    if invert and not new_map:
        raise CommandError('volume zone command currently does not support invert True with newMap False')
    from .zone import zone_operation
    zv = [zone_operation(v, near_atoms, range, bond_point_spacing,
                         minimal_bounds, new_map, invert, subregion, step, model_id)
          for v in volumes]
    return _volume_or_list(zv)

# -----------------------------------------------------------------------------
#
def volume_unzone(session, volumes = None):
    '''Stop restricting volume surface display to a zone.'''

    if volumes is None:
        from chimerax.map import Volume
        volumes = [v for v in session.models if isinstance(v,Volume)]

    from chimerax.surface.zone import surface_unzone
    for v in volumes:
        r = v.full_region()
        v.new_region(r[0], r[1])
        for s in v.surfaces:
            surface_unzone(s)

def _volume_or_list(volumes):
    return volumes[0] if len(volumes) == 1 else volumes
