# -----------------------------------------------------------------------------
# Command to perform volume operations that create a new volume, such as
# erasing an octant, Gaussian filtering, adding maps, ....
#
#   Syntax: vop <operation> <volumeSpec>
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
#               [scale_factors <f1,f2,...>]
#
# where op is one of octant, ~octant, resample, add, zFlip, subtract, fourier,
# laplacian, gaussian, permuteAxes, bin, median, scale, boxes, morph, cover,
# flatten, unbend, tile.
#
from ...errors import UserError as CommandError

def register_vop_command():

    from ...commands import CmdDesc, register, BoolArg, EnumOf, IntArg, Int3Arg
    from ...commands import FloatArg, Float3Arg, FloatsArg, ModelIdArg, AtomsArg
    from ..mapargs import MapsArg, MapStepArg, MapRegionArg, Float1or3Arg, ValueTypeArg
    from ..mapargs import BoxArg, Float2Arg

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
    ] + ssm_kw
    add_kw = resample_kw + [
        ('in_place', BoolArg),
        ('scale_factors', FloatsArg),
    ]
    add_desc = CmdDesc(required = varg, keyword = add_kw)
    register('vop add', add_desc, vop_add)

    bin_desc = CmdDesc(required = varg,
                       keyword = [('bin_size', MapStepArg)] + ssm_kw
    )
    register('vop bin', bin_desc, vop_bin)

    boxes_desc = CmdDesc(required = varg + [('markers', AtomsArg)],
                         keyword = [('size', FloatArg),
                                    ('use_marker_size', BoolArg)] + ssm_kw)
    register('vop boxes', boxes_desc, vop_boxes)

    cover_desc = CmdDesc(required = varg,
        keyword = [('atom_box', AtomsArg),
                   ('pad', FloatArg),
                   ('box', BoxArg), ('x', Float2Arg), ('y', Float2Arg), ('z', Float2Arg),
                   ('fbox', BoxArg), ('fx', Float2Arg), ('fy', Float2Arg), ('fz', Float2Arg),
                   ('ibox', BoxArg), ('ix', Float2Arg), ('iy', Float2Arg), ('iz', Float2Arg),
                   ('use_symmetry', BoolArg),
                   ('cell_size', Int3Arg),
                   ('step', MapStepArg),
                   ('model_id', ModelIdArg)]
    )
    register('vop cover', cover_desc, vop_cover)

    falloff_desc = CmdDesc(required = varg,
        keyword = [('iterations', IntArg), ('in_place', BoolArg)] + ssm_kw
    )
    register('vop falloff', falloff_desc, vop_falloff)

    flatten_desc = CmdDesc(required = varg,
                           keyword = [('method', EnumOf(('multiplyLinear', 'divideLinear'))),
                                      ('fitregion', MapRegionArg)] + ssm_kw)
    register('vop flatten', flatten_desc, vop_flatten)

    flip_desc = CmdDesc(required = varg,
                        keyword = [('axis', EnumOf(('x','y','z','xy', 'yz','xyz'))),
                                   ('in_place', BoolArg)] + ssm_kw)
    register('vop flip', flip_desc, vop_flip)

    fourier_desc = CmdDesc(required = varg,
                           keyword = [('phase', BoolArg)] + ssm_kw)
    register('vop fourier', fourier_desc, vop_fourier)

    gaussian_desc = CmdDesc(required = varg,
        keyword = [('s_dev', Float1or3Arg), ('value_type', ValueTypeArg)] + ssm_kw
    )
    register('vop gaussian', gaussian_desc, vop_gaussian)

    laplacian_desc = CmdDesc(required = varg, keyword = ssm_kw)
    register('vop laplacian', laplacian_desc, vop_laplacian)

    localcorr_desc = CmdDesc(required = varg,
                             keyword = [('window_size', IntArg),
                                        ('subtract_mean', BoolArg),
                                        ('model_id', ModelIdArg)])
    register('vop local_correlation', localcorr_desc, vop_local_correlation)

    maximum_desc = CmdDesc(required = varg, keyword = add_kw)
    register('vop maximum', maximum_desc, vop_maximum)

    median_desc = CmdDesc(required = varg,
                          keyword = [('bin_size', MapStepArg),
                                     ('iterations', IntArg)] + ssm_kw)
    register('vop median', median_desc, vop_median)

    morph_desc = CmdDesc(required = varg,
                         keyword = [('frames', IntArg),
                                    ('start', FloatArg),
                                    ('play_step', FloatArg),
                                    ('play_direction', IntArg),
                                    ('play_range', Float2Arg),
                                    ('add_mode', BoolArg),
                                    ('constant_volume', BoolArg),
                                    ('scale_factors', FloatsArg),
                                    ('hide_original_maps', BoolArg),
                                    ('interpolate_colors', BoolArg)] + ssm_kw)
    register('vop morph', morph_desc, vop_morph)

    multiply_desc = CmdDesc(required = varg, keyword = add_kw)
    register('vop multiply', multiply_desc, vop_multiply)

    oct_kw = [('center', Float3Arg),
              ('i_center', Int3Arg),
              ('fill_value', FloatArg),
              ('in_place', BoolArg)]
    octant_desc = CmdDesc(required = varg,
                          keyword = oct_kw + ssm_kw)
    register('vop octant', octant_desc, vop_octant)

    unoctant_desc = CmdDesc(required = varg,
                            keyword = oct_kw + ssm_kw)
    register('vop ~octant', unoctant_desc, vop_octant_complement)

    permuteaxes_desc = CmdDesc(required = varg,
                               keyword = [('axis_order', EnumOf(('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')))] + ssm_kw)
    register('vop permuteaxes', permuteaxes_desc, vop_permute_axes)

    resample_desc = CmdDesc(required = varg, keyword = resample_kw)
    register('vop resample', resample_desc, vop_resample)

    ridges_desc = CmdDesc(required = varg,
                          keyword = [('level', FloatArg)] + ssm_kw)
    register('vop ridges', ridges_desc, vop_ridges)

    scale_desc = CmdDesc(required = varg,
                         keyword = [('shift', FloatArg),
                                    ('factor', FloatArg),
                                    ('sd', FloatArg),
                                    ('rms', FloatArg),
                                    ('value_type', ValueTypeArg),
                                    ('type', ValueTypeArg)] + ssm_kw)
    register('vop scale', scale_desc, vop_scale)

    subtract_desc = CmdDesc(required = varg,
                            keyword = add_kw + [('min_rms', BoolArg)])
    register('vop subtract', subtract_desc, vop_subtract)

    threshold_desc = CmdDesc(required = varg,
        keyword = [('minimum', FloatArg), ('set', FloatArg),
                   ('maximum', FloatArg), ('set_maximum', FloatArg)] + ssm_kw
    )
    register('vop threshold', threshold_desc, vop_threshold)

    orders = ('ulh', 'ulv', 'urh', 'urv', 'llh', 'llv', 'lrh', 'lrv',
              'ulhr', 'ulvr', 'urhr', 'urvr', 'llhr', 'llvr', 'lrhr', 'lrvr')
    tile_desc = CmdDesc(required = varg,
                        keyword = [('axis', EnumOf(('x','y','z'))),
                                   ('pstep', IntArg),
                                   ('trim', IntArg),
                                   ('columns', IntArg),
                                   ('rows', IntArg),
                                   ('fill_order', EnumOf(orders))] + ssm_kw)
    register('vop tile', tile_desc, vop_tile)

    unbend_desc = CmdDesc(required = varg + [('path', AtomsArg),
                                             ('yaxis', Float3Arg),
                                             ('xsize', FloatArg),
                                             ('ysize', FloatArg)],
                          keyword = [('grid_spacing', FloatArg)] + ssm_kw)
    register('vop unbend', unbend_desc, vop_unbend)

    unroll_desc = CmdDesc(required = varg,
                          keyword = [('inner_radius', FloatArg),
                                     ('outer_radius', FloatArg),
                                     ('length', FloatArg),
                                     ('grid_spacing', FloatArg),
                                     ('axis', Float3Arg),
                                     ('center', Float3Arg)] + ssm_kw)
    register('vop unroll', unroll_desc, vop_unroll)

    zflip_desc = CmdDesc(required = varg,
                         keyword = [('axis', EnumOf(('x','y','z','xy', 'yz','xyz'))),
                                    ('in_place', BoolArg)] + ssm_kw)
    register('vop zFlip', zflip_desc, vop_flip)

    zone_desc = CmdDesc(required = varg + [('atoms', AtomsArg), ('radius', FloatArg)],
                        keyword = [('bond_point_spacing', FloatArg),
                                   ('minimal_bounds', BoolArg),
                                   ('invert', BoolArg)] + ssm_kw)
    register('vop zone', zone_desc, vop_zone)

# -----------------------------------------------------------------------------
#
def vop_add(session, volumes, on_grid = None, bounding_grid = None,
           subregion = 'all', step = 1,
           grid_subregion = 'all', grid_step = 1,
           in_place = False, scale_factors = None, model_id = None):
    '''Add maps.'''
    combine_op(volumes, 'add', on_grid, bounding_grid, subregion, step,
               grid_subregion, grid_step, in_place, scale_factors, model_id, session)

# -----------------------------------------------------------------------------
#
def vop_maximum(session, volumes, on_grid = None, bounding_grid = None,
               subregion = 'all', step = 1,
               grid_subregion = 'all', grid_step = 1,
               in_place = False, scale_factors = None, model_id = None):
    '''Pointwise maximum of maps.'''
    combine_op(volumes, 'maximum', on_grid, bounding_grid, subregion, step,
               grid_subregion, grid_step, in_place, scale_factors, model_id, session)

# -----------------------------------------------------------------------------
#
def vop_multiply(session, volumes, on_grid = None, bounding_grid = None,
                subregion = 'all', step = 1,
                grid_subregion = 'all', grid_step = 1,
                in_place = False, scale_factors = None, model_id = None):
    '''Pointwise multiply maps.'''
    combine_op(volumes, 'multiply', on_grid, bounding_grid, subregion, step,
               grid_subregion, grid_step, in_place, scale_factors, model_id, session)

# -----------------------------------------------------------------------------
#
def combine_op(volumes, operation = 'add', on_grid = None, bounding_grid = None,
               subregion = 'all', step = 1,
               grid_subregion = 'all', grid_step = 1,
               in_place = False, scale_factors = None, model_id = None, session = None):

    if bounding_grid is None and not in_place:
        bounding_grid = (on_grid is None)
    if on_grid is None:
        on_grid = volumes[:1]
    if in_place:
        if bounding_grid or grid_step != 1 or grid_subregion != 'all':
            raise CommandError("Can't use in_place option with bounding_grid or grid_step or grid_subregion options")
        for gv in on_grid:
            if not gv.data.writable:
                raise CommandError("Can't modify volume in place: %s" % gv.name)
            if not gv in volumes:
                raise CommandError("Can't change grid in place")
    if not scale_factors is None and len(scale_factors) != len(volumes):
        raise CommandError('Number of scale factors does not match number of volumes')
    for gv in on_grid:
        combine_operation(volumes, operation, subregion, step,
                          gv, grid_subregion, grid_step,
                          bounding_grid, in_place, scale_factors, model_id, session)

# -----------------------------------------------------------------------------
#
def combine_operation(volumes, operation, subregion, step,
                      gv, grid_subregion, grid_step,
                      bounding_grid, in_place, scale, model_id, session):

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
        value_type = v0.data.value_type if volumes else gv.data.value_type
        rg = gv.region_grid(r, value_type)
        if len(volumes) == 1:
            rg.name = v0.name + ' resampled'
        elif operation == 'subtract':
            rg.name = 'volume difference'
            rg.polar_values = True
        elif operation == 'maximum':
            rg.name = 'volume maximum'
        else:
            rg.name = 'volume sum'
        from .. import volume_from_grid_data
        rv = volume_from_grid_data(rg, session, model_id = model_id,
                                   show_data = False, show_dialog = False)
        rv.position = gv.position
        for i,v in enumerate(volumes):
            op = 'add' if i == 0 else operation
            rv.combine_interpolated_values(v, op, subregion = subregion, step = step,
                                           scale = scale[i])
    rv.data.values_changed()
    if volumes:
        rv.copy_settings_from(v0, copy_region = False, copy_xform = False)
        if rv.data.name.endswith('difference'):
            rv.set_parameters(cap_faces = False)

    rv.show()
    for v in volumes:
        if not v is rv:
            v.unshow()

# -----------------------------------------------------------------------------
#
def same_grids(volumes, subregion, step, gv, gr):

    from ..volume import same_grid
    for v in volumes:
        if not same_grid(v, v.subregion(step, subregion), gv, gr):
            return False
    return True

# -----------------------------------------------------------------------------
#
def volume_corners(volumes, subregion, step, place):

    from ..data import box_corners
    corners = []
    for v in volumes:
        xyz_min, xyz_max = v.xyz_bounds(step = step, subregion = subregion)
        vc = box_corners(xyz_min, xyz_max)
        c = place.inverse() * v.position * vc
        corners.extend(c)
    return corners

# -----------------------------------------------------------------------------
#
def vop_bin(session, volumes, subregion = 'all', step = 1,
           bin_size = (2,2,2), model_id = None):
    '''Reduce map by averaging over rectangular bins.'''
    from .bin import bin
    for v in volumes:
        bin(v, bin_size, step, subregion, model_id, session)

# -----------------------------------------------------------------------------
#
def vop_boxes(session, volumes, markers, size = 0, use_marker_size = False,
             subregion = 'all', step = 1, model_id = None):
    '''Extract boxes centered at marker positions.'''
    if size <= 0 and not use_marker_size:
        raise CommandError('Must specify size or enable use_marker_size')

    from .boxes import boxes
    for v in volumes:
        boxes(v, markers, size, use_marker_size, step, subregion, model_id)

# -----------------------------------------------------------------------------
#
def vop_cover(session, volumes, atom_box = None, pad = 5.0, 
             box = None, x = None, y = None, z = None,
             f_box = None, fx = None, fy = None, fz = None,
             i_box = None, ix = None, iy = None, iz = None,
             use_symmetry = True, cell_size = None,
             step = 1, model_id = None):
    '''Extend a map using symmetry to cover a specified region.'''
    if not atom_box is None and len(atom_box) == 0:
        raise CommandError('No atoms specified')
    box = parse_box(box, x, y, z, 'box', 'x', 'y', 'z')
    f_box = parse_box(f_box, fx, fy, fz, 'f_box', 'fx', 'fy', 'fz')
    i_box = parse_box(i_box, ix, iy, iz, 'i_box', 'ix', 'iy', 'iz')
    bc = len([b for b in (box, f_box, i_box, atom_box) if b])
    if bc == 0:
        raise CommandError('Must specify box to cover')
    if bc > 1:
        raise CommandError('Specify covering box in one way')

    from .. import volume_from_grid_data
    from .cover import cover_box_bounds, map_covering_box
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
        cv.show()

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
def vop_falloff(session, volumes, iterations = 10, in_place = False,
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
    for v in volumes:
        falloff(v, iterations, in_place, step, subregion, model_id, session)

# -----------------------------------------------------------------------------
#
def vop_flatten(session, volumes, method = 'multiplyLinear',
               fitregion = None, subregion = 'all', step = 1, model_id = None):
    '''Make map background flat.'''
    if fitregion is None:
        fitregion = subregion

    from .flatten import flatten
    for v in volumes:
        flatten(v, method, step, subregion, fitregion, model_id)

# -----------------------------------------------------------------------------
#
def vop_fourier(session, volumes, subregion = 'all', step = 1, model_id = None, phase = False):
    '''Fourier transform a map'''
    from .fourier import fourier_transform
    for v in volumes:
        fourier_transform(v, step, subregion, model_id, phase)

# -----------------------------------------------------------------------------
#
def vop_gaussian(session, volumes, s_dev = (1.0,1.0,1.0),
                subregion = 'all', step = 1, value_type = None, model_id = None):
    '''Smooth maps by Gaussian convolution.'''
    from .gaussian import gaussian_convolve
    for v in volumes:
        gaussian_convolve(v, s_dev, step, subregion, value_type, model_id, session = session)

# -----------------------------------------------------------------------------
#
def vop_laplacian(session, volumes, subregion = 'all', step = 1, model_id = None):
    '''Detect map edges with Laplacian filter.'''
    from .laplace import laplacian
    for v in volumes:
        laplacian(v, step, subregion, model_id)

# -----------------------------------------------------------------------------
#
def vop_local_correlation(session, volumes, window_size = 5, subtract_mean = False, model_id = None):
    '''Compute correlation between two maps over a sliding window.'''
    if len(volumes) != 2:
        raise CommandError('vop local_correlation operation requires '
                           'exactly two map arguments')
    v1, v2 = volumes
    if window_size < 2:
        raise CommandError('vop local_correlation window_size must be '
                           'an integer >= 2')
    if windowSize > min(v1.data.size):
        raise CommandError('vop local_correlation window_size must be '
                           'smaller than map size')

    from .localcorr import local_correlation
    mapc = local_correlation(v1, v2, window_size, subtract_mean, model_id)
    return mapc

# -----------------------------------------------------------------------------
#
def vop_median(session, volumes, bin_size = 3, iterations = 1,
              subregion = 'all', step = 1, model_id = None):
    '''Replace map values with median of neighboring values.'''
    for b in bin_size:
        if b <= 0 or b % 2 == 0:
            raise CommandError('Bin size must be positive odd integer, got %d' % b)

    from .median import median_filter
    for v in volumes:
        median_filter(v, bin_size, iterations, step, subregion, model_id)

# -----------------------------------------------------------------------------
#
def vop_morph(session, volumes, frames = 25, start = 0, play_step = 0.04,
             play_direction = 1, play_range = None, add_mode = False,
             constant_volume = False, scale_factors = None,
             hide_original_maps = True, interpolate_colors = True,
             subregion = 'all', step = 1, model_id = None):
    '''Linearly interpolate pointwise between maps.'''
    if play_range is None:
        if add_mode:
            prange = (-1.0,1.0)
        else:
            prange = (0.0,1.0)
    else:
        prange = play_range

    if not scale_factors is None and len(volumes) != len(scale_factors):
        raise CommandError('Number of scale factors (%d) doesn not match number of volumes (%d)'
                           % (len(scale_factors), len(volumes)))
    vs = [tuple(v.matrix_size(step = step, subregion = subregion))
          for v in volumes]
    if len(set(vs)) > 1:
        sizes = ' and '.join([str(s) for s in vs])
        raise CommandError("Volume grid sizes don't match: %s" % sizes)
    from MorphMap import morph_maps
    morph_maps(volumes, frames, start, play_step, play_direction, prange,
               add_mode, constant_volume, sfactors,
               hide_original_maps, interpolate_colors, subregion, step, model_id)
        
# -----------------------------------------------------------------------------
#
def vop_octant(session, volumes, center = None, i_center = None,
              subregion = 'all', step = 1, in_place = False,
              fill_value = 0, model_id = None):
    '''Extract an octant from a map.'''
    check_in_place(in_place, volumes)
    outside = True
    for v in volumes:
        octant_operation(v, outside, center, i_center, subregion, step,
                         in_place, fill_value, model_id)

# -----------------------------------------------------------------------------
#
def vop_octant_complement(session, volumes, center = None, i_center = None,
                         subregion = 'all', step = 1, in_place = False,
                         fill_value = 0, model_id = None):
    '''Zero an octant of a map.'''
    check_in_place(in_place, volumes)
    outside = False
    for v in volumes:
        octant_operation(v, outside, center, i_center, subregion, step,
                         in_place, fill_value, model_id)

# -----------------------------------------------------------------------------
#
def octant_operation(v, outside, center, i_center,
                     subregion, step, in_place, fill_value, model_id):

    vc = v.writable_copy(require_copy = not in_place,
                         subregion = subregion, step = step,
                         model_id = model_id, show = False)
    ic = submatrix_center(v, center, i_center, subregion, step)
    ijk_max = [i-1 for i in vc.data.size]
    from VolumeEraser import set_box_value
    set_box_value(vc.data, fill_value, ic, ijk_max, outside)
    vc.data.values_changed()
    vc.show()

# -----------------------------------------------------------------------------
#
def vop_permute_axes(session, volumes, axis_order = 'xyz',
                    subregion = 'all', step = 1, model_id = None):
    '''Permute map axes.'''
    ao = {'xyz':(0,1,2), 'xzy':(0,2,1), 'yxz':(1,0,2), 
          'yzx':(1,2,0), 'zxy':(2,0,1), 'zyx':(2,1,0)}
    from .permute import permute_axes
    for v in volumes:
        permute_axes(v, ao[axis_order], step, subregion, model_id)

# -----------------------------------------------------------------------------
#
def vop_resample(session, volumes, on_grid = None, bounding_grid = False,
                subregion = 'all', step = 1,
                grid_subregion = 'all', grid_step = 1,
                model_id = None):
    '''Interoplate a map on a new grid.'''
    if on_grid is None:
        raise CommandError('Resample operation must specify onGrid option')
    for v in volumes:
        for gv in on_grid:
            combine_operation([v], 'add', subregion, step,
                              gv, grid_subregion, grid_step,
                              bounding_grid, False, None, model_id, session)

# -----------------------------------------------------------------------------
#
def vop_ridges(session, volumes, level = None, subregion = 'all', step = 1, model_id = None):
    '''Find ridges in a map.'''
    from .ridges import ridges
    for v in volumes:
        ridges(v, level, step, subregion, model_id)

# -----------------------------------------------------------------------------
#
def vop_scale(session, volumes, shift = 0, factor = 1, sd = None, rms = None,
             value_type = None, type = None,
             subregion = 'all', step = 1, model_id = None):
    '''Scale, shift and convert number type of map values.'''
    if not sd is None and not rms is None:
        raise CommandError('vop scale: Cannot specify both sd and rms options')
    value_type = type if value_type is None else value_type

    from .scale import scaled_volume
    for v in volumes:
        scaled_volume(v, factor, sd, rms, shift, value_type, step, subregion, model_id)

# -----------------------------------------------------------------------------
#
def vop_subtract(session, volumes, on_grid = None, bounding_grid = False,
                subregion = 'all', step = 1,
                grid_subregion = 'all', grid_step = 1,
                in_place = False, scale_factors = None, min_rms = False,
                model_id = None):
    '''Subtract two maps.'''
    if len(volumes) != 2:
        raise CommandError('vop subtract operation requires exactly two volumes')
    if min_rms and scale_factors:
        raise CommandError('vop subtract cannot specify both minRMS and scaleFactors options.')
    mult = (1,'minrms') if min_rms else scale_factors

    combine_op(volumes, 'subtract', on_grid, bounding_grid, subregion, step,
               grid_subregion, grid_step, in_place, mult, model_id, session)

# -----------------------------------------------------------------------------
#
def vop_threshold(session, volumes, minimum = None, set = None,
                 maximum = None, set_maximum = None,
                 subregion = 'all', step = 1, model_id = None):
    '''Set map values below or above a threshold to a constant.'''
    from .threshold import threshold
    for v in volumes:
        threshold(v, minimum, set, maximum, set_maximum,
                  step, subregion, model_id, session)

# -----------------------------------------------------------------------------
#
def vop_tile(session, volumes, axis = 'z', pstep = 1, trim = 0,
            columns = None, rows = None, fill_order = 'ulh',
            subregion = 'shown', step = 1, model_id = None):
    '''Concatenate maps along an axis.'''
    from .tile import tile_planes
    for v in volumes:
        t = tile_planes(v, axis, pstep, trim, rows, columns, fill_order,
                        step, subregion, model_id)
        if t is None:
            raise CommandError('vop tile: no planes')

# -----------------------------------------------------------------------------
#
def vop_unbend(session, volumes, path, yaxis, xsize, ysize, grid_spacing = None,
              subregion = 'all', step = 1, model_id = None):
    '''Unbend a map near a smooth splined path.'''
    if len(path) < 2:
        raise CommandError('vop unbend path must have 2 or more nodes')

    from . import unbend
    p = unbend.atom_path(path)
    for v in volumes:
        gs = min(v.data.step) if grid_spacing is None else grid_spacing
        yax = v.place * yaxis
        unbend.unbend_volume(v, p, yax, xsize, ysize, gs,
                             subregion, step, model_id)

# -----------------------------------------------------------------------------
#
def vop_unroll(session, volumes, inner_radius = None, outer_radius = None, length = None,
              grid_spacing = None, axis = (0,0,1), center = (0,0,0),
              subregion = 'all', step = 1, model_id = None):
    '''Flatten a cylindrical shell within a map.'''
    for v in volumes:
        r0, r1, h = parse_cylinder_size(inner_radius, outer_radius, length,
                                        center, axis, v, subregion, step)
        gsp = parse_grid_spacing(grid_spacing, v, step)
        from . import unroll
        unroll.unroll_operation(v, r0, r1, h, center, axis, gsp,
                                subregion, step, model_id)

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
def parse_grid_spacing(gridSpacing, v, step):

    if grid_spacing is None:
        gsz = min([s*st for s, st in zip(v.data.step, step)])
    elif isinstance(grid_spacing, (int, float)):
        gsz = grid_spacing
    else:
        raise CommandError('grid_spacing must be a number')
    return gsz

# -----------------------------------------------------------------------------
#
def vop_flip(session, volumes, axis = 'z', subregion = 'all', step = 1,
            in_place = False, model_id = None):
    '''Flip a map axis reversing the hand.'''
    for v in volumes:
        flip_operation(v, axis, subregion, step, in_place, model_id)
        
# -----------------------------------------------------------------------------
#
def flip_operation(v, axes, subregion, step, in_place, model_id):

    g = v.grid_data(subregion = subregion, step = step, mask_zone = False)
    from . import flip
    if in_place:
        m = g.full_matrix()
        flip.flip_in_place(m, axes)
        v.data.values_changed()
        v.show()
    else:
        fg = flip.Flip_Grid(g, axes)
        from .. import volume_from_grid_data
        fv = volume_from_grid_data(fg, session, model_id = model_id)
        fv.copy_settings_from(v, copy_region = False)
        fv.show()
        v.unshow()

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
def vop_zone(session, volumes, atoms, radius, bond_point_spacing = None,
            minimal_bounds = False, invert = False,
            subregion = 'all', step = 1, model_id = None):

    '''Mask a map keeping only parts close to specified atoms.'''
    if len(atoms) == 0:
        raise CommandError('no atoms specified for zone')

    for v in volumes:
        zone_operation(v, atoms, radius, bond_point_spacing,
                       minimal_bounds, invert, subregion, step, model_id)

# -----------------------------------------------------------------------------
#
def zone_operation(v, atoms, radius, bond_point_spacing = None,
                   minimal_bounds = False, invert = False,
                   subregion = 'all', step = 1, model_id = None):

    from ... import molecule as M
    bonds = M.interatom_bonds(atoms) if bond_point_spacing else []

    import SurfaceZone as SZ
    points = SZ.path_points(atoms, bonds, v.place.inverse(),
                            bond_point_spacing)

    vz = zone_volume(v, points, radius, minimal_bounds, invert,
                     subregion, step, model_id)
    return vz

# -----------------------------------------------------------------------------
#
def zone_volume(volume, points, radius,
                minimal_bounds = False, invert = False,
                subregion = 'all', step = 1, model_id = None):

    region = volume.subregion(step, subregion)
    from .. import data
    sg = data.Grid_Subregion(volume.data, *region)

    mg = data.zone_masked_grid_data(sg, points, radius, invert, minimal_bounds)
    mg.name = volume.name + ' zone'

    from .. import volume_from_grid_data
    vz = volume_from_grid_data(mg, session, model_id = model_id, show_data = False)
    vz.copy_settings_from(volume, copy_colors = False, copy_zone = False)
    vz.show()
    volume.unshow()

    return vz
