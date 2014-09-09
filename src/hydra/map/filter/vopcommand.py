# -----------------------------------------------------------------------------
# Command to perform volume operations that create a new volume, such as
# erasing an octant, Gaussian filtering, adding maps, ....
#
#   Syntax: vop <operation> <volumeSpec>
#               [radius <r>]
#               [center <x,y,z>]
#               [iCenter <i,j,k>]
#               [fillValue <v>]
#               [sDev <d>]
#               [onGrid <id>]
#               [boundingGrid true|false>]
#               [axisOrder xyz|yxz|zxy|zyx|yzx|xzy]
#               [modelId <n>]
#               [inPlace true|false]
#               [subregion all|<i1,j1,k1,i2,j2,k2>]
#               [step <i>|<i,j,k>]
#               [gridSubregion all|<i1,j1,k1,i2,j2,k2>]
#               [gridStep <i>|<i,j,k>]
#               [frames <n>]
#               [start <f0>]
#               [playStep <fstep>]
#               [playDirection 1|-1]
#               [playRange <fmin,fmax>]
#               [addMode true|false]
#               [constantVolume true|false]
#               [scaleFactors <f1,f2,...>]
#
# where op is one of octant, ~octant, resample, add, zFlip, subtract, fourier,
# laplacian, gaussian, permuteAxes, bin, median, scale, boxes, morph, cover,
# flatten, unbend, tile.
#
def vop_command(cmd_name, args, session):

    from ...commands.parse import volumes_arg, volume_arg, int1or3_arg, int_arg
    from ...commands.parse import volume_region_arg, model_id_arg, perform_operation
    from ...commands.parse import float1or3_arg, floats_arg, float_arg, value_type_arg, bool_arg
    ops = {
        'bin': (bin_op,
                (('volumes', volumes_arg),),
                (),
                (('binSize', int1or3_arg),
                 ('subregion', volume_region_arg),
                 ('step', int1or3_arg),
                 ('modelId', model_id_arg),)),
        'falloff': (falloff_op,
                    (('volumes', volumes_arg),),
                    (),
                    (('iterations', int_arg),
                     ('inPlace', bool_arg),
                     ('subregion', volume_region_arg),
                     ('step', int1or3_arg),
                       ('modelId', model_id_arg),)),
        'gaussian': (gaussian_op,
                (('volumes', volumes_arg),),
                (),
                (('sDev', float1or3_arg),
                 ('subregion', volume_region_arg),
                 ('step', int1or3_arg),
                 ('valueType', value_type_arg),
                 ('modelId', model_id_arg),)),
        'maximum': (maximum_op,
                    (('volumes', volumes_arg),),
                    (),
                    (('onGrid', volumes_arg),
                     ('boundingGrid', bool_arg),
                     ('subregion', volume_region_arg),
                     ('step', int1or3_arg),
                     ('gridSubregion', volume_region_arg),
                     ('gridStep', int1or3_arg),
                     ('inPlace', bool_arg),
                     ('scaleFactors', floats_arg),
                     ('modelId', model_id_arg),)),
        'threshold': (threshold_op,
                      (('volumes', volumes_arg),),
                      (),
                      (('minimum', float_arg),
                       ('set', float_arg),
                       ('maximum', float_arg),
                       ('setMaximum', float_arg),
                       ('subregion', volume_region_arg),
                       ('step', int1or3_arg),
                       ('modelId', model_id_arg),)),
    }
    perform_operation(cmd_name, args, ops, session)

#
def old_vop_command(cmdname, args):

    vspec = ('volumeSpec','volumes','models')
    gspec = ('onGridSpec', 'onGrid', 'models')
    operations = {
        'add': (add_op, [vspec, gspec]),
        'boxes': (boxes_op, [vspec, ('markersSpec', 'markers', 'atoms')]),
        'cover': (cover_op, [vspec, ('atomBoxSpec', 'atomBox', 'atoms')]),
        'flatten': (flatten_op, [vspec]),
        'flip': (flip_op, [vspec]),
        'fourier': (fourier_op, [vspec]),
        'laplacian': (laplacian_op, [vspec]),
        'localCorrelation': (local_correlation_op,
                             [('map1Spec','map1','models'),
                              ('map2Spec','map2','models')]),
        'median': (median_op, [vspec]),
        'morph': (morph_op, [vspec]),
        'multiply': (multiply_op, [vspec, gspec]),
        'octant': (octant_op, [vspec]),
        '~octant': (octant_complement_op, [vspec]),
        'permuteAxes': (permute_axes_op, [vspec]),
        'resample': (resample_op, [vspec, gspec]),
        'ridges': (ridges_op, [vspec]),
        'scale': (scale_op, [vspec]),
        'subtract': (subtract_op, [('volume1Spec','vol1','models'),
                                   ('volume2Spec','vol2','models'),
                                   gspec]),
        'tile': (tile_op, [vspec]),
        'unbend': (unbend_op, [vspec, ('pathSpec', 'path', 'atoms')]),
        'unroll': (unroll_op, [vspec]),
        'zFlip': (flip_op, [vspec]),
        'zone': (zone_op, [vspec, ('atomSpec', 'atoms', 'atoms')]),
        }
    ops = operations.keys()

    sa = args.split(None, 2)
    if len(sa) < 2:
        raise CommandError('vop requires at least 2 arguments: vop <operation> <args...>')

    from Commands import parse_enumeration
    op = parse_enumeration(sa[0], ops)
    if op is None:
        # Handle old syntax where operation argument followed volume spec
        op = parse_enumeration(sa[1], ops)
        if op:
            sa = [sa[1], sa[0]] + sa[2:]
        else:
            raise CommandError('Unknown vop operation: %s' % sa[0])

    func, spec = operations[op]
    from Commands import doExtensionFunc
    fargs = ' '.join(sa[1:])
    doExtensionFunc(func, fargs, specInfo = spec)

# -----------------------------------------------------------------------------
#
def add_op(volumes, onGrid = None, boundingGrid = None,
           subregion = 'all', step = 1,
           gridSubregion = 'all', gridStep = 1,
           inPlace = False, scaleFactors = None, modelId = None, session = None):

    combine_op(volumes, 'add', onGrid, boundingGrid, subregion, step,
               gridSubregion, gridStep, inPlace, scaleFactors, modelId, session)

# -----------------------------------------------------------------------------
#
def maximum_op(volumes, onGrid = None, boundingGrid = None,
               subregion = 'all', step = 1,
               gridSubregion = 'all', gridStep = 1,
               inPlace = False, scaleFactors = None, modelId = None, session = None):

    combine_op(volumes, 'maximum', onGrid, boundingGrid, subregion, step,
               gridSubregion, gridStep, inPlace, scaleFactors, modelId, session)

# -----------------------------------------------------------------------------
#
def multiply_op(volumes, onGrid = None, boundingGrid = None,
                subregion = 'all', step = 1,
                gridSubregion = 'all', gridStep = 1,
                inPlace = False, scaleFactors = None, modelId = None, session = None):

    combine_op(volumes, 'multiply', onGrid, boundingGrid, subregion, step,
               gridSubregion, gridStep, inPlace, scaleFactors, modelId, session)

# -----------------------------------------------------------------------------
#
def combine_op(volumes, operation = 'add', onGrid = None, boundingGrid = None,
               subregion = 'all', step = 1,
               gridSubregion = 'all', gridStep = 1,
               inPlace = False, scaleFactors = None, modelId = None, session = None):

    if boundingGrid is None and not inPlace:
        boundingGrid = (onGrid is None)
    if onGrid is None:
        onGrid = volumes[:1]
    if inPlace:
        if boundingGrid or gridStep != 1 or gridSubregion != 'all':
            raise CommandError("Can't use inPlace option with boundingGrid or gridStep or gridSubregion options")
        for gv in onGrid:
            if not gv.data.writable:
                raise CommandError("Can't modify volume in place: %s" % gv.name)
            if not gv in volumes:
                raise CommandError("Can't change grid in place")
    if not scaleFactors is None and len(scaleFactors) != len(volumes):
        raise CommandError('Number of scale factors does not match number of volumes')
    for gv in onGrid:
        combine_operation(volumes, operation, subregion, step,
                          gv, gridSubregion, gridStep,
                          boundingGrid, inPlace, scaleFactors, modelId, session)

# -----------------------------------------------------------------------------
#
def combine_operation(volumes, operation, subregion, step,
                      gv, gridSubregion, gridStep,
                      boundingGrid, inPlace, scale, modelId, session):

    if scale is None:
        scale = [1]*len(volumes)
    if inPlace:
        rv = gv
        for i, v in enumerate(volumes):
            s = (scale[i] if v != rv else scale[i]-1)
            op = 'add' if i == 0 else operation
            rv.combine_interpolated_values(v, op, subregion = subregion,
                                           step = step, scale = s)
    else:
        gr = gv.subregion(step = gridStep, subregion = gridSubregion)
        if boundingGrid:
            if same_grids(volumes, subregion, step, gv, gr):
                # Avoid extending grid due to round-off errors.
                r = gr
            else:
                corners = volume_corners(volumes, subregion, step,
                                         gv.model_transform())
                r = gv.bounding_region(corners, step = gridStep, clamp = False)
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
        rv = volume_from_grid_data(rg, session, model_id = modelId,
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
def volume_corners(volumes, subregion, step, xform):

    from ..data import box_corners
    corners = []
    for v in volumes:
        xyz_min, xyz_max = v.xyz_bounds(step = step, subregion = subregion)
        vc = box_corners(xyz_min, xyz_max)
        from ..volume import transformed_points
        xf = xform.inverse()
        xf.multiply(v.model_transform())
        c = transformed_points(vc, xf)
        corners.extend(c)
    return corners

# -----------------------------------------------------------------------------
#
def bin_op(volumes, session, subregion = 'all', step = 1,
           binSize = (2,2,2), modelId = None):

    from .bin import bin
    for v in volumes:
        bin(v, binSize, step, subregion, modelId, session)

# -----------------------------------------------------------------------------
#
def boxes_op(volumes, markers, size = 0, useMarkerSize = False,
             subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    check_number(size, 'size')
    if size <= 0 and not useMarkerSize:
        raise CommandError('Must specify size or enable useMarkerSize')
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    if not modelId is None:
        modelId = parse_model_id(modelId)

    from .boxes import boxes
    for v in volumes:
        boxes(v, markers, size, useMarkerSize, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def cover_op(volumes, atomBox = None, pad = 5.0, 
             box = None, x = None, y = None, z = None,
             fBox = None, fx = None, fy = None, fz = None,
             iBox = None, ix = None, iy = None, iz = None,
             useSymmetry = True, cellSize = None,
             step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    check_number(pad, 'pad')

    if not atomBox is None and len(atomBox) == 0:
        raise CommandError('No atoms specified')
    box = parse_box(box, x, y, z, 'box', 'x', 'y', 'z')
    fBox = parse_box(fBox, fx, fy, fz, 'fBox', 'fx', 'fy', 'fz')
    iBox = parse_box(iBox, ix, iy, iz, 'iBox', 'ix', 'iy', 'iz')
    bc = len([b for b in (box, fBox, iBox, atomBox) if b])
    if bc == 0:
        raise CommandError('Must specify box to cover')
    if bc > 1:
        raise CommandError('Specify covering box in one way')

    if not cellSize is None:
        cellSize = parse_ints(cellSize, 'cellSize', 3)
    step = parse_step(step, require_3_tuple = True)
    modelId = parse_model_id(modelId)

    from .. import volume_from_grid_data
    from .cover import cover_box_bounds, map_covering_box
    for v in volumes:
        ijk_min, ijk_max = cover_box_bounds(v, step,
                                            atomBox, pad, box, fBox, iBox)
        ijk_cell_size = (getattr(v.data, 'unit_cell_size', v.data.size)
                         if cellSize is None else cellSize)
        syms = v.data.symmetries if useSymmetry else ()
        cg = map_covering_box(v, ijk_min, ijk_max, ijk_cell_size, syms, step)
                              
        cv = volume_from_grid_data(cg, session, model_id = modelId)
        cv.copy_settings_from(v, copy_region = False, copy_colors = False,
                              copy_zone = False)
        cv.show()

# -----------------------------------------------------------------------------
#
def parse_box(box, x, y, z, bname, xname, yname, zname):

    if box is None and x is None and y is None and z is None:
        return None
    if box:
        b6 = parse_floats(box, bname, 6)
        return (b6[:3], b6[3:])
    box = ([None,None,None], [None,None,None])
    for a,x,name in zip((0,1,2),(x,y,z),(xname,yname,zname)):
        if x:
            box[0][a], box[1][a] = parse_floats(x, name, 2)
    return box

# -----------------------------------------------------------------------------
#
def falloff_op(volumes, iterations = 10, inPlace = False,
               subregion = 'all', step = 1, modelId = None, session = None):

    if inPlace:
        ro = [v for v in volumes if not v.data.writable]
        if ro:
            raise CommandError("Can't use inPlace with read-only volumes "
                               + ', '.join(v.name for v in ro))
        if step != 1:
            raise CommandError('Step must be 1 to modify data in place')
        if subregion != 'all':
            raise CommandError('Require subregion "all" to modify data in place')

    from .falloff import falloff
    for v in volumes:
        falloff(v, iterations, inPlace, step, subregion, modelId, session)

# -----------------------------------------------------------------------------
#
def flatten_op(volumes, method = 'multiply linear',
               fitregion = None, subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    if method.startswith('m'):
        method = 'multiply linear'
    elif method.startswith('d'):
        method = 'divide linear'
    else:
        CommandError('Flatten method must be "multiplyLinear" or "divideLinear"')
    subregion = parse_subregion(subregion)
    fitregion = subregion if fitregion is None else parse_subregion(fitregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    from .flatten import flatten
    for v in volumes:
        flatten(v, method, step, subregion, fitregion, modelId)

# -----------------------------------------------------------------------------
#
def fourier_op(volumes, subregion = 'all', step = 1, modelId = None,
               phase = False):

    volumes = filter_volumes(volumes)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    from .fourier import fourier_transform
    for v in volumes:
        fourier_transform(v, step, subregion, modelId, phase)

# -----------------------------------------------------------------------------
#
def gaussian_op(volumes, session, sDev = (1.0,1.0,1.0),
                subregion = 'all', step = 1, valueType = None, modelId = None):

    from .gaussian import gaussian_convolve
    for v in volumes:
        gaussian_convolve(v, sDev, step, subregion, valueType, modelId, session = session)

# -----------------------------------------------------------------------------
#
def laplacian_op(volumes, subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)
    
    from .laplace import laplacian
    for v in volumes:
        laplacian(v, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def local_correlation_op(map1, map2, windowSize = 5, subtractMean = False, modelId = None):

    v1 = filter_volumes(map1)
    v2 = filter_volumes(map2)
    if len(v1) != 1 or len(v2) != 1:
        raise CommandError('vop localCorrelation operation requires '
                           'exactly two map arguments')
    v1, v2 = v1[0], v2[0]
    if not isinstance(windowSize,int) or windowSize < 2:
        raise CommandError('vop localCorrelation windowSize must be '
                           'an integer >= 2')
    if windowSize > min(v1.data.size):
        raise CommandError('vop localCorrelation windowSize must be '
                           'smaller than map size')
    modelId = parse_model_id(modelId)

    from .localcorr import local_correlation
    mapc = local_correlation(v1, v2, windowSize, subtractMean, modelId)
    return mapc

# -----------------------------------------------------------------------------
#
def median_op(volumes, binSize = 3, iterations = 1,
              subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    check_number(iterations, 'iterations', positive = True)
    binSize = parse_step(binSize, 'binSize', require_3_tuple = True)
    for b in binSize:
        if b <= 0 or b % 2 == 0:
            raise CommandError('Bin size must be positive odd integer, got %d' % b)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    from .median import median_filter
    for v in volumes:
        median_filter(v, binSize, iterations, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def morph_op(volumes, frames = 25, start = 0, playStep = 0.04,
             playDirection = 1, playRange = None, addMode = False,
             constantVolume = False, scaleFactors = None,
             hideOriginalMaps = True, interpolateColors = True,
             subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    check_number(frames, 'frames', int, nonnegative = True)
    check_number(start, 'start')
    check_number(playStep, 'playStep', nonnegative = True)
    if playRange is None:
        if addMode:
            prange = (-1.0,1.0)
        else:
            prange = (0.0,1.0)
    else:
        prange = parse_floats(playRange, 'playRange', 2)
    check_number(playDirection, 'playDirection')
    sfactors = parse_floats(scaleFactors, 'scaleFactors', len(volumes))
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)
    vs = [tuple(v.matrix_size(step = step, subregion = subregion))
          for v in volumes]
    if len(set(vs)) > 1:
        sizes = ' and '.join([str(s) for s in vs])
        raise CommandError("Volume grid sizes don't match: %s" % sizes)
    from MorphMap import morph_maps
    morph_maps(volumes, frames, start, playStep, playDirection, prange,
               addMode, constantVolume, sfactors,
               hideOriginalMaps, interpolateColors, subregion, step, modelId)
        
# -----------------------------------------------------------------------------
#
def octant_op(volumes, center = None, iCenter = None,
              subregion = 'all', step = 1, inPlace = False,
              fillValue = 0, modelId = None):

    volumes = filter_volumes(volumes)
    center = parse_floats(center, 'center', 3)
    iCenter = parse_floats(iCenter, 'iCenter', 3)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    check_in_place(inPlace, volumes)
    check_number(fillValue, 'fillValue')
    modelId = parse_model_id(modelId)
    outside = True
    for v in volumes:
        octant_operation(v, outside, center, iCenter, subregion, step,
                         inPlace, fillValue, modelId)

# -----------------------------------------------------------------------------
#
def octant_complement_op(volumes, center = None, iCenter = None,
              subregion = 'all', step = 1, inPlace = False,
              fillValue = 0, modelId = None):

    volumes = filter_volumes(volumes)
    center = parse_floats(center, 'center', 3)
    iCenter = parse_floats(iCenter, 'iCenter', 3)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    check_in_place(inPlace, volumes)
    check_number(fillValue, 'fillValue')
    modelId = parse_model_id(modelId)
    outside = False
    for v in volumes:
        octant_operation(v, outside, center, iCenter, subregion, step,
                         inPlace, fillValue, modelId)

# -----------------------------------------------------------------------------
#
def octant_operation(v, outside, center, iCenter,
                     subregion, step, inPlace, fillValue, modelId):

    vc = v.writable_copy(require_copy = not inPlace,
                         subregion = subregion, step = step,
                         model_id = modelId, show = False)
    ic = submatrix_center(v, center, iCenter, subregion, step)
    ijk_max = [i-1 for i in vc.data.size]
    from VolumeEraser import set_box_value
    set_box_value(vc.data, fillValue, ic, ijk_max, outside)
    vc.data.values_changed()
    vc.show()

# -----------------------------------------------------------------------------
#
def permute_axes_op(volumes, axisOrder = 'xyz',
                    subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    ao = {'xyz':(0,1,2), 'xzy':(0,2,1), 'yxz':(1,0,2), 
          'yzx':(1,2,0), 'zxy':(2,0,1), 'zyx':(2,1,0)}
    if not axisOrder in ao:
        raise CommandError('Axis order must be xyz, xzy, zxy, zyx, yxz, or yzx')
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    from .permute import permute_axes
    for v in volumes:
        permute_axes(v, ao[axisOrder], step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def resample_op(volumes, onGrid = None, boundingGrid = False,
                subregion = 'all', step = 1,
                gridSubregion = 'all', gridStep = 1,
                modelId = None, session = None):

    volumes = filter_volumes(volumes)
    if onGrid is None:
        raise CommandError('Resample operation must specify onGrid option')
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    gridSubregion = parse_subregion(gridSubregion, 'gridSubregion')
    gridStep = parse_step(gridStep, 'gridStep')
    onGrid = filter_volumes(onGrid, 'onGrid')
    modelId = parse_model_id(modelId)
    for v in volumes:
        for gv in onGrid:
            combine_operation([v], 'add', subregion, step,
                              gv, gridSubregion, gridStep,
                              boundingGrid, False, None, modelId, session)

# -----------------------------------------------------------------------------
#
def ridges_op(volumes, level = None, subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    check_number(level, 'level', allow_none = True)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    from .ridges import ridges
    for v in volumes:
        ridges(v, level, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def scale_op(volumes, shift = 0, factor = 1, sd = None, rms = None,
             valueType = None, type = None,
             subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    check_number(shift, 'shift')
    check_number(factor, 'factor')
    check_number(sd, 'sd', allow_none = True)
    check_number(rms, 'rms', allow_none = True)
    if not sd is None and not rms is None:
        raise CommandError('vop scale: Cannot specify both sd and rms options')
    value_type = parse_value_type(type if valueType is None else valueType)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    from .scale import scaled_volume
    for v in volumes:
        scaled_volume(v, factor, sd, rms, shift, value_type, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def subtract_op(vol1, vol2, onGrid = None, boundingGrid = False,
                subregion = 'all', step = 1,
                gridSubregion = 'all', gridStep = 1,
                inPlace = False, scaleFactors = None, minRMS = False,
                modelId = None, session = None):

    vol1 = filter_volumes(vol1)
    vol2 = filter_volumes(vol2)
    if len(vol1) != 1 or len(vol2) != 1:
        raise CommandError('vop subtract operation requires exactly two volumes')
    if minRMS and scaleFactors:
        raise CommandError('vop subtract cannot specify both minRMS and scaleFactors options.')
    mult = (1,'minrms') if minRMS else scaleFactors

    combine_op(vol1+vol2, 'subtract', onGrid, boundingGrid, subregion, step,
               gridSubregion, gridStep, inPlace, mult, modelId, session)

# -----------------------------------------------------------------------------
#
def threshold_op(volumes, minimum = None, set = None,
                 maximum = None, setMaximum = None,
                 subregion = 'all', step = 1, modelId = None, session = None):

    from .threshold import threshold
    for v in volumes:
        threshold(v, minimum, set, maximum, setMaximum,
                  step, subregion, modelId, session)

# -----------------------------------------------------------------------------
#
def tile_op(volumes, axis = 'z', pstep = 1, trim = 0,
            columns = None, rows = None, fillOrder = 'ulh',
            subregion = 'shown', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    if not axis in ('x', 'y', 'z'):
        raise CommandError('vop tile axis must be "x", "y", or "z"')
    if not isinstance(pstep,int) or pstep <= 0:
        raise CommandError('vop tile pstep must be positive integer')
    if not isinstance(trim,int):
        raise CommandError('vop tile trim must be an integer')
    if not columns is None and (not isinstance(columns,int) or columns <= 0):
        raise CommandError('vop tile columns must be positive integer')
    if not rows is None and (not isinstance(rows,int) or rows <= 0):
        raise CommandError('vop tile rows must be positive integer')
    orders = ('ulh', 'ulv', 'urh', 'urv', 'llh', 'llv', 'lrh', 'lrv',
              'ulhr', 'ulvr', 'urhr', 'urvr', 'llhr', 'llvr', 'lrhr', 'lrvr')
    if not fillOrder in orders:
        raise CommandError('vop tile fillOrder must be one of %s'
                           % ', '.join(orders))
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    from .tile import tile_planes
    for v in volumes:
        t = tile_planes(v, axis, pstep, trim, rows, columns, fillOrder,
                        step, subregion, modelId)
        if t is None:
            raise CommandError('vop tile: no planes')

# -----------------------------------------------------------------------------
#
def unbend_op(volumes, path, yaxis, xsize, ysize, gridSpacing = None,
              subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    if len(path) < 2:
        raise CommandError('vop unbend path must have 2 or more nodes')
    yaxis, axis_pt, csys = parse_axis(yaxis, 'vop unbend')
    if not (isinstance(xsize, (int,float)) and isinstance(ysize, (int,float))):
        raise CommandError('vop unbend xsize/ysize must be float values')
    modelId = parse_model_id(modelId)

    from . import unbend
    p = unbend.atom_path(path)
    for v in volumes:
        gs = min(v.data.step) if gridSpacing is None else gridSpacing
        xf = v.openState.xform if csys is None else csys.xform
        yax = xf.apply(yaxis).data()
        unbend.unbend_volume(v, p, yax, xsize, ysize, gs,
                             subregion, step, modelId)

# -----------------------------------------------------------------------------
#
def unroll_op(volumes, innerRadius = None, outerRadius = None, length = None,
              gridSpacing = None, axis = 'z', center = (0,0,0),
              coordinateSystem = None,
              subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    subregion = parse_subregion(subregion)
    step = parse_step(step, require_3_tuple = True)
    modelId = parse_model_id(modelId)

    for v in volumes:
        c, a, csys = parse_center_axis(center, axis, coordinateSystem,
                                       'vop unroll')
        if not csys is None:
            c, a = transform_center_axis(c, a, csys.xform, v.openState.xform)
        r0, r1, h = parse_cylinder_size(innerRadius, outerRadius, length,
                                        c, a, v, subregion, step)
        gsp = parse_grid_spacing(gridSpacing, v, step)
        from . import unroll
        unroll.unroll_operation(v, r0, r1, h, c, a, gsp,
                                subregion, step, modelId)
    
# -----------------------------------------------------------------------------
#
def transform_center_axis(c, a, from_xf, to_xf):

    from chimera import Point, Vector
    to_xf_inv = to_xf.inverse()
    tc = to_xf_inv.apply(from_xf.apply(Point(*tuple(c)))).data()
    ta = to_xf_inv.apply(from_xf.apply(Vector(*tuple(a)))).data()
    return tc, ta

# -----------------------------------------------------------------------------
#
def parse_cylinder_size(innerRadius, outerRadius, length, center, axis,
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
    if innerRadius is None or outerRadius is None:
        from . import unroll
        rmin, rmax = unroll.cylinder_radii(v, center, axis)
        pad = 0.10
        r0 = rmin * (1 - pad) if innerRadius is None else innerRadius
        r1 = rmax * (1 + pad) if outerRadius is None else outerRadius
    else:
        r0, r1 = innerRadius, outerRadius
    if not isinstance(r0, (float, int)):
        raise CommandError('innerRadius must be a number')
    if not isinstance(r1, (float, int)):
        raise CommandError('outerRadius must be a number')

    return r0, r1, h

# -----------------------------------------------------------------------------
#
def parse_grid_spacing(gridSpacing, v, step):

    if gridSpacing is None:
        gsz = min([s*st for s, st in zip(v.data.step, step)])
    elif isinstance(gridSpacing, (int, float)):
        gsz = gridSpacing
    else:
        raise CommandError('gridSpacing must be a number')
    return gsz

# -----------------------------------------------------------------------------
#
def flip_op(volumes, axis = 'z', subregion = 'all', step = 1,
              inPlace = False, modelId = None):

    volumes = filter_volumes(volumes)
    axes = ''.join(a for a in str(axis).lower() if a in ('x','y','z'))
    if len(axes) == 0:
        raise CommandError('flip axes must be one or more letters x, y, z, got "%s"' % axis)
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    check_in_place(inPlace, volumes[:1])
    modelId = parse_model_id(modelId)

    for v in volumes:
        flip_operation(v, axes, subregion, step, inPlace, modelId)
        
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
def zone_op(volumes, atoms, radius, bondPointSpacing = None,
            minimalBounds = False, invert = False,
            subregion = 'all', step = 1, modelId = None):

    volumes = filter_volumes(volumes)
    if len(atoms) == 0:
        raise CommandError('no atoms specified for zone')
    if not isinstance(radius, (float, int)):
        raise CommandError('radius value must be a number')
    subregion = parse_subregion(subregion)
    step = parse_step(step)
    modelId = parse_model_id(modelId)

    for v in volumes:
        zone_operation(v, atoms, radius, bondPointSpacing,
                       minimalBounds, invert, subregion, step, modelId)

# -----------------------------------------------------------------------------
#
def zone_operation(v, atoms, radius, bond_point_spacing = None,
                   minimal_bounds = False, invert = False,
                   subregion = 'all', step = 1, model_id = None):

    from ... import molecule as M
    bonds = M.interatom_bonds(atoms) if bond_point_spacing else []

    import SurfaceZone as SZ
    points = SZ.path_points(atoms, bonds, v.openState.xform.inverse(),
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
