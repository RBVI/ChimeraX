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
from ...cli import UserError as CommandError

def register_vop_command():

    from ...cli import CmdDesc, register, BoolArg, EnumOf, IntArg, Int3Arg
    from ...cli import FloatArg, Float3Arg, FloatsArg, ModelIdArg
    from ..mapargs import MapsArg, MapStepArg, MapRegionArg, Float1or3Arg, ValueTypeArg
    from ..mapargs import BoxArg, Float2Arg
    from ...structure import AtomsArg

    varg = [('volumes', MapsArg)]
    ssm_kw = [
        ('subregion', MapRegionArg),
        ('step', MapStepArg),
        ('modelId', ModelIdArg),
    ]
    resample_kw = [
        ('onGrid', MapsArg),
        ('boundingGrid', BoolArg),
        ('gridSubregion', MapRegionArg),
        ('gridStep', MapStepArg),
    ] + ssm_kw
    add_kw = resample_kw + [
        ('inPlace', BoolArg),
        ('scaleFactors', FloatsArg),
    ]
    add_desc = CmdDesc(required = varg, keyword = add_kw)
    register('vop add', add_desc, add_op)

    bin_desc = CmdDesc(required = varg,
                       keyword = [('binSize', MapStepArg)] + ssm_kw
    )
    register('vop bin', bin_desc, bin_op)

    boxes_desc = CmdDesc(required = varg + [('markers', AtomsArg)],
                         keyword = [('size', FloatArg),
                                    ('useMarkerSize', BoolArg)] + ssm_kw)
    register('vop boxes', boxes_desc, boxes_op)

    cover_desc = CmdDesc(required = varg,
        keyword = [('atomBox', AtomsArg),
                   ('pad', FloatArg),
                   ('box', BoxArg), ('x', Float2Arg), ('y', Float2Arg), ('z', Float2Arg),
                   ('fbox', BoxArg), ('fx', Float2Arg), ('fy', Float2Arg), ('fz', Float2Arg),
                   ('ibox', BoxArg), ('ix', Float2Arg), ('iy', Float2Arg), ('iz', Float2Arg),
                   ('useSymmetry', BoolArg),
                   ('cellSize', Int3Arg),
                   ('step', MapStepArg),
                   ('modelId', ModelIdArg)]
    )
    register('vop cover', cover_desc, cover_op)

    falloff_desc = CmdDesc(required = varg,
        keyword = [('iterations', IntArg), ('inPlace', BoolArg)] + ssm_kw
    )
    register('vop falloff', falloff_desc, falloff_op)

    flatten_desc = CmdDesc(required = varg,
                           keyword = [('method', EnumOf(('multiplyLinear', 'divideLinear'))),
                                      ('fitregion', MapRegionArg)] + ssm_kw)
    register('vop flatten', flatten_desc, flatten_op)

    flip_desc = CmdDesc(required = varg,
                        keyword = [('axis', EnumOf(('x','y','z','xy', 'yz','xyz'))),
                                   ('inPlace', BoolArg)] + ssm_kw)
    register('vop flip', flip_desc, flip_op)

    fourier_desc = CmdDesc(required = varg,
                           keyword = [('phase', BoolArg)] + ssm_kw)
    register('vop fourier', fourier_desc, fourier_op)

    gaussian_desc = CmdDesc(required = varg,
        keyword = [('sDev', Float1or3Arg), ('valueType', ValueTypeArg)] + ssm_kw
    )
    register('vop gaussian', gaussian_desc, gaussian_op)

    laplacian_desc = CmdDesc(required = varg, keyword = ssm_kw)
    register('vop laplacian', laplacian_desc, laplacian_op)

    localcorr_desc = CmdDesc(required = varg,
                             keyword = [('windowSize', IntArg),
                                        ('subtractMean', BoolArg),
                                        ('modelId', ModelIdArg)])
    register('vop localCorrelation', localcorr_desc, local_correlation_op)

    maximum_desc = CmdDesc(required = varg, keyword = add_kw)
    register('vop maximum', maximum_desc, maximum_op)

    median_desc = CmdDesc(required = varg,
                          keyword = [('binSize', MapStepArg),
                                     ('iterations', IntArg)] + ssm_kw)
    register('vop median', median_desc, median_op)

    morph_desc = CmdDesc(required = varg,
                         keyword = [('frames', IntArg),
                                    ('start', FloatArg),
                                    ('playStep', FloatArg),
                                    ('playDirection', IntArg),
                                    ('playRange', Float2Arg),
                                    ('addMode', BoolArg),
                                    ('constantVolume', BoolArg),
                                    ('scaleFactors', FloatsArg),
                                    ('hideOriginalMaps', BoolArg),
                                    ('interpolateColors', BoolArg)] + ssm_kw)
    register('vop morph', morph_desc, morph_op)

    multiply_desc = CmdDesc(required = varg, keyword = add_kw)
    register('vop multiply', multiply_desc, multiply_op)

    oct_kw = [('center', Float3Arg),
              ('iCenter', Int3Arg),
              ('fillValue', FloatArg),
              ('inPlace', BoolArg)]
    octant_desc = CmdDesc(required = varg,
                          keyword = oct_kw + ssm_kw)
    register('vop octant', octant_desc, octant_op)

    unoctant_desc = CmdDesc(required = varg,
                            keyword = oct_kw + ssm_kw)
    register('vop ~octant', unoctant_desc, octant_complement_op)

    permuteaxes_desc = CmdDesc(required = varg,
                               keyword = [('axisOrder', EnumOf(('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')))] + ssm_kw)
    register('vop permuteaxes', permuteaxes_desc, permute_axes_op)

    resample_desc = CmdDesc(required = varg, keyword = resample_kw)
    register('vop resample', resample_desc, resample_op)

    ridges_desc = CmdDesc(required = varg,
                          keyword = [('level', FloatArg)] + ssm_kw)
    register('vop ridges', ridges_desc, ridges_op)

    scale_desc = CmdDesc(required = varg,
                         keyword = [('shift', FloatArg),
                                    ('factor', FloatArg),
                                    ('sd', FloatArg),
                                    ('rms', FloatArg),
                                    ('valueType', ValueTypeArg),
                                    ('type', ValueTypeArg)] + ssm_kw)
    register('vop scale', scale_desc, scale_op)

    subtract_desc = CmdDesc(required = varg,
                            keyword = add_kw + [('minRMS', BoolArg)])
    register('vop subtract', subtract_desc, subtract_op)

    threshold_desc = CmdDesc(required = varg,
        keyword = [('minimum', FloatArg), ('set', FloatArg),
                   ('maximum', FloatArg), ('setMaximum', FloatArg)] + ssm_kw
    )
    register('vop threshold', threshold_desc, threshold_op)

    orders = ('ulh', 'ulv', 'urh', 'urv', 'llh', 'llv', 'lrh', 'lrv',
              'ulhr', 'ulvr', 'urhr', 'urvr', 'llhr', 'llvr', 'lrhr', 'lrvr')
    tile_desc = CmdDesc(required = varg,
                        keyword = [('axis', EnumOf(('x','y','z'))),
                                   ('pstep', IntArg),
                                   ('trim', IntArg),
                                   ('columns', IntArg),
                                   ('rows', IntArg),
                                   ('fillOrder', EnumOf(orders))] + ssm_kw)
    register('vop tile', tile_desc, tile_op)

    unbend_desc = CmdDesc(required = varg + [('path', AtomsArg),
                                             ('yaxis', Float3Arg),
                                             ('xsize', FloatArg),
                                             ('ysize', FloatArg)],
                          keyword = [('gridSpacing', FloatArg)] + ssm_kw)
    register('vop unbend', unbend_desc, unbend_op)

    unroll_desc = CmdDesc(required = varg,
                          keyword = [('innerRadius', FloatArg),
                                     ('outerRadius', FloatArg),
                                     ('length', FloatArg),
                                     ('gridSpacing', FloatArg),
                                     ('axis', Float3Arg),
                                     ('center', Float3Arg)] + ssm_kw)
    register('vop unroll', unroll_desc, unroll_op)

    zflip_desc = CmdDesc(required = varg,
                         keyword = [('axis', EnumOf(('x','y','z','xy', 'yz','xyz'))),
                                    ('inPlace', BoolArg)] + ssm_kw)
    register('vop zFlip', zflip_desc, flip_op)

    zone_desc = CmdDesc(required = varg + [('atoms', AtomsArg), ('radius', FloatArg)],
                        keyword = [('bondPointSpacing', FloatArg),
                                   ('minimalBounds', BoolArg),
                                   ('invert', BoolArg)] + ssm_kw)
    register('vop zone', zone_desc, zone_op)

# -----------------------------------------------------------------------------
#
def add_op(session, volumes, onGrid = None, boundingGrid = None,
           subregion = 'all', step = 1,
           gridSubregion = 'all', gridStep = 1,
           inPlace = False, scaleFactors = None, modelId = None):

    combine_op(volumes, 'add', onGrid, boundingGrid, subregion, step,
               gridSubregion, gridStep, inPlace, scaleFactors, modelId, session)

# -----------------------------------------------------------------------------
#
def maximum_op(session, volumes, onGrid = None, boundingGrid = None,
               subregion = 'all', step = 1,
               gridSubregion = 'all', gridStep = 1,
               inPlace = False, scaleFactors = None, modelId = None):

    combine_op(volumes, 'maximum', onGrid, boundingGrid, subregion, step,
               gridSubregion, gridStep, inPlace, scaleFactors, modelId, session)

# -----------------------------------------------------------------------------
#
def multiply_op(session, volumes, onGrid = None, boundingGrid = None,
                subregion = 'all', step = 1,
                gridSubregion = 'all', gridStep = 1,
                inPlace = False, scaleFactors = None, modelId = None):

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
def bin_op(session, volumes, subregion = 'all', step = 1,
           binSize = (2,2,2), modelId = None):

    from .bin import bin
    for v in volumes:
        bin(v, binSize, step, subregion, modelId, session)

# -----------------------------------------------------------------------------
#
def boxes_op(session, volumes, markers, size = 0, useMarkerSize = False,
             subregion = 'all', step = 1, modelId = None):

    if size <= 0 and not useMarkerSize:
        from ... import cli
        raise CommandError('Must specify size or enable useMarkerSize')

    from .boxes import boxes
    for v in volumes:
        boxes(v, markers, size, useMarkerSize, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def cover_op(session, volumes, atomBox = None, pad = 5.0, 
             box = None, x = None, y = None, z = None,
             fBox = None, fx = None, fy = None, fz = None,
             iBox = None, ix = None, iy = None, iz = None,
             useSymmetry = True, cellSize = None,
             step = 1, modelId = None):

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
        return box
    box = ([None,None,None], [None,None,None])
    for a,x,name in zip((0,1,2),(x,y,z),(xname,yname,zname)):
        if x:
            box[0][a], box[1][a] = parse_floats(x, name, 2)
    return box

# -----------------------------------------------------------------------------
#
def falloff_op(session, volumes, iterations = 10, inPlace = False,
               subregion = 'all', step = 1, modelId = None):

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
def flatten_op(session, volumes, method = 'multiplyLinear',
               fitregion = None, subregion = 'all', step = 1, modelId = None):

    if fitregion is None:
        fitregion = subregion

    from .flatten import flatten
    for v in volumes:
        flatten(v, method, step, subregion, fitregion, modelId)

# -----------------------------------------------------------------------------
#
def fourier_op(session, volumes, subregion = 'all', step = 1, modelId = None, phase = False):

    from .fourier import fourier_transform
    for v in volumes:
        fourier_transform(v, step, subregion, modelId, phase)

# -----------------------------------------------------------------------------
#
def gaussian_op(session, volumes, sDev = (1.0,1.0,1.0),
                subregion = 'all', step = 1, valueType = None, modelId = None):

    from .gaussian import gaussian_convolve
    for v in volumes:
        gaussian_convolve(v, sDev, step, subregion, valueType, modelId, session = session)

# -----------------------------------------------------------------------------
#
def laplacian_op(session, volumes, subregion = 'all', step = 1, modelId = None):

    from .laplace import laplacian
    for v in volumes:
        laplacian(v, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def local_correlation_op(session, volumes, windowSize = 5, subtractMean = False, modelId = None):

    if len(volumes) != 2:
        raise CommandError('vop localCorrelation operation requires '
                           'exactly two map arguments')
    v1, v2 = volumes
    if windowSize < 2:
        raise CommandError('vop localCorrelation windowSize must be '
                           'an integer >= 2')
    if windowSize > min(v1.data.size):
        raise CommandError('vop localCorrelation windowSize must be '
                           'smaller than map size')

    from .localcorr import local_correlation
    mapc = local_correlation(v1, v2, windowSize, subtractMean, modelId)
    return mapc

# -----------------------------------------------------------------------------
#
def median_op(session, volumes, binSize = 3, iterations = 1,
              subregion = 'all', step = 1, modelId = None):

    for b in binSize:
        if b <= 0 or b % 2 == 0:
            raise CommandError('Bin size must be positive odd integer, got %d' % b)

    from .median import median_filter
    for v in volumes:
        median_filter(v, binSize, iterations, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def morph_op(session, volumes, frames = 25, start = 0, playStep = 0.04,
             playDirection = 1, playRange = None, addMode = False,
             constantVolume = False, scaleFactors = None,
             hideOriginalMaps = True, interpolateColors = True,
             subregion = 'all', step = 1, modelId = None):

    if playRange is None:
        if addMode:
            prange = (-1.0,1.0)
        else:
            prange = (0.0,1.0)
    else:
        prange = playRange

    if not scaleFactors is None and len(volumes) != len(scaleFactors):
        raise CommandError('Number of scale factors (%d) doesn not match number of volumes (%d)'
                           % (len(scaleFactors), len(volumes)))
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
def octant_op(session, volumes, center = None, iCenter = None,
              subregion = 'all', step = 1, inPlace = False,
              fillValue = 0, modelId = None):

    check_in_place(inPlace, volumes)
    outside = True
    for v in volumes:
        octant_operation(v, outside, center, iCenter, subregion, step,
                         inPlace, fillValue, modelId)

# -----------------------------------------------------------------------------
#
def octant_complement_op(session, volumes, center = None, iCenter = None,
                         subregion = 'all', step = 1, inPlace = False,
                         fillValue = 0, modelId = None):

    check_in_place(inPlace, volumes)
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
def permute_axes_op(session, volumes, axisOrder = 'xyz',
                    subregion = 'all', step = 1, modelId = None):

    ao = {'xyz':(0,1,2), 'xzy':(0,2,1), 'yxz':(1,0,2), 
          'yzx':(1,2,0), 'zxy':(2,0,1), 'zyx':(2,1,0)}
    from .permute import permute_axes
    for v in volumes:
        permute_axes(v, ao[axisOrder], step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def resample_op(session, volumes, onGrid = None, boundingGrid = False,
                subregion = 'all', step = 1,
                gridSubregion = 'all', gridStep = 1,
                modelId = None):

    if onGrid is None:
        raise CommandError('Resample operation must specify onGrid option')
    for v in volumes:
        for gv in onGrid:
            combine_operation([v], 'add', subregion, step,
                              gv, gridSubregion, gridStep,
                              boundingGrid, False, None, modelId, session)

# -----------------------------------------------------------------------------
#
def ridges_op(session, volumes, level = None, subregion = 'all', step = 1, modelId = None):

    from .ridges import ridges
    for v in volumes:
        ridges(v, level, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def scale_op(session, volumes, shift = 0, factor = 1, sd = None, rms = None,
             valueType = None, type = None,
             subregion = 'all', step = 1, modelId = None):

    if not sd is None and not rms is None:
        raise CommandError('vop scale: Cannot specify both sd and rms options')
    value_type = type if valueType is None else valueType

    from .scale import scaled_volume
    for v in volumes:
        scaled_volume(v, factor, sd, rms, shift, value_type, step, subregion, modelId)

# -----------------------------------------------------------------------------
#
def subtract_op(session, volumes, onGrid = None, boundingGrid = False,
                subregion = 'all', step = 1,
                gridSubregion = 'all', gridStep = 1,
                inPlace = False, scaleFactors = None, minRMS = False,
                modelId = None):

    if len(volumes) != 2:
        from ... import cli
        raise CommandError('vop subtract operation requires exactly two volumes')
    if minRMS and scaleFactors:
        from ... import cli
        raise CommandError('vop subtract cannot specify both minRMS and scaleFactors options.')
    mult = (1,'minrms') if minRMS else scaleFactors

    combine_op(volumes, 'subtract', onGrid, boundingGrid, subregion, step,
               gridSubregion, gridStep, inPlace, mult, modelId, session)

# -----------------------------------------------------------------------------
#
def threshold_op(session, volumes, minimum = None, set = None,
                 maximum = None, setMaximum = None,
                 subregion = 'all', step = 1, modelId = None):

    from .threshold import threshold
    for v in volumes:
        threshold(v, minimum, set, maximum, setMaximum,
                  step, subregion, modelId, session)

# -----------------------------------------------------------------------------
#
def tile_op(session, volumes, axis = 'z', pstep = 1, trim = 0,
            columns = None, rows = None, fillOrder = 'ulh',
            subregion = 'shown', step = 1, modelId = None):

    from .tile import tile_planes
    for v in volumes:
        t = tile_planes(v, axis, pstep, trim, rows, columns, fillOrder,
                        step, subregion, modelId)
        if t is None:
            raise CommandError('vop tile: no planes')

# -----------------------------------------------------------------------------
#
def unbend_op(session, volumes, path, yaxis, xsize, ysize, gridSpacing = None,
              subregion = 'all', step = 1, modelId = None):

    if len(path) < 2:
        raise CommandError('vop unbend path must have 2 or more nodes')

    from . import unbend
    p = unbend.atom_path(path)
    for v in volumes:
        gs = min(v.data.step) if gridSpacing is None else gridSpacing
        xf = v.openState.xform
        yax = xf.apply(yaxis).data()
        unbend.unbend_volume(v, p, yax, xsize, ysize, gs,
                             subregion, step, modelId)

# -----------------------------------------------------------------------------
#
def unroll_op(session, volumes, innerRadius = None, outerRadius = None, length = None,
              gridSpacing = None, axis = (0,0,1), center = (0,0,0),
              subregion = 'all', step = 1, modelId = None):


    for v in volumes:
        r0, r1, h = parse_cylinder_size(innerRadius, outerRadius, length,
                                        center, axis, v, subregion, step)
        gsp = parse_grid_spacing(gridSpacing, v, step)
        from . import unroll
        unroll.unroll_operation(v, r0, r1, h, center, axis, gsp,
                                subregion, step, modelId)

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
def flip_op(session, volumes, axis = 'z', subregion = 'all', step = 1,
            inPlace = False, modelId = None):

    for v in volumes:
        flip_operation(v, axis, subregion, step, inPlace, modelId)
        
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
def zone_op(session, volumes, atoms, radius, bondPointSpacing = None,
            minimalBounds = False, invert = False,
            subregion = 'all', step = 1, modelId = None):

    if len(atoms) == 0:
        raise CommandError('no atoms specified for zone')

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
