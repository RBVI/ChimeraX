# -----------------------------------------------------------------------------
# Simulate an electron density map for an atomic model at a specfied
# resolution.  The simulated map is useful for fitting the model into
# an experimental map using correlation coefficient as a goodness of fit.
#
def register_molmap_command():

    from .. import cli, atomspec
    molmap_desc = cli.CmdDesc(
        required = [
            ('atoms', atomspec.AtomSpecArg),
            ('resolution', cli.FloatArg),
        ],
        keyword = [
            ('gridSpacing', cli.FloatArg),
            ('edgePadding', cli.FloatArg),
            ('cutoffRange', cli.FloatArg),
            ('sigmaFactor', cli.FloatArg),
            ('balls', cli.BoolArg),
#            ('symmetry', cli.StringArg),
#            ('center', cli.StringArg),   # Can be a 3 floats or atom spec
#            ('axis', cli.StringArg),     # Can be a 3 floats or atom spec
#            ('coordinateSystem', openstate_arg),
            ('displayThreshold', cli.FloatArg),
#            ('modelId', model_id_arg),
            ('replace', cli.BoolArg),
            ('showDialog', cli.BoolArg),
        ]
    )
    cli.register('molmap', molmap_desc, molecule_map)

# -----------------------------------------------------------------------------
#
from math import sqrt, pi
def molecule_map(session,
                 atoms,
                 resolution,
                 gridSpacing = None,    # default is 1/3 resolution
                 edgePadding = None,    # default is 3 times resolution
                 cutoffRange = 5,       # in standard deviations
                 sigmaFactor = 1/(pi*sqrt(2)), # standard deviation / resolution
                 balls = False,         # Use balls instead of Gaussians
                 symmetry = None,       # Equivalent to sym group option.
                 center = (0,0,0),      # Center of symmetry.
                 axis = (0,0,1),        # Axis of symmetry.
                 coordinateSystem = None,       # Coordinate system of symmetry.
                 displayThreshold = 0.95, # fraction of total density
                 modelId = None, # integer
                 replace = True,
		 showDialog = True
                 ):

    asr = atoms.evaluate(session)
    a = asr.atoms

    molecules = asr.models
    if len(molecules) > 1:
        name = 'map %.3g' % (resolution,)
    else:
        for m in molecules:
            name = '%s map %.3g' % (m.name, resolution)

    if len(a) == 0:
        raise cli.UserError('No atoms specified')

    if edgePadding is None:
        pad = 3*resolution
    else:
        pad = edgePadding

    if gridSpacing is None:
        step = (1./3) * resolution
    else:
        step = gridSpacing

    csys = None
    if symmetry is None:
        transforms = []
    else:
        from ..commands.parse import openstate_arg
        if coordinateSystem:
            csys = openstate_arg(coordinateSystem)
        from .SymmetryCopies.symcmd import parse_symmetry
        transforms, csys = parse_symmetry(symmetry, center, axis, csys,
                                          atoms[0].molecule, 'molmap')

    if not modelId is None:
        from ..commands.parse import parse_model_id
        modelId = parse_model_id(modelId)

    v = make_molecule_map(a, resolution, step, pad,
                          cutoffRange, sigmaFactor, balls, transforms, csys,
                          displayThreshold, modelId, replace, showDialog, name, session)
    return v

# -----------------------------------------------------------------------------
#
def make_molecule_map(atoms, resolution, step, pad, cutoff_range,
                      sigma_factor, balls, transforms, csys,
                      display_threshold, model_id,
                      replace, show_dialog, name, session):

    grid = molecule_grid_data(atoms, resolution, step, pad,
                              cutoff_range, sigma_factor, balls,
                              transforms, csys, name)

    if replace:
        from .volume import volume_list
        vlist = [v for v in volume_list(session)
                 if getattr(v, 'molmap_atoms', None) == atoms]
        session.models.close(vlist)

    from . import volume_from_grid_data
    v = volume_from_grid_data(grid, session, open_model = False,
                              show_dialog = show_dialog)
    v.initialize_thresholds(mfrac = (display_threshold, 1), replace = True)
    v.show()

    v.molmap_atoms = atoms   # Remember atoms used to calculate volume
    v.molmap_parameters = (resolution, step, pad, cutoff_range, sigma_factor)

    session.models.add([v])
    return v

# -----------------------------------------------------------------------------
#
def molecule_grid_data(atoms, resolution, step, pad,
                       cutoff_range, sigma_factor, balls = False,
                       transforms = [], csys = None, name = 'molmap'):

    xyz = atoms.coords

    # Transform coordinates to local coordinates of the molecule containing
    # the first atom.  This handles multiple unaligned molecules.
#    m0 = atoms[0].molecule
#    tf = m0.position
#    tf.inverse().move(xyz)
#    if csys:
#        tf = csys.inverse() * tf
#    tfinv = tf.inverse()
#    tflist = [(tfinv * t * tf) for t in transforms]
    tflist = transforms

    if balls:
        radii = atoms.radii
        grid = balls_grid_data(xyz, radii, resolution, step, pad,
                               cutoff_range, sigma_factor, tflist)
    else:
        from math import pow, pi
        normalization = pow(2*pi,-1.5)*pow(sigma_factor*resolution,-3)
        weights = atoms.element_numbers*normalization
        grid = gaussian_grid_data(xyz, weights, resolution, step, pad,
                                  cutoff_range, sigma_factor, tflist)
    grid.name = name

    return grid

# -----------------------------------------------------------------------------
#
def gaussian_grid_data(xyz, weights, resolution, step, pad,
                       cutoff_range, sigma_factor, transforms = []):

    from ..geometry import bounds
    b = bounds.point_bounds(xyz, transforms)

    origin = [x-pad for x in b.xyz_min]
    sdev = sigma_factor * resolution / step
    from numpy import zeros, float32, empty
    sdevs = zeros((len(xyz),3), float32)
    sdevs[:] = sdev
    from math import ceil
    shape = [int(ceil((b.xyz_max[a] - b.xyz_min[a] + 2*pad) / step))
             for a in (2,1,0)]
    matrix = zeros(shape, float32)

    from ..geometry.place import Place, identity
    xyz_to_ijk_tf = Place(((1.0/step, 0, 0, -origin[0]/step),
                           (0, 1.0/step, 0, -origin[1]/step),
                           (0, 0, 1.0/step, -origin[2]/step)))
    if len(transforms) == 0:
        transforms = [identity()]
    from ._map import sum_of_gaussians
    ijk = empty(xyz.shape, float32)
    for tf in transforms:
        ijk[:] = xyz
        (xyz_to_ijk_tf * tf).move(ijk)
        sum_of_gaussians(ijk, weights, sdevs, cutoff_range, matrix)
    
    from .data import Array_Grid_Data
    grid = Array_Grid_Data(matrix, origin, (step,step,step))

    return grid

# -----------------------------------------------------------------------------
#
def balls_grid_data(xyz, radii, resolution, step, pad,
                       cutoff_range, sigma_factor, transforms = []):

    from ..geometry import bounds
    b = bounds.point_bounds(xyz, transforms)

    origin = [x-pad for x in b.xyz_min]
    sdev = sigma_factor * resolution / step
    from numpy import zeros, float32, empty
    from math import ceil
    shape = [int(ceil((b.xyz_max[a] - b.xyz_min[a] + 2*pad) / step))
             for a in (2,1,0)]
    matrix = zeros(shape, float32)

    from ..geometry.place import Place, identity
    xyz_to_ijk_tf = Place(((1.0/step, 0, 0, -origin[0]/step),
                           (0, 1.0/step, 0, -origin[1]/step),
                           (0, 0, 1.0/step, -origin[2]/step)))
    if len(transforms) == 0:
        transforms = [identity()]
    from ._map import sum_of_balls
    ijk = empty(xyz.shape, float32)
    r = (radii / step) - sdev
    for tf in transforms:
        ijk[:] = xyz
        (xyz_to_ijk_tf * tf).move(ijk)
        sum_of_balls(ijk, r, sdev, cutoff_range, matrix)
    
    from .data import Array_Grid_Data
    grid = Array_Grid_Data(matrix, origin, (step,step,step))

    return grid
