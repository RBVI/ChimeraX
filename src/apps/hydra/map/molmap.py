# -----------------------------------------------------------------------------
# Simulate an electron density map for an atomic model at a specfied
# resolution.  The simulated map is useful for fitting the model into
# an experimental map using correlation coefficient as a goodness of fit.
#
def molmap_command(cmdname, args, session):

    from ..commands.parse import atoms_arg, float_arg, string_arg, openstate_arg
    from ..commands.parse import model_id_arg, bool_arg, parse_arguments
    req_args = (('atoms', atoms_arg),
                ('resolution', float_arg))
    opt_args = ()
    kw_args = (('gridSpacing', float_arg),
               ('edgePadding', float_arg),
               ('cutoffRange', float_arg),
               ('sigmaFactor', float_arg),
               ('balls', bool_arg),
               ('symmetry', string_arg),
               ('center', string_arg),
               ('axis', string_arg),
               ('coordinateSystem', openstate_arg),
               ('displayThreshold', float_arg),
               ('modelId', model_id_arg),
               ('replace', bool_arg),
               ('showDialog', bool_arg))

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    molecule_map(**kw)

# -----------------------------------------------------------------------------
#
from math import sqrt, pi
def molecule_map(atoms, resolution, session,
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

    from ..commands.parse import CommandError
    if atoms.count() == 0:
        raise CommandError('No atoms specified')

    for vname in ('resolution', 'gridSpacing', 'edgePadding',
                  'cutoffRange', 'sigmaFactor'):
        value = locals()[vname]
        if not isinstance(value, (float,int,type(None))):
            raise CommandError('%s must be number, got "%s"' % (vname,str(value)))

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

    v = make_molecule_map(atoms, resolution, step, pad,
                          cutoffRange, sigmaFactor, balls, transforms, csys,
                          displayThreshold, modelId, replace, showDialog, session)
    return v

# -----------------------------------------------------------------------------
#
def make_molecule_map(atoms, resolution, step, pad, cutoff_range,
                      sigma_factor, balls, transforms, csys,
                      display_threshold, model_id,
                      replace, show_dialog, session):

    grid, molecules = molecule_grid_data(atoms, resolution, step, pad,
                                         cutoff_range, sigma_factor, balls,
                                         transforms, csys)

    if replace:
        from . import volume_list
        vlist = [v for v in volume_list(session)
                 if getattr(v, 'molmap_atoms', None) == atoms]
        session.close_models(vlist)

    from . import volume_from_grid_data
    v = volume_from_grid_data(grid, session, open_model = False,
                              show_dialog = show_dialog)
    v.initialize_thresholds(mfrac = (display_threshold, 1), replace = True)
    v.show()

    v.molmap_atoms = atoms   # Remember atoms used to calculate volume
    v.molmap_parameters = (resolution, step, pad, cutoff_range, sigma_factor)

    session.add_model(v)
    return v

# -----------------------------------------------------------------------------
#
def molecule_grid_data(atoms, resolution, step, pad,
                       cutoff_range, sigma_factor, balls = False,
                       transforms = [], csys = None):

    xyz = atoms.coordinates()

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

    molecules = atoms.molecules()
    if len(molecules) > 1:
        name = 'map %.3g' % (resolution,)
    else:
        name = '%s map %.3g' % (molecules[0].name, resolution)

    if balls:
        radii = atoms.radii()
        grid = balls_grid_data(xyz, radii, resolution, step, pad,
                               cutoff_range, sigma_factor, tflist)
    else:
        from math import pow, pi
        normalization = pow(2*pi,-1.5)*pow(sigma_factor*resolution,-3)
        weights = atoms.element_numbers()*normalization
        grid = gaussian_grid_data(xyz, weights, resolution, step, pad,
                                  cutoff_range, sigma_factor, tflist)
    grid.name = name

    return grid, molecules

# -----------------------------------------------------------------------------
#
def gaussian_grid_data(xyz, weights, resolution, step, pad,
                       cutoff_range, sigma_factor, transforms = []):

    from ..geometry import bounds
    xyz_min, xyz_max = bounds.point_bounds(xyz, transforms)

    origin = [x-pad for x in xyz_min]
    sdev = sigma_factor * resolution / step
    from numpy import zeros, float32, empty
    sdevs = zeros((len(xyz),3), float32)
    sdevs[:] = sdev
    from math import ceil
    shape = [int(ceil((xyz_max[a] - xyz_min[a] + 2*pad) / step))
             for a in (2,1,0)]
    matrix = zeros(shape, float32)

    from ..geometry.place import Place, identity
    xyz_to_ijk_tf = Place(((1.0/step, 0, 0, -origin[0]/step),
                           (0, 1.0/step, 0, -origin[1]/step),
                           (0, 0, 1.0/step, -origin[2]/step)))
    if len(transforms) == 0:
        transforms = [identity()]
    from ..map_cpp import sum_of_gaussians
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
    xyz_min, xyz_max = bounds.point_bounds(xyz, transforms)

    origin = [x-pad for x in xyz_min]
    sdev = sigma_factor * resolution / step
    from numpy import zeros, float32, empty
    from math import ceil
    shape = [int(ceil((xyz_max[a] - xyz_min[a] + 2*pad) / step))
             for a in (2,1,0)]
    matrix = zeros(shape, float32)

    from ..geometry.place import Place, identity
    xyz_to_ijk_tf = Place(((1.0/step, 0, 0, -origin[0]/step),
                           (0, 1.0/step, 0, -origin[1]/step),
                           (0, 0, 1.0/step, -origin[2]/step)))
    if len(transforms) == 0:
        transforms = [identity()]
    from ..map_cpp import sum_of_balls
    ijk = empty(xyz.shape, float32)
    r = (radii / step) - sdev
    for tf in transforms:
        ijk[:] = xyz
        (xyz_to_ijk_tf * tf).move(ijk)
        sum_of_balls(ijk, r, sdev, cutoff_range, matrix)
    
    from .data import Array_Grid_Data
    grid = Array_Grid_Data(matrix, origin, (step,step,step))

    return grid
