# vim: set expandtab ts=4 sw=4:
# -----------------------------------------------------------------------------
# Simulate an electron density map for an atomic model at a specfied
# resolution.  The simulated map is useful for fitting the model into
# an experimental map using correlation coefficient as a goodness of fit.
#
def register_molmap_command():

    from ..commands import CmdDesc, register, AtomsArg, BoolArg, FloatArg
    from ..map import MapArg
    molmap_desc = CmdDesc(
        required = [
            ('atoms', AtomsArg),
            ('resolution', FloatArg),
        ],
        keyword = [
            ('grid_spacing', FloatArg),
            ('edge_padding', FloatArg),
            ('on_grid', MapArg),
            ('cutoff_range', FloatArg),
            ('sigma_factor', FloatArg),
            ('balls', BoolArg),
#            ('symmetry', StringArg),
#            ('center', StringArg),   # Can be a 3 floats or atom spec
#            ('axis', StringArg),     # Can be a 3 floats or atom spec
#            ('coordinate_system', openstate_arg),
            ('display_threshold', FloatArg),
#            ('modelId', model_id_arg),
            ('replace', BoolArg),
            ('show_dialog', BoolArg),
        ]
    )
    register('molmap', molmap_desc, molmap)

# -----------------------------------------------------------------------------
#
from math import sqrt, pi
def molmap(session,
           atoms,
           resolution,
           grid_spacing = None,    # default is 1/3 resolution
           edge_padding = None,    # default is 3 times resolution
           on_grid = None,	  # use this grid instead of bounding grid
           cutoff_range = 5,       # in standard deviations
           sigma_factor = 1/(pi*sqrt(2)), # standard deviation / resolution
           balls = False,         # Use balls instead of Gaussians
           symmetry = None,       # Equivalent to sym group option.
           center = (0,0,0),      # Center of symmetry.
           axis = (0,0,1),        # Axis of symmetry.
           coordinate_system = None,       # Coordinate system of symmetry.
           display_threshold = 0.95, # fraction of total density
           model_id = None, # integer
           replace = True,
           show_dialog = True
          ):
    '''
    Create a density map by placing Gaussians centered on atoms.

    Parameters
    ----------
    atoms : Atoms
    resolution : float
    grid_spacing : float
      Default is 1/3 resolution.
    edge_padding : float
      Default is 3 times resolution.
    cutoff_range : float
      In standard deviations.
    sigma_factor : float
      Scale factor equal to standard deviation / resolution, default 1/(pi*sqrt(2)).
    balls : bool
      Use balls instead of Gaussians
    symmetry : not supported
      Apply symmetry operations to atoms
    center : 3 floats or atom spec
      Center of symmetry.
    axis : 3 floats
      Axis of symmetry.
    coordinate_system : model spec
      Coordinate system of symmetry.
    display_threshold : float
      Initial contour level as fraction of total density, default 0.95.
    model_id : list of integers
    replace : bool
      Default true
    show_dialog : bool, not supported
    '''

    molecules = atoms.unique_structures
    if len(molecules) > 1:
        name = 'map %.3g' % (resolution,)
    else:
        for m in molecules:
            name = '%s map %.3g' % (m.name, resolution)

    if len(atoms) == 0:
        from ..errors import UserError
        raise UserError('No atoms specified')

    pad = 3*resolution if edge_padding is None else edge_padding
    step = (1./3) * resolution if grid_spacing is None else grid_spacing

    csys = None
    if symmetry is None:
        transforms = []
    else:
        from ..commands.parse import openstate_arg
        if coordinate_system:
            csys = openstate_arg(coordinate_system)
        from .SymmetryCopies.symcmd import parse_symmetry
        transforms, csys = parse_symmetry(symmetry, center, axis, csys,
                                          atoms[0].structure, 'molmap')

    if not model_id is None:
        from ..commands.parse import parse_model_id
        model_id = parse_model_id(model_id)

    v = make_molecule_map(atoms, resolution, step, pad, on_grid,
                          cutoff_range, sigma_factor, balls, transforms, csys,
                          display_threshold, model_id, replace, show_dialog, name, session)

    return v

molecule_map = molmap

# -----------------------------------------------------------------------------
#
def make_molecule_map(atoms, resolution, step, pad, on_grid, cutoff_range,
                      sigma_factor, balls, transforms, csys,
                      display_threshold, model_id,
                      replace, show_dialog, name, session):

    grid = molecule_grid_data(atoms, resolution, step, pad, on_grid,
                              cutoff_range, sigma_factor, balls,
                              transforms, csys, name)

    if replace:
        from .volume import volume_list
        vlist = [v for v in volume_list(session)
                 if hasattr(v, 'molmap_atoms') and v.molmap_atoms == atoms]
        session.models.close(vlist)

    from . import volume_from_grid_data
    v = volume_from_grid_data(grid, session, open_model = False,
                              show_dialog = show_dialog)
    v.initialize_thresholds(mfrac = (display_threshold, 1), replace = True)
    tf = on_grid.position if on_grid else atoms[0].structure.position
    v.position = tf
    v.show()

    v.molmap_atoms = atoms   # Remember atoms used to calculate volume
    v.molmap_parameters = (resolution, step, pad, cutoff_range, sigma_factor)

    session.models.add([v])
    return v

# -----------------------------------------------------------------------------
#
def molecule_grid_data(atoms, resolution, step, pad, on_grid,
                       cutoff_range, sigma_factor, balls = False,
                       transforms = [], csys = None, name = 'molmap'):

    if len(atoms.unique_structures) == 1 and not on_grid:
        xyz = atoms.coords
    else:
        # Transform coordinates to local coordinates of the molecule containing
        # the first atom.  This handles multiple unaligned molecules.
        # Or if on_grid give transform to grid coordinates.
        xyz = atoms.scene_coords
        tf = on_grid.position if on_grid else atoms[0].structure.position
        tf.inverse().move(xyz)

# TODO: Adjust transforms to correct coordinate system
#    if csys:
#        tf = csys.inverse() * tf
#    tfinv = tf.inverse()
#    tflist = [(tfinv * t * tf) for t in transforms]

    tflist = transforms

    if on_grid:
        from numpy import float32
        grid = on_grid.region_grid(on_grid.region, float32)
    else:
        grid = bounding_grid(xyz, step, pad, tflist)
    grid.name = name

    sdev = resolution * sigma_factor
    if balls:
        radii = atoms.radii
        add_balls(grid, xyz, radii, sdev, cutoff_range, tflist)
    else:
        weights = atoms.element_numbers
        add_gaussians(grid, xyz, weights, sdev, cutoff_range, tflist)

    return grid

# -----------------------------------------------------------------------------
#
def bounding_grid(xyz, step, pad, transforms):
    from ..geometry import bounds
    b = bounds.point_bounds(xyz, transforms)
    origin = [x-pad for x in b.xyz_min]
    from math import ceil
    shape = [int(ceil((b.xyz_max[a] - b.xyz_min[a] + 2*pad) / step))
             for a in (2,1,0)]
    from numpy import zeros, float32
    matrix = zeros(shape, float32)
    from .data import Array_Grid_Data
    grid = Array_Grid_Data(matrix, origin, (step,step,step))
    return grid

# -----------------------------------------------------------------------------
#
def add_gaussians(grid, xyz, weights, sdev, cutoff_range, transforms = []):

    from numpy import zeros, float32, empty
    sdevs = zeros((len(xyz),3), float32)
    for a in (0,1,2):
        sdevs[:,a] = sdev / grid.step[a]

    if len(transforms) == 0:
        from ..geometry import identity
        transforms = [identity()]
    from ._map import sum_of_gaussians
    ijk = empty(xyz.shape, float32)
    matrix = grid.matrix()
    for tf in transforms:
        ijk[:] = xyz
        (grid.xyz_to_ijk_transform * tf).move(ijk)
        sum_of_gaussians(ijk, weights, sdevs, cutoff_range, matrix)
    
    from math import pow, pi
    normalization = pow(2*pi,-1.5)*pow(sdev,-3)
    matrix *= normalization

# -----------------------------------------------------------------------------
#
def add_balls(grid, xyz, radii, sdev, cutoff_range, transforms = []):

    if len(transforms) == 0:
        from ..geometry import identity
        transforms = [identity()]
    from numpy import empty, float32
    ijk = empty(xyz.shape, float32)
    r = (radii - sdev) / grid.step[0]
    matrix = grid.matrix()
    from ._map import sum_of_balls
    for tf in transforms:
        ijk[:] = xyz
        (grid.xyz_to_ijk_transform * tf).move(ijk)
        sum_of_balls(ijk, r, sdev, cutoff_range, matrix)
