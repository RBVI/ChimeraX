# vim: set expandtab ts=4 sw=4:

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
# Simulate an electron density map for an atomic model at a specfied
# resolution.  The simulated map is useful for fitting the model into
# an experimental map using correlation coefficient as a goodness of fit.
#
def register_molmap_command(logger):

    from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg, PositiveFloatArg
    from chimerax.core.commands import CenterArg, AxisArg, CoordSysArg
    from chimerax.atomic import SymmetryArg, AtomsArg
    from . import MapArg
    molmap_desc = CmdDesc(
        required = [
            ('atoms', AtomsArg),
            ('resolution', PositiveFloatArg),
        ],
        keyword = [
            ('grid_spacing', PositiveFloatArg),
            ('edge_padding', FloatArg),
            ('on_grid', MapArg),
            ('cutoff_range', FloatArg),
            ('sigma_factor', FloatArg),
            ('balls', BoolArg),
            ('symmetry', SymmetryArg),
            ('center', CenterArg),   # Can be a 3 floats or atom spec
            ('axis', AxisArg),     # Can be a 3 floats or atom spec
            ('coordinate_system', CoordSysArg),
            ('display_threshold', FloatArg),
#            ('modelId', model_id_arg),
            ('replace', BoolArg),
            ('show_dialog', BoolArg),
        ],
        synopsis = 'Compute a map by placing Gaussians at atom positions'
    )
    register('molmap', molmap_desc, molmap, logger=logger)

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
           center = None,         # Center of symmetry.
           axis = None,           # Axis of symmetry.
           coordinate_system = None,       # Coordinate system of symmetry.
           display_threshold = 0.95, # fraction of total density
           model_id = None, # integer
           replace = True,
           show_dialog = True,
           open_model = True,   # if calling directly from Python, may not want model opened
                                # implies show_dialog=False
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
    symmetry : Symmetry object or None.
      Apply symmetry operations to atoms
    center : Center object or None
      Center of symmetry.
    axis : Axis object or None
      Axis of symmetry.
    coordinate_system : Place
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
        from chimerax.core.errors import UserError
        raise UserError('No atoms specified')

    pad = 3*resolution if edge_padding is None else edge_padding
    step = (1./3) * resolution if grid_spacing is None else grid_spacing

    if symmetry is None:
        transforms = None
    else:
        transforms = symmetry.positions(center, axis, coordinate_system,
                                        atoms[0].structure)

    if not model_id is None:
        from chimerax.core.commands.parse import parse_model_id
        model_id = parse_model_id(model_id)

    if not open_model:
        show_dialog = False
    v = make_molecule_map(atoms, resolution, step, pad, on_grid,
                          cutoff_range, sigma_factor, balls, transforms,
                          display_threshold, model_id, replace, show_dialog, name, session,
                          open_model=open_model)

    return v

molecule_map = molmap

# -----------------------------------------------------------------------------
#
def make_molecule_map(atoms, resolution, step, pad, on_grid, cutoff_range,
                      sigma_factor, balls, transforms,
                      display_threshold, model_id,
                      replace, show_dialog, name, session, open_model = True):

    grid = molecule_grid_data(atoms, resolution, step, pad, on_grid,
                              cutoff_range, sigma_factor, balls,
                              transforms, name)

    if replace:
        from .volume import volume_list
        vlist = [v for v in volume_list(session)
                 if hasattr(v, 'molmap_atoms') and v.molmap_atoms == atoms]
        session.models.close(vlist)

    from . import volume_from_grid_data
    v = volume_from_grid_data(grid, session, style = 'surface', open_model = False,
                              show_dialog = show_dialog)
    levels, colors = v.initial_surface_levels(mfrac = (display_threshold,1))
    v.set_parameters(surface_levels = levels, surface_colors = colors)

    tf = on_grid.position if on_grid else atoms[0].structure.position
    v.position = tf

    v.molmap_atoms = atoms   # Remember atoms used to calculate volume
    v.molmap_parameters = (resolution, step, pad, cutoff_range, sigma_factor)

    if open_model:
        session.models.add([v])
    return v

# -----------------------------------------------------------------------------
#
def molecule_grid_data(atoms, resolution, step, pad, on_grid,
                       cutoff_range, sigma_factor, balls = False,
                       transforms = None, name = 'molmap'):

    if len(atoms.unique_structures) == 1 and not on_grid:
        xyz = atoms.coords
        tf = atoms[0].structure.position
    else:
        # Transform coordinates to local coordinates of the molecule containing
        # the first atom.  This handles multiple unaligned molecules.
        # Or if on_grid give transform to grid coordinates.
        xyz = atoms.scene_coords
        tf = on_grid.position if on_grid else atoms[0].structure.position
        tf.inverse().transform_points(xyz, in_place = True)

    if transforms:
        # Adjust transforms to correct coordinate system
        transforms = transforms.transform_coordinates(tf)

    if on_grid:
        from numpy import float32
        grid = on_grid.region_grid(on_grid.region, float32)
    else:
        grid = bounding_grid(xyz, step, pad, transforms)
    grid.name = name

    sdev = resolution * sigma_factor
    if balls:
        radii = atoms.radii
        add_balls(grid, xyz, radii, sdev, cutoff_range, transforms)
    else:
        weights = atoms.element_numbers
        add_gaussians(grid, xyz, weights, sdev, cutoff_range, transforms)

    return grid

# -----------------------------------------------------------------------------
#
def bounding_grid(xyz, step, pad, transforms = None):
    from chimerax.geometry import bounds
    b = bounds.point_bounds(xyz, transforms)
    origin = [x-pad for x in b.xyz_min]
    from math import ceil
    shape = [int(ceil((b.xyz_max[a] - b.xyz_min[a] + 2*pad) / step))
             for a in (2,1,0)]
    from numpy import zeros, float32
    matrix = zeros(shape, float32)
    from chimerax.map_data import ArrayGridData
    grid = ArrayGridData(matrix, origin, (step,step,step))
    return grid

# -----------------------------------------------------------------------------
#
def add_gaussians(grid, xyz, weights, sdev, cutoff_range, transforms = None,
                  normalize = True):

    from numpy import zeros, float32, empty
    sdevs = zeros((len(xyz),3), float32)
    for a in (0,1,2):
        sdevs[:,a] = sdev / grid.step[a]

    if transforms is None:
        from chimerax.geometry import Places
        transforms = Places()
    from ._map import sum_of_gaussians
    ijk = empty(xyz.shape, float32)
    matrix = grid.matrix()
    for tf in transforms:
        ijk[:] = xyz
        (grid.xyz_to_ijk_transform * tf).transform_points(ijk, in_place = True)
        sum_of_gaussians(ijk, weights, sdevs, cutoff_range, matrix)

    if normalize:
        from math import pow, pi
        normalization = pow(2*pi,-1.5)*pow(sdev,-3)
        matrix *= normalization

# -----------------------------------------------------------------------------
#
def add_balls(grid, xyz, radii, sdev, cutoff_range, transforms = None):

    if transforms is None or len(transforms) == 0:
        from chimerax.geometry import Places
        transforms = Places()
    from numpy import empty, float32
    ijk = empty(xyz.shape, float32)
    r = (radii - sdev) / grid.step[0]
    matrix = grid.matrix()
    from ._map import sum_of_balls
    for tf in transforms:
        ijk[:] = xyz
        (grid.xyz_to_ijk_transform * tf).transform_points(ijk, in_place = True)
        sum_of_balls(ijk, r, sdev, cutoff_range, matrix)
