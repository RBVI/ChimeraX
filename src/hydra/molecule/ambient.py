#
# Test of ambient occlusion using gaussian map from molecule.
#
def ambient_occlusion_coloring(atoms, fineness = 0.3, light = 0.8, dark = 0.1):
    from time import time
    t0 = time()
    points = atoms.coordinates()
    xyz_min, xyz_max = points.min(axis=0), points.max(axis=0)
    size = max(xyz_max - xyz_min)
    resolution = fineness * size
    step = 0.5*resolution
    pad = 3*resolution
    cutoff_range = 3
    from math import pi, sqrt
    sigma_factor = 1/(pi*sqrt(2))
    from ..map.molmap import molecule_grid_data
    grid, molecules = molecule_grid_data(atoms, resolution, step, pad,
                                         cutoff_range, sigma_factor)
    ptransform = grid.xyz_to_ijk_transform
    from ..map.data import interpolate_volume_data
    values, outside = interpolate_volume_data(points, ptransform, grid.full_matrix())
    vmin, vmax = values.min(), values.max()
    lev0, lev1 = dark,light
    v0, v1 = vmax - lev0*(vmax-vmin), vmin + (1-lev1)*(vmax-vmin)
    scale = (values-v0)/(v1-v0)
    if lev1 < 1:
        from numpy import minimum
        minimum(scale, 1.0, scale)
    if lev0 > 0:
        from numpy import maximum
        maximum(scale, 0, scale)
    atoms.scale_atom_colors(scale)
    t1 = time()
    print('aoc time %.2f, %d atoms, grid %s' % (t1-t0, atoms.count(), ','.join('%d'%s for s in grid.size)))

def ambient_occlusion_command(cmdname, args, session):

  from ..ui.commands import atoms_arg, float_arg, parse_arguments
  req_args = (('atoms', atoms_arg),)
  opt_args = ()
  kw_args = (('fineness', float_arg),
             ('light', float_arg),
             ('dark', float_arg),)
  kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
  ambient_occlusion_coloring(**kw)
