#
# Test of ambient occlusion using gaussian map from molecule.
#
def ambient_occlusion_coloring(atoms, fineness = 0.3, light = 0.8, dark = 0.1):

    from time import time
    t0 = time()

    points = atoms.coordinates()

    # Compute density map for atoms
    from .. import molecule_cpp
    xyz_min, xyz_max = molecule_cpp.point_bounds(points)
    size3 = xyz_max - xyz_min
    size = size3.max()
    resolution = fineness * size
    step = 0.5*resolution
    pad = resolution

    t1 = time()

    # from math import pi, sqrt
    # from ..map.molmap import molecule_grid_data
    # grid, molecules = molecule_grid_data(atoms, resolution, step, pad,
    #                                      cutoff_range = 3,
    #                                      sigma_factor = 1/(pi*sqrt(2)))
    # m = grid.full_matrix()
    # tf = grid.xyz_to_ijk_transform

    origin = xyz_min - (pad,pad,pad)
    step3 = (step, step, step)
    from numpy import ceil, zeros, float32
    xsz,ysz,zsz = [int(i) for i in ceil((size3 + (2*pad,2*pad,2*pad))/step)]
    m = zeros((zsz,ysz,xsz), float32)
    from .. import map_cpp
    map_cpp.fill_occupancy_map(points, origin, step3, m)
    from ..geometry import place
    tf = place.Place(((1/step, 0, 0, -origin[0]/step),
                      (0, 1/step, 0, -origin[1]/step),
                      (0, 0, 1/step, -origin[2]/step)))

    t2 = time()

    # Interpolate map to find color scale factors
    from ..map.data import interpolate_volume_data
    values, outside = interpolate_volume_data(points, tf, m)

    t3 = time()

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

    t4 = time()

    # Scale colors
    atoms.scale_atom_colors(scale)

    t5 = time()
    print('aoc time %.3f (coords %.3f, map %.3f, interp %.3f, ramp %.3f, color %.3f), %d atoms, grid %s, range %g,%g, '
          % (t5-t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4,
             atoms.count(), ','.join('%d'%s for s in m.shape[::-1]), vmin, vmax))

def ambient_occlusion_command(cmdname, args, session):

  from ..ui.commands import atoms_arg, float_arg, parse_arguments
  req_args = (('atoms', atoms_arg),)
  opt_args = ()
  kw_args = (('fineness', float_arg),
             ('light', float_arg),
             ('dark', float_arg),)
  kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
  ambient_occlusion_coloring(**kw)
