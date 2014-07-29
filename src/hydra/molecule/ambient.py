#
# Test of ambient occlusion using gaussian map from molecule.
#
def ambient_occlusion_coloring(model, fineness = None, light = None, dark = 0.1, bin_size = 7,
                               texture = False, gaussian = False):

    atoms = model.atom_set()
    latom = 0.8 if light is None else light
    if texture:
        ambient_occlusion_color_atoms_3d_texture(atoms, fineness, latom, dark, gaussian)
    else:
        ambient_occlusion_color_atoms(atoms, fineness, latom, dark, gaussian)

    satoms = model.atom_set(include_surface_atoms = True)
    satoms.remove_duplicates()
    ambient_occlusion_color_molecular_surfaces(model.molecular_surfaces(), satoms, fineness, latom, dark)

    lmap = 0.5 if light is None else light
    for m in model.maps():
        ambient_occlusion_color_map(m, bin_size, lmap, dark)

def ambient_occlusion_color_atoms(atoms, fineness = None, light = 0.8, dark = 0.1, gaussian = False):

    if atoms.count() == 0:
        return

    from time import time
    t0 = time()
    points = atoms.coordinates()
    t1 = time()

    m, tf = ambient_atom_density(points, fineness, light, dark)
    t2 = time()

    # Interpolate map to find color scale factors
    from ..map.data import interpolate_volume_data
    values, outside = interpolate_volume_data(points, tf, m)

    t3 = time()
    scale = darkness_ramp(values, dark, light)
    t4 = time()

    # Scale colors
    atoms.scale_atom_colors(scale)

    t5 = time()
    print('aoc time %.3f (coords %.3f, map %.3f, interp %.3f, ramp %.3f, color %.3f), %d atoms, grid %s'
          % (t5-t0, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4,
             atoms.count(), ','.join('%d'%s for s in m.shape[::-1])))

def ambient_occlusion_color_atoms_3d_texture(atoms, fineness = None, light = 0.8, dark = 0.1, gaussian = False):

    if atoms.count() == 0:
        return

    points = atoms.coordinates()
    m, tf = ambient_atom_density(points, fineness, light, dark, gaussian)

    # Rescale matrix.
    r0, r1 = m.min(), m.max()
    m -= r1 - dark*(r1-r0)
    m *= 1 / ((light-dark)*(r0-r1))
    if dark > 0:
        from numpy import maximum
        maximum(m, 0, m)
    if light < 1:
        from numpy import minimum
        minimum(m, 1, m)

    from ..graphics import Texture
    t = Texture(m, dimension = 3)
    m0 = atoms.molecules()[0]
    d = m0.atoms_drawing
    d.ambient_texture = t
    from ..geometry.place import scale
    d.ambient_texture_transform = scale(tuple(1/(s-1) for s in m.shape[::-1])) * tf

def ambient_atom_density(points, fineness = None, light = 0.8, dark = 0.1, gaussian = False, weights = None):

    # Set default fineness
    if fineness is None:
        fineness = 0.3
        if len(points) > 70000:
            # For large atom counts try to emphasize protein boundaries.
            from math import sqrt
            fineness /= sqrt(len(points)/70000)

    # Compute density map for atoms
    from .. import molecule_cpp
    xyz_min, xyz_max = molecule_cpp.point_bounds(points)
    size3 = xyz_max - xyz_min
    size = size3.max()
    if size == 0:
        size = 1
    resolution = fineness * size
    pad = resolution

    if gaussian:
        step = 0.2*resolution
        if weights is None:
            from numpy import ones, float32
            weights = ones((len(points),), float32)
        from math import pi, sqrt
        from ..map.molmap import gaussian_grid_data
        grid = gaussian_grid_data(points, weights, resolution, step, pad,
                                  cutoff_range = 3,
                                  sigma_factor = 1/(pi*sqrt(2)))
        m = grid.full_matrix()
        tf = grid.xyz_to_ijk_transform
    else:
        # Binning
        origin = xyz_min - (pad,pad,pad)
        step = 0.5*resolution
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

    return m, tf

def ambient_occlusion_color_molecular_surfaces(surfs, atoms, fineness = None, light = 0.8, dark = 0.1):

    if len(surfs) == 0 or atoms.count() == 0:
        return
    points = atoms.coordinates()
    m, tf = ambient_atom_density(points, fineness, light, dark)
    from ..map.data import interpolate_volume_data
    for surf in surfs:
        values, outside = interpolate_volume_data(surf.vertices, tf, m)
        scale = darkness_ramp(values, dark, light)
        scale_vertex_colors(surf, scale)

def darkness_ramp(values, dark, light):
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
    return scale

def scale_vertex_colors(surf, scale):
    colors = surf.vertex_colors
    if colors is None:
        from numpy import empty, uint8
        colors = empty((len(surf.vertices),4), uint8)
        colors[:] = surf.get_color()
    else:
        colors = colors.copy()  # Need this so Drawing knows to update opengl buffers.
    for c in (0,1,2):
        colors[:,c] *= scale
    surf.vertex_colors = colors

def ambient_occlusion_color_map(v, bin_size = 7, light = 0.5, dark = 0.1):
    s = bin_size
    from ..map.filter import bin
    g = bin.bin_grid(v, (s,s,s))
    from ..geometry import place
    from ..map.data import interpolate_volume_data
    for p in v.surface_drawings:
        bscale = place.scale(1.0/bin_size)
        b2 = -0.5*(bin_size-1)
        bshift = place.translation((b2,b2,b2))
        tf = bscale * bshift * v.data.xyz_to_ijk_transform
        values, outside = interpolate_volume_data(p.vertices, tf, g.full_matrix())
        # TODO: outside 0 values should not effect ramp.
        scale = darkness_ramp(values, dark, light)
        scale_vertex_colors(p, scale)

def ambient_occlusion_command(cmdname, args, session):

  from ..ui.commands import specifier_arg, float_arg, int_arg, bool_arg, parse_arguments
  req_args = (('model', specifier_arg),)
  opt_args = ()
  kw_args = (('fineness', float_arg),
             ('light', float_arg),
             ('dark', float_arg),
             ('bin_size', int_arg),
             ('texture', bool_arg),
             ('gaussian', bool_arg),
  )
  kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
  ambient_occlusion_coloring(**kw)
