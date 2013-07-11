# -----------------------------------------------------------------------------
# Optimize N random placements of a model and collect unique fits.
#
# Points should be in volume local coordinates.
# Models can contain atomic models and maps that are the source of the points.
#
def fit_search(models, points, point_weights, volume, n,
               rotations = True, shifts = True, radius = None,
               angle_tolerance = 6, shift_tolerance = 3,
               asymmetric_unit = True,
               minimum_points_in_contour = 0.5,
               metric = 'sum product',
               optimize_translation = True, optimize_rotation = True,
               max_steps = 2000,
               ijk_step_size_min = 0.01, ijk_step_size_max = 0.5,
               request_stop_cb = None):

    b = volume.bounds()
    xyz_min, xyz_max = b if not b is None else volume.xyz_bounds(step = 1)
    asym_center_f = (.75,.55,.55)
    asym_center = tuple(x0 + (x1-x0)*f
                        for x0, x1, f in zip(xyz_min, xyz_max, asym_center_f)) 

    from .. import matrix as M
    center = points.mean(axis=0)
    ctf = M.translation_matrix([-x for x in center])
    ijk_to_xyz_tf = volume.matrix_indices_to_xyz_transform(step = 1)
    xyz_to_ijk_tf = M.invert_matrix(ijk_to_xyz_tf)
    data_array = volume.matrix(step = 1)
    vtfinv = M.invert_matrix(volume.place)
    mtv_list = [M.multiply_matrices(vtfinv, m.place) for m in models]

    flist = []
    outside = 0
    from math import pi
    from .. import bins
    b = bins.Binned_Transforms(angle_tolerance*pi/180, shift_tolerance, center)
    fo = {}
    from .fitmap import locate_maximum
    for i in range(n):
        if request_stop_cb and request_stop_cb('Fit %d of %d' % (i+1,n)):
            break
        shift = ((random_translation(xyz_min, xyz_max) if radius is None
                  else random_translation_step(center, radius)) if shifts
                  else M.translation_matrix(center))
        rot = random_rotation() if rotations else M.identity_matrix()
        tf = M.multiply_matrices(shift, rot, ctf)
        optimize = True
        max_opt = 2
        while optimize and max_opt > 0:
            p_to_ijk_tf = M.multiply_matrices(xyz_to_ijk_tf, tf)
            move_tf, stats = \
              locate_maximum(points, point_weights, data_array, p_to_ijk_tf,
                             max_steps, ijk_step_size_min, ijk_step_size_max,
                             optimize_translation, optimize_rotation,
                             metric, request_stop_cb = None)
            ptf = M.multiply_matrices(tf, move_tf)
            optimize = False
            max_opt -= 1
            if asymmetric_unit:
                atf = unique_symmetry_position(ptf, center, asym_center,
                                               volume.data.symmetries)
                if not atf is ptf:
                    ptf = tf = atf
                    optimize = not b.close_transforms(ptf)
        close = b.close_transforms(ptf)
        if len(close) == 0:
            transforms = [M.multiply_matrices(ptf, mtv) for mtv in mtv_list]
            stats['hits'] = 1
            f = Fit(models, transforms, volume, stats)
            f.ptf = ptf
            flist.append(f)
            b.add_transform(ptf)
            fo[id(ptf)] = f
        else:
            s = fo[id(close[0])].stats
            s['hits'] += 1

    # Filter out solutions with too many points outside volume contour.
    fflist = [f for f in flist if (in_contour(f.ptf, points, volume, f.stats)
                                   >= minimum_points_in_contour)]
    fset = set(fflist)
    outside = sum([f.stats['hits'] for f in flist if not f in fset])
    
    # Sort fits by correlation, then average map value.
    fflist.sort(key = fit_order)
    fflist.reverse()    # Best first

    return fflist, outside

# -----------------------------------------------------------------------------
#
def in_contour(tf, points, volume, stats):

    from .. import matrix as M
    vtf = M.multiply_matrices(volume.place, tf)
    from . import fitmap as FM
    poc, clevel = FM.points_outside_contour(points, vtf, volume)
    stats['atoms outside contour'] = poc
    stats['contour level'] = clevel
    pic = float(len(points) - poc) / len(points)
    return pic

# -----------------------------------------------------------------------------
#
class Fit:

    def __init__(self, models, transforms, volume, stats):

        # Atomic models and/or maps fit into volume by a single rigid motion.
        self.models = models

        # Transform list contains a transform mtf for each model
        # mapping model local coordinates to volume local coordinates
        # to achieve the alignment. Alignment is achieved by
        #   model.openState.xform = volume.openState.xform * mtf
        # If transforms is None then derive the transforms from the current
        # model positions assuming the fit motion has been done.
        if transforms is None:
            from ..matrix import invert_matrix, multiply_matrices
            vtfinv = invert_matrix(volume.place)
            transforms = [multiply_matrices(vtfinv, m.place) for m in models]
        self.transforms = transforms

        self.volume = volume               # Volume being fit into.
        self.stats = stats

    def correlation(self):
        return self.stats.get('correlation', None)
    def average_map_value(self):
        return self.stats.get('average map value', None)
    def points_inside_contour(self):
        o = self.stats.get('atoms outside contour', None)
        if o is None:
            return None
        n = self.stats['points']
        f = float(n - o) / n
        return f
    def hits(self):
        return self.stats.get('hits', 1)  # Number of times this fit was found.

    # -------------------------------------------------------------------------
    #
    def place_models(self, frames = 0):

        if frames > 0:
            move_models(self.models, self.transforms, self.volume, frames)
        else:
            for m, tf in self.model_transforms():
                m.set_place(tf)

    # -------------------------------------------------------------------------
    #
    def model_transforms(self):

        mtf_list = []
        v = self.volume
        if v.__destroyed__:
            return mtf_list
        for m, tf in zip(self.models, self.transforms):
            if m is None or m.__destroyed__:
                continue
            from .. import matrix as M
            vtf = M.multiply_matrices(v.place, tf)
            mtf_list.append((m, vtf))
        return mtf_list

    # -------------------------------------------------------------------------
    #
    def place_copies(self):

        from chimera import Molecule, openModels as om
        from Molecule import copy_molecule

        mxf_list = [(m,xf) for m,xf in self.model_xforms()
                    if isinstance(m, Molecule)]
        copies = [copy_molecule(m) for m, xf in mxf_list]
        if copies:
            om.add(copies)

        for c, (m,xf) in zip(copies, mxf_list):
            c.openState.xform = xf

        return copies

    # -------------------------------------------------------------------------
    #
    def clash(self):

        if 'clash' in self.stats:
            return self.stats['clash']

        v = self.volume
        if v is None or v.__destroyed__:
            return None
        
        # Check if volume has symmetries.
        symlist = self.volume.data.symmetries
        if len(symlist) <= 1:
            return None

        # Look for exactly one map that was fit into volume.
        from VolumeViewer import Volume
        vtf = [(m,tf) for m, tf in zip(self.models, self.transforms)
               if m and not m.__destroyed__ and isinstance(m, Volume)]
        if len(vtf) != 1:
            return None
        m, tf = vtf[0]

        # Find grid points in base volume local coordinates.
        matrix, xyz_to_ijk_tf = m.matrix_and_transform(None, subregion = None,
                                                       step = None)
        threshold = min(m.surface_levels)
        import _volume
        points_int = _volume.high_indices(matrix, threshold)
        from numpy import float32
        points = points_int.astype(float32)
        import Matrix as M
        M.transform_points(points, M.multiply_matrices(tf, M.invert_matrix(xyz_to_ijk_tf)))

        # Transform points by volume symmetries and count how many are inside
        # contour of map m.
        tfinv = M.invert_matrix(tf)
        inside = 0
        for s in symlist:
            if not M.is_identity_matrix(s):
                p = points.copy()
                M.transform_points(p, M.multiply_matrices(tfinv, s))
                v = m.interpolated_values(p)
                inside += (v >= threshold).sum()

        self.stats['clash'] = f = float(inside) / len(points)
        
        return f

    # -------------------------------------------------------------------------
    #
    def fit_message(self):

        from . import fitmap as FM
        v = self.volume
        mmap = self.fit_map()
        if mmap:
            message = (FM.map_fit_message(mmap, v, self.stats) +
                       FM.transformation_matrix_message(mmap, v))
        else:
            mols = self.fit_molecules()
            message = FM.atom_fit_message(mols, v, self.stats)
            message += '\n'.join([FM.transformation_matrix_message(m,v)
                                  for m in mols])

        return message

    # -------------------------------------------------------------------------
    #
    def fit_map(self):

        from ..VolumeViewer import Volume
        vols = [m for m in self.models if isinstance(m, Volume)]
        mmap = vols[0] if vols else None
        return mmap

    # -------------------------------------------------------------------------
    #
    def fit_molecules(self):

        from ..molecule import Molecule
        mols = [m for m in self.models if isinstance(m, Molecule)]
        return mols
    
# -----------------------------------------------------------------------------
#
def fit_order(f):

    return (f.correlation(), f.average_map_value())

# -----------------------------------------------------------------------------
#
move_table = {}
def move_models(models, transforms, base_model, frames):

    global move_table
    add = (len(move_table) == 0)
    if base_model.__destroyed__:
        return
    for m, tf in zip(models, transforms):
        if m and not m.__destroyed__:
            move_table[m] = [tf, base_model, frames]
    if move_table and add:
        cb = []
        def mstep(mt = move_table, cb = cb):
            move_step(mt, cb)
        cb.append(mstep)
        from ..gui import main_window
        main_window.view.add_new_frame_callback(mstep)

# -----------------------------------------------------------------------------
#
def move_step(move_table, cb):

    mt = move_table
    from .. import matrix as M
    for m, (rxf, base_model, frames) in tuple(mt.items()):
        if m.__destroyed__ or base_model.__destroyed__:
            del mt[m]
            continue
        tf = M.multiply_matrices(base_model.place, rxf)
        b = m.bounds()
        if b:
            c = M.linear_combination(.5, b[0], .5, b[1])
            m.set_place(M.interpolate_transforms(m.place, c, tf, 1.0/frames))
            if frames <= 1:
                del mt[m]
            else:
                mt[m][2] = frames-1
        else:
            m.set_place(tf)
            del mt[m]

    if len(mt) == 0:
        from ..gui import main_window
        main_window.view.remove_new_frame_callback(cb[0])

# -----------------------------------------------------------------------------
#
def random_translation_step(center, radius):

    v = random_direction()
    from .. import matrix as M
    from random import random
    r = radius * random()
    tf = M.translation_matrix(M.linear_combination(1, center, r, v))
    return tf

# -----------------------------------------------------------------------------
#
def random_translation(xyz_min, xyz_max):

    from random import random
    shift = [x0+random()*(x1-x0) for x0,x1 in zip(xyz_min, xyz_max)]
    from .. import matrix as M
    tf = M.translation_matrix(shift)
    return tf

# -----------------------------------------------------------------------------
#
def random_rotation():

    y, z = random_direction(), random_direction()
    from .. import matrix as M
    f = M.orthonormal_frame(z, y)
    return ((f[0][0], f[1][0], f[2][0], 0),
            (f[0][1], f[1][1], f[2][1], 0),
            (f[0][2], f[1][2], f[2][2], 0))

# -----------------------------------------------------------------------------
#
def random_direction():

    z = (1,1,1)
    from ..matrix import norm, normalize_vector
    from random import random
    while norm(z) > 1:
        z = (1-2*random(), 1-2*random(), 1-2*random())
    return normalize_vector(z)

# -----------------------------------------------------------------------------
#
def any_close_tf(tf, tflist, angle_tolerance, shift_tolerance, shift_point):

    from .. import matrix as M
    for i, t in enumerate(tflist):
        if M.same_transform(t, tf, angle_tolerance, shift_tolerance, shift_point):
            return i
    return None

# -----------------------------------------------------------------------------
#
def unique_symmetry_position(tf, center, ref_point, sym_list):

    if len(sym_list) == 0:
        return tf

    import Matrix as M
    import numpy as n
    i = n.argmin([M.distance(M.apply_matrix(M.multiply_matrices(sym,tf), center), ref_point) for sym in sym_list])
    if M.is_identity_matrix(sym_list[i]):
        return tf
    return M.multiply_matrices(sym_list[i],tf)
