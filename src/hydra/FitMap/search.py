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

    have_bbox, bbox = volume.bbox()
    xyz_min, xyz_max = ((bbox.llf.data(), bbox.urb.data()) if have_bbox
                        else volume.xyz_bounds(step = 1))
    asym_center_f = (.75,.55,.55)
    asym_center = tuple(x0 + (x1-x0)*f
                        for x0, x1, f in zip(xyz_min, xyz_max, asym_center_f)) 

    import Matrix as M
    center = points.mean(axis=0)
    ctf = M.translation_matrix([-x for x in center])
    ijk_to_xyz_tf = volume.matrix_indices_to_xyz_transform(step = 1)
    xyz_to_ijk_tf = M.invert_matrix(ijk_to_xyz_tf)
    data_array = volume.matrix(step = 1)
    vtfinv = M.xform_matrix(volume.openState.xform.inverse())
    mtv_list = [M.multiply_matrices(vtfinv, M.xform_matrix(m.openState.xform))
                for m in models]

    flist = []
    outside = 0
    from math import pi
    from CrystalContacts import bins
    b = bins.Binned_Transforms(angle_tolerance*pi/180, shift_tolerance, center)
    fo = {}
    from fitmap import locate_maximum
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
    fflist.sort(order_fits)

    return fflist, outside

# -----------------------------------------------------------------------------
#
def in_contour(tf, points, volume, stats):

    import Matrix as M
    xf = M.chimera_xform(tf)
    xf.premultiply(volume.openState.xform)
    import fitmap as FM
    poc, clevel = FM.points_outside_contour(points, xf, volume)
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
            for m, xf in self.model_xforms():
                m.openState.xform = xf

    # -------------------------------------------------------------------------
    #
    def model_xforms(self):

        mxf_list = []
        v = self.volume
        if v.__destroyed__:
            return mxf_list
        for m, tf in zip(self.models, self.transforms):
            if m is None or m.__destroyed__:
                continue
            xf = v.openState.xform
            import Matrix
            xf.multiply(Matrix.chimera_xform(tf))
            mxf_list.append((m, xf))
        return mxf_list

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

        import fitmap as FM
        v = self.volume
        mmap = self.fit_map()
        if mmap:
            message = (FM.map_fit_message(mmap, v, self.stats) +
                       FM.transformation_matrix_message(mmap, v))
        else:
            mols = self.fit_molecules()
            atoms = sum([m.atoms for m in mols], [])
            message = FM.atom_fit_message(atoms, v, self.stats)
            message += '\n'.join([FM.transformation_matrix_message(m,v)
                                  for m in mols])

        return message

    # -------------------------------------------------------------------------
    #
    def fit_map(self):

        from VolumeViewer import Volume
        vols = [m for m in self.models if isinstance(m, Volume)]
        mmap = vols[0] if vols else None
        return mmap

    # -------------------------------------------------------------------------
    #
    def fit_molecules(self):

        from chimera import Molecule
        mols = [m for m in self.models if isinstance(m, Molecule)]
        return mols
    
# -----------------------------------------------------------------------------
#
def order_fits(fa, fb):

    return cmp((fb.correlation(), fb.average_map_value()),
               (fa.correlation(), fa.average_map_value()))

# -----------------------------------------------------------------------------
#
move_table = {}
def move_models(models, transforms, base_model, frames):

    global move_table
    add = (len(move_table) == 0)
    if base_model.__destroyed__:
        return
    bos = base_model.openState
    from Matrix import chimera_xform
    for m, tf in zip(models, transforms):
        if m and not m.__destroyed__:
            os = m.openState
            rxf = chimera_xform(tf)
            move_table[os] = [rxf, bos, frames]
    if move_table and add:
        mth = [move_table]
        from chimera import triggers
        mth.append(triggers.addHandler('new frame', move_step, mth))

# -----------------------------------------------------------------------------
#
def move_step(tname, mth, tdata):

    mt, h = mth
    for os, (rxf, bos, frames) in mt.items():
        if os.__destroyed__ or bos.__destroyed__:
            del mt[os]
            continue
        xf = bos.xform
        xf.multiply(rxf)
        have_box, box = os.bbox()
        if have_box:
            c = box.center()
            import Matrix
            os.xform = Matrix.interpolate_xforms(os.xform, c, xf, 1.0/frames)
            if frames <= 1:
                del mt[os]
            else:
                mt[os][2] = frames-1
        else:
            os.xform = xf
            del mt[os]

    if len(mt) == 0:
        from chimera import triggers
        triggers.deleteHandler('new frame', h)

# -----------------------------------------------------------------------------
#
def random_translation_step(center, radius):

    v = random_direction()
    import Matrix as M
    from random import random
    r = radius * random()
    tf = M.translation_matrix(M.linear_combination(1, center, r, v))
    return tf

# -----------------------------------------------------------------------------
#
def random_translation(xyz_min, xyz_max):

    from random import random
    shift = [x0+random()*(x1-x0) for x0,x1 in zip(xyz_min, xyz_max)]
    import Matrix
    tf = Matrix.translation_matrix(shift)
    return tf

# -----------------------------------------------------------------------------
#
def random_rotation():

    y, z = random_direction(), random_direction()
    import Matrix
    f = Matrix.orthonormal_frame(z, y)
    return ((f[0][0], f[1][0], f[2][0], 0),
            (f[0][1], f[1][1], f[2][1], 0),
            (f[0][2], f[1][2], f[2][2], 0))

# -----------------------------------------------------------------------------
#
def random_direction():

    z = (1,1,1)
    from Matrix import norm, normalize_vector
    from random import random
    while norm(z) > 1:
        z = (1-2*random(), 1-2*random(), 1-2*random())
    return normalize_vector(z)

# -----------------------------------------------------------------------------
#
def any_close_tf(tf, tflist, angle_tolerance, shift_tolerance, shift_point):

    import Matrix
    xf = Matrix.chimera_xform(tf)
    for i, t in enumerate(tflist):
        if Matrix.same_xform(Matrix.chimera_xform(t), xf,
                             angle_tolerance, shift_tolerance, shift_point):
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

# -----------------------------------------------------------------------------
#
def test():

    import VolumeViewer as v
    vol = v.active_volume()
    from chimera import selection as s
    atoms = s.currentAtoms()
    n = 500
    from time import clock
    t0 = clock()
    flist, outside = fit_search(atoms, vol, n)
    t1 = clock()
    print('found %d fits with %d searches, %d outside, in %.2f seconds' % (len(flist), n, outside, t1-t0))
    import Matrix
    xflist = [Matrix.chimera_xform(tf) for s,tf,c in flist if s > 1]
    mset = set([a.molecule for a in atoms])
    import Molecule
    from chimera import openModels
    for xf in xflist:
        for m in mset:
            mc = Molecule.copy_molecule(m)
            openModels.add([mc])
            mc.openState.xform = m.openState.xform
            mc.openState.globalXform(xf)
    print('top average map values and frequency', [(s,c) for s,tf,c in flist[:10]])
