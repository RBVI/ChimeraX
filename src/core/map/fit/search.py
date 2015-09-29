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

    b = volume.bounds(positions = False)
    xyz_min, xyz_max = b if not b is None else volume.xyz_bounds(step = 1)
    asym_center_f = (.75,.55,.55)
    asym_center = tuple(x0 + (x1-x0)*f
                        for x0, x1, f in zip(xyz_min, xyz_max, asym_center_f)) 

    from ...geometry import translation, identity
    center = points.mean(axis=0)
    ctf = translation(-center)
    ijk_to_xyz_tf = volume.matrix_indices_to_xyz_transform(step = 1)
    xyz_to_ijk_tf = ijk_to_xyz_tf.inverse()
    data_array = volume.matrix(step = 1)
    vtfinv = volume.position.inverse()
    mtv_list = [vtfinv * m.position for m in models]

    flist = []
    outside = 0
    from math import pi
    from ...geometry import bins
    b = bins.Binned_Transforms(angle_tolerance*pi/180, shift_tolerance, center)
    fo = {}
    from .fitmap import locate_maximum
    for i in range(n):
        if request_stop_cb and request_stop_cb('Fit %d of %d' % (i+1,n)):
            break
        shift = ((random_translation(xyz_min, xyz_max) if radius is None
                  else random_translation_step(center, radius)) if shifts
                  else translation(center))
        rot = random_rotation() if rotations else identity()
        tf = shift * rot * ctf
        optimize = True
        max_opt = 2
        while optimize and max_opt > 0:
            p_to_ijk_tf = xyz_to_ijk_tf * tf
            move_tf, stats = \
              locate_maximum(points, point_weights, data_array, p_to_ijk_tf,
                             max_steps, ijk_step_size_min, ijk_step_size_max,
                             optimize_translation, optimize_rotation,
                             metric, request_stop_cb = None)
            ptf = tf * move_tf
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
            transforms = [ptf * mtv for mtv in mtv_list]
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

    vtf = volume.position * tf
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
            vtfinv = volume.position.inverse()
            transforms = [vtfinv * m.position for m in models]
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
    def place_models(self, session, frames = 0):

        if frames > 0:
            move_models(self.models, self.transforms, self.volume, frames, session)
        else:
            for m, tf in self.model_transforms():
                m.set_place(tf)

    # -------------------------------------------------------------------------
    #
    def model_transforms(self):

        mtf_list = []
        v = self.volume
        if v.was_deleted:
            return mtf_list
        for m, tf in zip(self.models, self.transforms):
            if m is None or m.was_deleted:
                continue
            vtf = v.position * tf
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
        if v is None or v.was_deleted:
            return None
        
        # Check if volume has symmetries.
        symlist = self.volume.data.symmetries
        if len(symlist) <= 1:
            return None

        # Look for exactly one map that was fit into volume.
        from .. import Volume
        vtf = [(m,tf) for m, tf in zip(self.models, self.transforms)
               if m and not m.was_deleted and isinstance(m, Volume)]
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
        (tf * xyz_to_ijk_tf.inverse()).move(points)

        # Transform points by volume symmetries and count how many are inside
        # contour of map m.
        tfinv = tf.inverse()
        inside = 0
        for s in symlist:
            if not s.is_identity():
                p = points.copy()
                (tfinv * s).move(p)
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

        from .. import Volume
        vols = [m for m in self.models if isinstance(m, Volume)]
        mmap = vols[0] if vols else None
        return mmap

    # -------------------------------------------------------------------------
    #
    def fit_molecules(self):

        from ...atomic import AtomicStructure
        mols = [m for m in self.models if isinstance(m, AtomicStructure)]
        return mols
    
# -----------------------------------------------------------------------------
#
def fit_order(f):

    return (f.correlation(), f.average_map_value())

# -----------------------------------------------------------------------------
#
def move_models(models, transforms, base_model, frames, session):

    if not hasattr(session, move_table):
        session.move_table = {}            # Map motion handlers for animating moves to fit positions

    move_table = session.move_table
    add = (len(move_table) == 0)
    if base_model.was_deleted:
        return
    for m, tf in zip(models, transforms):
        if m and not m.was_deleted:
            move_table[m] = [tf, base_model, frames]
    if move_table and add:
        cb = []
        def mstep(*_, mt = move_table):
            return move_step(mt, session)
        session.triggers.add_handler('new frame', mstep)

# -----------------------------------------------------------------------------
#
def move_step(move_table, session):

    mt = session.move_table
    for m, (rxf, base_model, frames) in tuple(mt.items()):
        if m.was_deleted or base_model.was_deleted:
            del mt[m]
            continue
        tf = base_model.position * rxf
        b = m.bounds(positions = False)
        if b:
            c = .5 * (b[0] + b[1])
            m.set_place(m.position.interpolate(tf, c, 1.0/frames))
            if frames <= 1:
                del mt[m]
            else:
                mt[m][2] = frames-1
        else:
            m.set_place(tf)
            del mt[m]

    if len(mt) == 0:
		from ...triggerset import DEREGISTER
        return DEREGISTER

# -----------------------------------------------------------------------------
#
def random_translation_step(center, radius):

    v = random_direction()
    from random import random
    r = radius * random()
    from ...geometry import translation
    tf = translation(center + r*v)
    return tf

# -----------------------------------------------------------------------------
#
def random_translation(xyz_min, xyz_max):

    from random import random
    shift = [x0+random()*(x1-x0) for x0,x1 in zip(xyz_min, xyz_max)]
    from ...geometry import translation
    tf = translation(shift)
    return tf

# -----------------------------------------------------------------------------
#
def random_rotation():

    y, z = random_direction(), random_direction()
    from ...geometry import place
    f = place.orthonormal_frame(z, y)
    return f

# -----------------------------------------------------------------------------
#
def random_direction():

    z = (1,1,1)
    from ...geometry import vector
    from random import random
    while vector.norm(z) > 1:
        z = (1-2*random(), 1-2*random(), 1-2*random())
    return vector.normalize_vector(z)

# -----------------------------------------------------------------------------
#
def any_close_tf(tf, tflist, angle_tolerance, shift_tolerance, shift_point):

    for i, t in enumerate(tflist):
        if t.same(tf, angle_tolerance, shift_tolerance, shift_point):
            return i
    return None

# -----------------------------------------------------------------------------
#
def unique_symmetry_position(tf, center, ref_point, sym_list):

    if len(sym_list) == 0:
        return tf

    from ...geometry.place import distance
    import numpy as n
    i = n.argmin([distance(sym*tf*center, ref_point) for sym in sym_list])
    if sym_list[i].is_identity():
        return tf
    return sym_list[i]*tf
