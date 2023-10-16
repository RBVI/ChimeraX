# vim: set expandtab shiftwidth=4 softtabstop=4:

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

    bounds = volume.surface_bounds()
    if bounds is None:
        xyz_min, xyz_max = volume.xyz_bounds(step = 1)
        from chimerax.geometry import Bounds
        bounds = Bounds(xyz_min, xyz_max)

    asym_center_f = (.75,.55,.55)
    asym_center = tuple(x0 + (x1-x0)*f
                        for x0, x1, f in zip(bounds.xyz_min, bounds.xyz_max, asym_center_f)) 

    from chimerax.geometry import translation, identity
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
    from chimerax.geometry import bins
    b = bins.Binned_Transforms(angle_tolerance*pi/180, shift_tolerance, center)
    fo = {}
    from .fitmap import locate_maximum
    for i in range(n):
        if request_stop_cb and request_stop_cb('Fit %d of %d' % (i+1,n)):
            break
        shift = ((random_translation(bounds) if radius is None
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
                m.position = tf

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

        from chimerax.atomic import Structure
        mpos_list = [(m,position) for m,position in self.model_transforms()
                    if isinstance(m, Structure)]
        copies = [m.copy() for m, position in mpos_list]
        if copies:
            session = copies[0].session
            session.models.add(copies)

        for c, (m,pos) in zip(copies, mpos_list):
            c.position = pos

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
        from chimerax.map import Volume
        vtf = [(m,tf) for m, tf in zip(self.models, self.transforms)
               if m and not m.was_deleted and isinstance(m, Volume)]
        if len(vtf) != 1:
            return None
        m, tf = vtf[0]

        # Find grid points in base volume local coordinates.
        matrix, xyz_to_ijk_tf = m.matrix_and_transform(None, subregion = None,
                                                       step = None)
        threshold = m.minimum_surface_level
        from chimerax.map import high_indices
        points_int = high_indices(matrix, threshold)
        from numpy import float32
        points = points_int.astype(float32)
        (tf * xyz_to_ijk_tf.inverse()).transform_points(points, in_place = True)

        # Transform points by volume symmetries and count how many are inside
        # contour of map m.
        tfinv = tf.inverse()
        inside = 0
        for s in symlist:
            if not s.is_identity():
                p = points.copy()
                (tfinv * s).transform_points(p, in_place = True)
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
                       FM.transformation_matrix_message(mmap, v, self.fit_transform(mmap)))
        else:
            mols = self.fit_molecules()
            message = FM.atom_fit_message(mols, v, self.stats)
            message += '\n'.join([FM.transformation_matrix_message(m,v,self.fit_transform(m))
                                  for m in mols])

        return message

    # -------------------------------------------------------------------------
    #
    def fit_map(self):

        from chimerax.map import Volume
        vols = [m for m in self.models if isinstance(m, Volume)]
        mmap = vols[0] if vols else None
        return mmap

    # -------------------------------------------------------------------------
    #
    def fit_molecules(self):

        from chimerax.atomic import Structure
        mols = [m for m in self.models if isinstance(m, Structure)]
        return mols

    # -------------------------------------------------------------------------
    #
    def fit_transform(self, model):

        for m, tf in zip(self.models, self.transforms):
            if m is model:
                return tf
        return None

# -----------------------------------------------------------------------------
#
def save_fits(session, fits, path = None):

    mlist = sum([f.fit_molecules() for f in fits], [])
    if len(mlist) == 0:
        session.logger.warning('No fits of molecules chosen from list.')
        return

    idir = ifile = None
    vlist = [f.volume for f in fits]
    pmlist = [m for m in mlist + vlist if hasattr(m, 'filename')]
    if pmlist:
        for m in pmlist:
            import os.path
            dpath, fname = os.path.split(m.filename)
            base, suf = os.path.splitext(fname)
            if ifile is None:
                suffix = '_fit%d.pdb' if len(fits) > 1 else '_fit.pdb'
                ifile = base + suffix
            if dpath and idir is None:
                idir = dpath

    if path is None:
        from chimerax.ui.open_save import SaveDialog
        d = SaveDialog(session, caption = 'Save Fit Molecules',
                       data_formats = [session.data_formats['PDB']],
                       directory = idir)
        if ifile:
            d.selectFile(ifile)
        if not d.exec():
            return
        paths = d.selectedFiles()
        if paths:
            path = paths[0]
        else:
            return
        
    if len(fits) > 1 and path.find('%d') == -1:
        base, suf = os.path.splitext(path)
        path = base + '_fit%d' + suf

    from chimerax.pdb import save_pdb
    deleted = 0
    for i, fit in enumerate(fits):
        p = path if len(fits) == 1 else path % (i+1)
        fit.place_models(session)
        models = fit.fit_molecules()
        mfits = [m for m in models if not m.deleted]
        deleted += len(models) - len(mfits)
        if mfits:
            save_pdb(session, p, models = mfits, rel_model = fit.volume)

    if deleted:
        session.logger.warning(f'{deleted} fit molecules were deleted and cannot be saved')
        
# -----------------------------------------------------------------------------
#
def save_fit_positions_and_metrics(fit_list, path, delimiter = ' ', float_precision = 5):
    lines = []
    metrics = ('correlation', 'correlation about mean', 'overlap', 'average map value', 'points', 'atoms outside contour', 'clash', 'contour level', 'steps', 'shift', 'angle')
    ntf = 1
    for fit in fit_list:
        ntf = len(fit.transforms)
        values = []
        for tf in fit.transforms:
            (r00,r01,r02,t0),(r10,r11,r12,t1),(r20,r21,r22,t2) = tf.matrix
            values.extend((r00,r01,r02,r10,r11,r12,r20,r21,r22,t0,t1,t2))
        values.extend([fit.stats.get(attr, None) for attr in metrics])
        if float_precision is not None:
            format = f'%.{float_precision}g'
            values =  [(format % v if isinstance(v, float) else v) for v in values]
        lines.append(delimiter.join(str(v) for v in values))

    tf_fields = ('Rxx','Rxy','Rxz','Ryx','Ryy','Ryz','Rzx','Rzy','Rzz','Tx','Ty','Tz')
    headings = []
    for i in range(ntf):
        headings.extend(tf_fields)
    headings.extend(metrics)
    fields = delimiter.join(h.replace(' ','_') for h in headings)

    text = fields + '\n' + '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text)
    
# -----------------------------------------------------------------------------
#
def fit_order(f):

    return (f.correlation(), f.average_map_value())

# -----------------------------------------------------------------------------
#
def move_models(models, transforms, base_model, frames, session):

    if not hasattr(session, '_move_table'):
        session._move_table = {}            # Map motion handlers for animating moves to fit positions

    move_table = session._move_table
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

    mt = session._move_table
    for m, (rxf, base_model, frames) in tuple(mt.items()):
        if m.was_deleted or base_model.was_deleted:
            del mt[m]
            continue
        tf = base_model.position * rxf
        b = m.bounds()
        if b:
            c = .5 * (b.xyz_min + b.xyz_max)
            m.position = m.position.interpolate(tf, c, 1.0/frames)
            if frames <= 1:
                del mt[m]
            else:
                mt[m][2] = frames-1
        else:
            m.position = tf
            del mt[m]

    if len(mt) == 0:
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

# -----------------------------------------------------------------------------
#
def random_translation_step(center, radius):

    v = random_direction()
    from random import random
    r = radius * random()
    from chimerax.geometry import translation
    tf = translation(center + r*v)
    return tf

# -----------------------------------------------------------------------------
#
def random_translation(bounds):

    from random import random
    shift = [x0+random()*(x1-x0) for x0,x1 in zip(bounds.xyz_min, bounds.xyz_max)]
    from chimerax.geometry import translation
    tf = translation(shift)
    return tf

# -----------------------------------------------------------------------------
#
def random_rotation():

    y, z = random_direction(), random_direction()
    from chimerax.geometry import orthonormal_frame
    f = orthonormal_frame(z, y)
    return f

# -----------------------------------------------------------------------------
#
def random_direction():

    z = (1,1,1)
    from chimerax.geometry import norm, normalize_vector
    from random import random
    while norm(z) > 1:
        z = (1-2*random(), 1-2*random(), 1-2*random())
    return normalize_vector(z)

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

    if sym_list is None or len(sym_list) == 0:
        return tf

    from chimerax.geometry import distance
    import numpy as n
    i = n.argmin([distance(sym*tf*center, ref_point) for sym in sym_list])
    if sym_list[i].is_identity():
        return tf
    return sym_list[i]*tf
