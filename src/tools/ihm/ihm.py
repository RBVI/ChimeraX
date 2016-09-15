# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
ihm: Integrative Hybrid Model file format support
=================================================
"""
def read_ihm(session, filename, name, *args, **kw):
    """Read an integrative hybrid models file creating sphere models and restraint models

    :param filename: either the name of a file or a file-like object

    Extra arguments are ignored.
    """

    if hasattr(filename, 'read'):
        # Got a stream
        stream = filename
        filename = stream.name
        stream.close()

    from os.path import basename, splitext
    name = splitext(basename(filename))[0]

    table_names = ['ihm_sphere_obj_site', 'ihm_cross_link_restraint', 'ihm_gaussian_obj_ensemble']
    from chimerax.core.atomic import mmcif
    spheres_obj_site, xlink_restraint, gaussian_obj = mmcif.get_mmcif_tables(filename, table_names)
        
    models = make_sphere_models(session, spheres_obj_site, name)
    xlinks = []
    if xlink_restraint is not None:
        xlinks = make_crosslink_pseudobonds(xlink_restraint, models)
    pgrids = []
    if gaussian_obj is not None:
        pgrids = make_probability_grids(session, gaussian_obj)

    msg = ('Opened IHM file %s containing %d sphere models, %d distance restraints, %d probability maps' %
           (filename, len(models), len(xlinks), len(pgrids)))
    return models + pgrids, msg

# -----------------------------------------------------------------------------
#
def make_sphere_models(session, spheres_obj_site, name):

    sos_fields = [
        'seq_id_begin',
        'seq_id_end',
        'asym_id',
        'cartn_x',
        'cartn_y',
        'cartn_z',
        'object_radius',
        'model_id']
    spheres = spheres_obj_site.fields(sos_fields)
    mspheres = {}
    for seq_beg, seq_end, asym_id, x, y, z, radius, model_id in spheres:
        sb, se = int(seq_beg), int(seq_end)
        xyz = float(x), float(y), float(z)
        r = float(radius)
        mid = int(model_id)
        mspheres.setdefault(mid, []).append((sb,se,asym_id,xyz,r))

    models = [IHMSphereModel(session, '%s %d' % (name,mid) , slist) for mid, slist in mspheres.items()]
    for m in models[1:]:
        m.display = False	# Only show first model.

    return models

# -----------------------------------------------------------------------------
#
def make_crosslink_pseudobonds(xlink_restraint, models,
                               radius = 1.0,
                               color = (0,255,0,255),		# Green
                               long_color = (255,0,0,255)):	# Red

    xlink_fields = [
        'asym_id_1',
        'seq_id_1',
        'asym_id_2',
        'seq_id_2',
        'type',
        'distance_threshold'
        ]
    xlink_rows = xlink_restraint.fields(xlink_fields)
    xlinks = {}
    for asym_id_1, seq_id_1, asym_id_2, seq_id_2, type, distance_threshold in xlink_rows:
        xl = ((asym_id_1, int(seq_id_1)), (asym_id_2, int(seq_id_2)), float(distance_threshold))
        xlinks.setdefault(type, []).append(xl)

    if xlinks:
        for m in models:
            for type, xl in xlinks.items():
                xname = '%d %s crosslinks' % (len(xl), type)
                g = m.pseudobond_group(xname)
                g.name = xname
                for r1, r2, d in xl:
                    s1, s2 = m.residue_sphere(*r1), m.residue_sphere(*r2)
                    if s1 and s2 and s1 is not s2:
                        b = g.new_pseudobond(s1, s2)
                        b.color = long_color if b.length > d else color
                        b.radius = radius
                        b.halfbond = False
                        b.restraint_distance = d

    return xlinks

# -----------------------------------------------------------------------------
#
def make_probability_grids(session, gaussian_obj, level = 0.2, opacity = 0.5):
    '''Level sets surface threshold so that fraction of mass is outside the surface.'''
    gauss_fields = ['asym_id',
                   'mean_cartn_x',
                   'mean_cartn_y',
                   'mean_cartn_z',
                   'weight',
                   'covariance_matrix[1][1]',
                   'covariance_matrix[1][2]',
                   'covariance_matrix[1][3]',
                   'covariance_matrix[2][1]',
                   'covariance_matrix[2][2]',
                   'covariance_matrix[2][3]',
                   'covariance_matrix[3][1]',
                   'covariance_matrix[3][2]',
                   'covariance_matrix[3][3]',
                   'ensemble_id']
    cov = {}	# Map model_id to dictionary mapping asym id to list of (weight,center,covariance) triples
    gauss_rows = gaussian_obj.fields(gauss_fields)
    from numpy import array, float64
    for asym_id, x, y, z, w, c11, c12, c13, c21, c22, c23, c31, c32, c33, mid in gauss_rows:
        center = array((float(x), float(y), float(z)), float64)
        weight = float(w)
        covar = array(((float(c11),float(c12),float(c13)),
                       (float(c21),float(c22),float(c23)),
                       (float(c31),float(c32),float(c33))), float64)
        model_id = int(mid)
        cov.setdefault(model_id, {}).setdefault(asym_id, []).append((weight, center, covar))

    # Compute probability volume models
    pmods = []
    from chimerax.core.models import Model
    from chimerax.core.map import volume_from_grid_data
    from chimerax.core.atomic.colors import chain_rgba
    first_model_id = min(cov.keys())
    for model_id, asyms in cov.items():
        m = Model('Ensemble %d' % model_id, session)
        m.display = (model_id == first_model_id)
        pmods.append(m)
        for asym_id, clist in asyms.items():
            g = probability_grid(clist)
            g.name = asym_id
            g.rgba = chain_rgba(asym_id)[:3] + (opacity,)
            v = volume_from_grid_data(g, session, show_data = False,
                                      open_model = False, show_dialog = False)
            v.initialize_thresholds()
            ms = v.matrix_value_statistics()
            vlev = ms.mass_rank_data_value(level)
            v.set_parameters(surface_levels = [vlev])
            v.show()
            m.add([v])

    return pmods

# -----------------------------------------------------------------------------
#
def probability_grid(wcc, voxel_size = 5, cutoff_sigmas = 3):
    # Find bounding box for probability distribution
    from chimerax.core.geometry import Bounds, union_bounds
    from math import sqrt, ceil
    bounds = []
    for weight, center, covar in wcc:
        sigmas = [sqrt(covar[a,a]) for a in range(3)]
        xyz_min = [x-s for x,s in zip(center,sigmas)]
        xyz_max = [x+s for x,s in zip(center,sigmas)]
        bounds.append(Bounds(xyz_min, xyz_max))
    b = union_bounds(bounds)
    isize,jsize,ksize = [int(ceil(s  / voxel_size)) for s in b.size()]
    from numpy import zeros, float32, array
    a = zeros((ksize,jsize,isize), float32)
    xyz0 = b.xyz_min
    vsize = array((voxel_size, voxel_size, voxel_size), float32)

    # Add Gaussians to probability distribution
    for weight, center, covar in wcc:
        acenter = (center - xyz0) / vsize
        cov = covar.copy()
        cov *= 1/(voxel_size*voxel_size)
        add_gaussian(weight, acenter, cov, a)

    from chimerax.core.map.data import Array_Grid_Data
    g = Array_Grid_Data(a, origin = xyz0, step = vsize)
    return g

# -----------------------------------------------------------------------------
#
def add_gaussian(weight, center, covar, array):

    from numpy import linalg
    cinv = linalg.inv(covar)
    d = linalg.det(covar)
    from math import pow, sqrt, pi
    s = weight * pow(2*pi, -1.5) / sqrt(d)	# Normalization to sum 1.
    covariance_sum(cinv, center, s, array)

# -----------------------------------------------------------------------------
#
def covariance_sum(cinv, center, s, array):
    from numpy import dot
    from math import exp
    ksize, jsize, isize = array.shape
    i0,j0,k0 = center
    for k in range(ksize):
        for j in range(jsize):
            for i in range(isize):
                v = (i-i0, j-j0, k-k0)
                array[k,j,i] += s*exp(-0.5*dot(v, dot(cinv, v)))

from chimerax.core.map import covariance_sum

# -----------------------------------------------------------------------------
#
def register():
    from chimerax.core import io
    from chimerax.core.atomic import structure
    io.register_format("Integrative Hybrid Model", structure.CATEGORY, (".ihm",), ("ihm",),
                       open_func=read_ihm)

# -----------------------------------------------------------------------------
#
from chimerax.core.atomic import Structure
class IHMSphereModel(Structure):
    def __init__(self, session, name, sphere_list):
        Structure.__init__(self, session, name = name, smart_initial_display = False)

        self._res_sphere = rs = {}	# (asym_id, res_num) -> sphere atom
        
        from chimerax.core.atomic.colors import chain_rgba8
        for (sb,se,asym_id,xyz,r) in sphere_list:
            aname = ''
            a = self.new_atom(aname, 'H')
            a.coord = xyz
            a.radius = r
            a.draw_mode = a.SPHERE_STYLE
            a.color = chain_rgba8(asym_id)
            rname = '%d' % (se-sb+1)
            r = self.new_residue(rname, asym_id, sb)
            r.add_atom(a)
            for s in range(sb, se+1):
                rs[(asym_id,s)] = a
        self.new_atoms()

    def residue_sphere(self, asym_id, res_num):

        return self._res_sphere.get((asym_id, res_num))
    
