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
def read_ihm(session, filename, name, *args, load_linked_files = True, fetch_templates = False,
             show_sphere_crosslinks = True, show_atom_crosslinks = False, **kw):
    """Read an integrative hybrid models file creating sphere models and restraint models

    :param filename: either the name of a file or a file-like object

    Extra arguments are ignored.
    """

    if hasattr(filename, 'read'):
        # Got a stream
        stream = filename
        filename = stream.name
        stream.close()

    from os.path import basename, splitext, dirname
    ihm_dir = dirname(filename)
    name = splitext(basename(filename))[0]
    from chimerax.core.models import Model
    ihm_model = Model(name, session)

    # Read mmcif tables
    table_names = ['ihm_struct_assembly',  	# Asym ids, entity ids, and entity names
                   'ihm_model_list',		# Model groups
                   'ihm_sphere_obj_site',	# Bead model for each cluster
                   'ihm_cross_link_restraint',	# Crosslinks
                   'ihm_ensemble_info',		# Names of ensembles, e.g. cluster 1, 2, ...
                   'ihm_gaussian_obj_ensemble',	# Distribution of ensemble models
                   'ihm_ensemble_localization', # Distribution of ensemble models
                   'ihm_dataset_other',		# Comparative models, EM data, DOI references
                   'ihm_starting_model_details', # Starting models, including compararative model templates
    ]
    from chimerax.core.atomic import mmcif
    table_list = mmcif.get_mmcif_tables(filename, table_names)
    tables = dict(zip(table_names, table_list))

    # Assembly composition
    acomp = assembly_components(tables['ihm_struct_assembly'])

    # Starting atomic models, including experimental and comparative structures and templates.
    xmodels, cmodels, seqmodels = create_starting_models(session, tables['ihm_starting_model_details'],
                                                         tables['ihm_dataset_other'], acomp,
                                                         load_linked_files, fetch_templates, ihm_model, ihm_dir)

    # Sphere models, ensemble models, groups
    smodels, emodels, gmodels = create_sphere_models(session, tables['ihm_model_list'],
                                                     tables['ihm_sphere_obj_site'], acomp, ihm_dir)
    
    # Align starting models to first sphere model
    if xmodels:
        align_atomic_models_to_spheres(xmodels, smodels)
    if cmodels:
        align_atomic_models_to_spheres(cmodels, smodels)

    # Crosslinks
    xlinks = create_crosslinks(session, tables['ihm_cross_link_restraint'],
                               show_sphere_crosslinks, smodels, emodels,
                               show_atom_crosslinks, xmodels+cmodels,
                               ihm_model)

    # 2D electron microscopy projections
    em2d = create_2dem_images(session, tables['ihm_dataset_other'], ihm_dir, ihm_model)

    # Added sphere models
    ihm_model.add(gmodels)
    
    # Ensemble localization
    pgrids = create_localization_maps(session, tables['ihm_ensemble_info'],
                                      tables['ihm_ensemble_localization'],
                                      tables['ihm_gaussian_obj_ensemble'],
                                      ihm_dir, ihm_model)
    xldesc = ', '.join('%d %s crosslinks' % (len(xls),type) for type,xls in xlinks.items())
    msg = ('Opened IHM file %s\n'
           ' %d xray/nmr models, %d comparative models, %d sequence alignments, %d templates\n'
           ' %s, %d 2D electron microscopy images\n'
           ' %d sphere models, %d ensembles with %s models, %d localization maps' %
           (filename, len(xmodels), len(cmodels),
            len(seqmodels), sum([len(sqm.db_templates) for sqm in seqmodels], 0),
            xldesc, len(em2d), len(smodels),
            len(emodels), ' and '.join('%d'%em.num_coord_sets for em in emodels),
            sum([len(pg.child_models()) for pg in pgrids], 0)))
    return [ihm_model], msg

# -----------------------------------------------------------------------------
#
class Assembly:
    def __init__(self, assembly_id, entity_id, entity_description, asym_id, seq_beg, seq_end):
        self.assembly_id = assembly_id
        self.entity_id = entity_id
        self.entity_description = entity_description
        self.asym_id = asym_id
        self.seq_begin = seq_beg
        self.seq_end = seq_end

# -----------------------------------------------------------------------------
#
def assembly_components(ihm_struct_assembly_table):
    sa_fields = [
        'assembly_id',
        'entity_description',
        'entity_id',
        'asym_id',
        'seq_id_begin',
        'seq_id_end']
    sa = ihm_struct_assembly_table.fields(sa_fields)
    acomp = [Assembly(aid, eid, edesc, asym_id, seq_beg, seq_end)
             for aid, edesc, eid, asym_id, seq_beg, seq_end in sa]
    return acomp
    
# -----------------------------------------------------------------------------
#
def create_sphere_models(session, ihm_model_list_table, ihm_sphere_obj_site_table,
                         acomp, ihm_dir):
    gmodels = make_sphere_model_groups(session, ihm_model_list_table)
    for g in gmodels[1:]:
        g.display = False	# Only show first group.
    
    smodels, emodels = make_sphere_models(session, ihm_model_list_table,
                                          ihm_sphere_obj_site_table, gmodels, acomp, ihm_dir)
    return smodels, emodels, gmodels

# -----------------------------------------------------------------------------
#
def make_sphere_model_groups(session, ihm_model_list_table):
    ml_fields = [
        'model_id',
        'model_group_id',
        'model_group_name',]
    ml = ihm_model_list_table.fields(ml_fields)
    gm = {}
    for mid, gid, gname in ml:
        gm.setdefault((gid, gname), []).append(mid)
    models = []
    from chimerax.core.models import Model
    for (gid, gname), mid_list in gm.items():
        m = Model(gname, session)
        m.ihm_group_id = gid
        m.ihm_model_ids = mid_list
        models.append(m)
    models.sort(key = lambda m: m.ihm_group_id)
    return models

# -----------------------------------------------------------------------------
#
def make_sphere_models(session, model_list_table, spheres_obj_site,
                       group_models, acomp, ihm_dir):
    ml_fields = [
        'model_id',
        'model_name',
        'model_group_id',
        'file',]
    ml = model_list_table.fields(ml_fields, allow_missing_fields = True)
    mnames = {mid:mname for mid,mname,gid,file in ml}

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
        mspheres.setdefault(model_id, {}).setdefault(asym_id, []).append((sb,se,xyz,r))

    aname = {a.asym_id:a.entity_description for a in acomp}
    smodels = []
    for mid, asym_spheres in mspheres.items():
        sm = SphereModel(mnames[mid], mid, session)
        smodels.append(sm)
        models = [SphereAsymModel(session, aname[asym_id], asym_id, mid, slist)
                  for asym_id, slist in asym_spheres.items()]
        models.sort(key = lambda m: m.asym_id)
        sm.add_asym_models(models)
    smodels.sort(key = lambda m: m.ihm_model_id)

    # Add sphere models to group
    gmodel = {id:g for g in group_models for id in g.ihm_model_ids}
    for sm in smodels:
        gmodel[sm.ihm_model_id].add([sm])

    # Undisplay all but first sphere model in each group
    gfound = set()
    for sm in smodels:
        g = gmodel[sm.ihm_model_id]
        if g in gfound:
            sm.display = False
        else:
            gfound.add(g)

    # Open ensemble sphere models that are not included in ihm sphere obj table.
    emodels = []
    from os.path import isfile, join
    smids = set(sm.ihm_model_id for sm in smodels)
    for mid, mname, gid, file in ml:
        path = join(ihm_dir, file)
        if file and isfile(path) and file.endswith('.pdb') and mid not in smids:
            from chimerax.core.atomic.pdb import open_pdb
            mlist,msg = open_pdb(session, path, mname,
                                 smart_initial_display = False, explode = False)
            sm = mlist[0]
            sm.display = False
            sm.ss_assigned = True	# Don't assign secondary structure to sphere model
            atoms = sm.atoms
            from chimerax.core.atomic.colors import chain_colors
            atoms.colors = chain_colors(atoms.residues.chain_ids)
            if isfile(path + '.crd'):
                from .coordsets import read_coordinate_sets
                read_coordinate_sets(path + '.crd', sm)
            gmodel[gid].add([sm])
            emodels.append(sm)

    # Copy bead radii from best score model to ensemble models
    if smodels and emodels:
        sams = smodels[0].child_models()
        from numpy import concatenate
        r = concatenate([sm.atoms.radii for sm in sams])
        for em in emodels:
            em.atoms.radii = r

    return smodels, emodels

# -----------------------------------------------------------------------------
#
def create_crosslinks(session, ihm_cross_link_restraint_table,
                      show_sphere_crosslinks, smodels, emodels,
                      show_atom_crosslinks, amodels,
                      ihm_model):
    if ihm_cross_link_restraint_table is None:
        return []
    
    # Crosslinks
    xlinks = crosslinks(ihm_cross_link_restraint_table)
    if len(xlinks) == 0:
        return xlinks
    
    xpbgs = []
    if show_sphere_crosslinks:
        # Create cross links for sphere models
        for i,smodel in enumerate(smodels):
            pbgs = make_crosslink_pseudobonds(session, xlinks, smodel.residue_sphere,
                                              name = smodel.ihm_model_id)
            if i == 0:
                # Show only multi-residue spheres and crosslink end-point spheres
                for sm in smodel.child_models():
                    satoms = sm.atoms
                    satoms.displays = False
                    satoms.filter(satoms.residues.names != '1').displays = True
                for pbg in pbgs:
                    a1,a2 = pbg.pseudobonds.atoms
                    a1.displays = True
                    a2.displays = True
            else:
                # Hide crosslinks for all but first sphere model
                for pbg in pbgs:
                    pbg.display = False
            xpbgs.extend(pbgs)

        if emodels and smodels:
            sindex = smodels[0].sphere_index
            for emodel in emodels:
                make_crosslink_pseudobonds(session, xlinks,
                                           ensemble_sphere_lookup(emodel, sindex),
                                           parent = emodel)
    if show_atom_crosslinks:
        # Create cross links for starting atomic models.
        if amodels:
            # These are usually missing disordered regions.
            pbgs = make_crosslink_pseudobonds(session, xlinks, atom_lookup(amodels))
            xpbgs.extend(pbgs)
    if pbgs:
        xl_group = Model('Crosslinks', session)
        xl_group.add(xpbgs)
        ihm_model.add([xl_group])

    return xlinks

# -----------------------------------------------------------------------------
#
class Crosslink:
    def __init__(self, asym1, seq1, asym2, seq2, dist):
        self.asym1 = asym1
        self.seq1 = seq1
        self.asym2 = asym2
        self.seq2 = seq2
        self.distance = dist

# -----------------------------------------------------------------------------
#
def crosslinks(xlink_restraint_table):

    xlink_fields = [
        'asym_id_1',
        'seq_id_1',
        'asym_id_2',
        'seq_id_2',
        'type',
        'distance_threshold'
        ]
    xlink_rows = xlink_restraint_table.fields(xlink_fields)
    xlinks = {}
    for asym_id_1, seq_id_1, asym_id_2, seq_id_2, type, distance_threshold in xlink_rows:
        xl = Crosslink(asym_id_1, int(seq_id_1), asym_id_2, int(seq_id_2), float(distance_threshold))
        xlinks.setdefault(type, []).append(xl)

    return xlinks

# -----------------------------------------------------------------------------
#
def make_crosslink_pseudobonds(session, xlinks, atom_lookup,
                               name = None,
                               parent = None,
                               radius = 1.0,
                               color = (0,255,0,255),		# Green
                               long_color = (255,0,0,255)):	# Red
    
    pbgs = []
    new_pbgroup = session.pb_manager.get_group if parent is None else parent.pseudobond_group
    for type, xlist in xlinks.items():
        xname = '%d %s crosslinks' % (len(xlist), type)
        if name is not None:
            xname += ' ' + name
        g = new_pbgroup(xname)
        pbgs.append(g)
        missing = []
        apairs = {}
        for xl in xlist:
            a1 = atom_lookup(xl.asym1, xl.seq1)
            a2 = atom_lookup(xl.asym2, xl.seq2)
            if (a1,a2) in apairs or (a2,a1) in apairs:
                # Crosslink already created between multiresidue beads
                continue
            if a1 and a2 and a1 is not a2:
                b = g.new_pseudobond(a1, a2)
                b.color = long_color if b.length > xl.distance else color
                b.radius = radius
                b.halfbond = False
                b.restraint_distance = xl.distance
            elif a1 is None:
                missing.append((xl.asym1, xl.seq1))
            elif a2 is None:
                missing.append((xl.asym2, xl.seq2))
        if missing:
            session.logger.info('Missing %d crosslink residues %s'
                                % (len(missing), ','.join('/%s:%d' for asym_id, seq_num in missing)))
                
    return pbgs

# -----------------------------------------------------------------------------
#
def create_2dem_images(session, ihm_dataset_other_table, ihm_dir, ihm_model):
    em2d = []
    fields = ['data_type', 'file']
    for data_type, filename in ihm_dataset_other_table.fields(fields, allow_missing_fields = True):
        if data_type == '2DEM class average' and filename.endswith('.mrc'):
            from os.path import join, isfile
            image_path = join(ihm_dir, filename)
            if isfile(image_path):
                from chimerax.core.map.volume import open_map
                maps,msg = open_map(session, image_path)
                v = maps[0]
                v.initialize_thresholds(vfrac = (0.01,1), replace = True)
                v.show()
                em2d.append(v)
    if em2d:
        em_group = Model('2D electron microscopy', session)
        em_group.add(em2d)
        ihm_model.add([em_group])

    return em2d

# -----------------------------------------------------------------------------
#
def create_localization_maps(session, ihm_ensemble_info_table,
                             ihm_ensemble_localization_table,
                             ihm_gaussian_obj_ensemble_table,
                             ihm_dir, ihm_model):
    pgrids = []
    ensembles_table = ihm_ensemble_info_table
    if ensembles_table is None:
        return pgrids

    gaussian_table = ihm_gaussian_obj_ensemble_table
    localization_table = ihm_ensemble_localization_table
    if localization_table is not None:
        pgrids = read_localization_maps(session, ensembles_table, localization_table, ihm_dir)
    elif gaussian_table is not None:
        pgrids = make_probability_grids(session, ensembles_table, gaussian_table)
    if pgrids:
        for g in pgrids[1:]:
            g.display = False	# Only show first ensemble
        el_group = Model('Ensemble localization', session)
        el_group.display = False
        el_group.add(pgrids)
        ihm_model.add([el_group])
        
    return pgrids

# -----------------------------------------------------------------------------
#
def read_localization_maps(session, ensemble_table, localization_table,
                           ihm_dir, level = 0.2, opacity = 0.5):
    '''Level sets surface threshold so that fraction of mass is outside the surface.'''

    ensemble_fields = ['ensemble_id', 'model_group_id', 'num_ensemble_models']
    ens = ensemble_table.fields(ensemble_fields)
    ens_group = {id:(gid,int(n)) for id, gid, n in ens}
    
    loc_fields = ['asym_id', 'ensemble_id', 'file']
    loc = localization_table.fields(loc_fields)
    ens = {}
    for asym_id, ensemble_id, file in loc:
        ens.setdefault(ensemble_id, []).append((asym_id, file))

    pmods = []
    from chimerax.core.models import Model
    from chimerax.core.map.volume import open_map
    from chimerax.core.atomic.colors import chain_rgba
    from os.path import join
    for ensemble_id in sorted(ens.keys()):
        asym_loc = ens[ensemble_id]
        gid, n = ens_group[ensemble_id]
        m = Model('Ensemble %s of %d models' % (ensemble_id, n), session)
        pmods.append(m)
        for asym_id, filename in sorted(asym_loc):
            map_path = join(ihm_dir, filename)
            maps,msg = open_map(session, map_path, show = False, show_dialog=False)
            color = chain_rgba(asym_id)[:3] + (opacity,)
            v = maps[0]
            ms = v.matrix_value_statistics()
            vlev = ms.mass_rank_data_value(level)
            v.set_parameters(surface_levels = [vlev], surface_colors = [color])
            v.show_in_volume_viewer = False
            v.show()
            m.add([v])

    return pmods

# -----------------------------------------------------------------------------
#
def make_probability_grids(session, ensemble_table, gaussian_table,
                           level = 0.2, opacity = 0.5):
    '''Level sets surface threshold so that fraction of mass is outside the surface.'''

    ensemble_fields = ['ensemble_id', 'model_group_id', 'num_ensemble_models']
    ens = ensemble_table.fields(ensemble_fields)
    ens_group = {id:(gid,int(n)) for id, gid, n in ens}
    
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
    gauss_rows = gaussian_table.fields(gauss_fields)
    from numpy import array, float64
    for asym_id, x, y, z, w, c11, c12, c13, c21, c22, c23, c31, c32, c33, eid in gauss_rows:
        center = array((float(x), float(y), float(z)), float64)
        weight = float(w)
        covar = array(((float(c11),float(c12),float(c13)),
                       (float(c21),float(c22),float(c23)),
                       (float(c31),float(c32),float(c33))), float64)
        cov.setdefault(eid, {}).setdefault(asym_id, []).append((weight, center, covar))

    # Compute probability volume models
    pmods = []
    from chimerax.core.models import Model
    from chimerax.core.map import volume_from_grid_data
    from chimerax.core.atomic.colors import chain_rgba
    
    for ensemble_id in sorted(cov.keys()):
        asym_gaussians = cov[ensemble_id]
        gid, n = ens_group[ensemble_id]
        m = Model('Ensemble %s of %d models' % (ensemble_id, n), session)
        pmods.append(m)
        for asym_id in sorted(asym_gaussians.keys()):
            g = probability_grid(asym_gaussians[asym_id])
            g.name = '%s Gaussians' % asym_id
            g.rgba = chain_rgba(asym_id)[:3] + (opacity,)
            v = volume_from_grid_data(g, session, show_data = False,
                                      open_model = False, show_dialog = False)
            v.initialize_thresholds()
            ms = v.matrix_value_statistics()
            vlev = ms.mass_rank_data_value(level)
            v.set_parameters(surface_levels = [vlev])
            v.show_in_volume_viewer = False
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
def create_starting_models(session, ihm_starting_model_details_table,
                           ihm_dataset_other_table, acomp,
                           load_linked_files, fetch_templates, ihm_model, ihm_dir):

    # Experimental starting models.
    # Comparative model templates.
    dataset_entities = {}
    xmodels = tmodels = seqmodels = []
    starting_models = ihm_starting_model_details_table
    if starting_models:
        dataset_entities, xmodels, tmodels, seqmodels = \
            read_starting_models(session, starting_models, fetch_templates, ihm_dir)
        if xmodels:
            from chimerax.core.models import Model
            am_group = Model('Experimental models', session)
            am_group.add(xmodels)
            ihm_model.add([am_group])

    # Comparative models
    cmodels = []
    datasets_table = ihm_dataset_other_table
    if datasets_table and load_linked_files:
        cmodels = read_linked_datasets(session, datasets_table, acomp, dataset_entities, ihm_dir)
        if cmodels:
            from chimerax.core.models import Model
            comp_group = Model('Comparative models', session)
            comp_group.add(cmodels)
            ihm_model.add([comp_group])

    # Align templates with comparative models
    if tmodels and cmodels:
        # Group templates with grouping name matching comparative model name.
        tg = group_template_models(session, tmodels, cmodels)
        ihm_model.add([tg])
        # Align templates to comparative models using matchmaker.
        align_template_models(session, cmodels)

    if seqmodels:
        from chimerax.core.models import Model
        sa_group = Model('Sequence Alignments', session)
        sa_group.add(seqmodels)
        ihm_model.add([sa_group])
        assign_comparative_models_to_sequences(cmodels, seqmodels)

    return xmodels, cmodels, seqmodels

# -----------------------------------------------------------------------------
#
def read_starting_models(session, starting_models, fetch_templates, ihm_dir):
    fields = ['entity_id', 'asym_id', 'seq_id_begin', 'seq_id_end', 'starting_model_source',
              'starting_model_db_name', 'starting_model_db_code', 'starting_model_db_pdb_auth_asym_id',
              'dataset_list_id', 'alignment_file']
    rows = starting_models.fields(fields, allow_missing_fields = True)
    dataset_entities = {}
    xmodels = []
    tmodels = []
    seqpaths = []	# Sequence alignment files for comparative model templates
    for eid, asym_id, seq_beg, seq_end, source, db_name, db_code, db_asym_id, did, seqfile in rows:
        dataset_entities[did] = (eid, asym_id)
        if (source in ('experimental model', 'comparative model') and
            db_name == 'PDB' and db_code != '?'):
            if source == 'comparative model' and not fetch_templates:
                models = []
            else:
                from chimerax.core.atomic.mmcif import fetch_mmcif
                models, msg = fetch_mmcif(session, db_code, smart_initial_display = False)
                name = '%s %s' % (db_code, db_asym_id)
                for m in models:
                    keep_one_chain(m, db_asym_id)
                    m.name = name
                    m.entity_id = eid
                    m.asym_id = asym_id
                    m.seq_begin, m.seq_end = int(seq_beg), int(seq_end)
                    m.dataset_id = did
                    show_colored_ribbon(m, asym_id)
            if source == 'experimental model':
                xmodels.extend(models)
            elif source == 'comparative model':
                tmodels.extend(models)
                if seqfile:
                    from os.path import join, isfile
                    p = join(ihm_dir, seqfile)
                    if isfile(p):
                        seqpaths.append(((p, asym_id, did), (db_name, db_code, db_asym_id)))

    # Make models for comparative model template alignments
    from collections import OrderedDict
    alignments = OrderedDict()
    for a,db in seqpaths:
        alignments.setdefault(a, []).append(db)
    seqmodels = [SequenceAlignmentModel(session, alignment_file, asym_id, dataset_id, db_templates)
                 for (alignment_file, asym_id, dataset_id), db_templates in alignments.items()]
            
    return dataset_entities, xmodels, tmodels, seqmodels

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class SequenceAlignmentModel(Model):
    def __init__(self, session, alignment_file, asym_id, dataset_id, db_templates):
        self.alignment_file = alignment_file
        self.asym_id = asym_id			# Identifies comparative model
        self.dataset_id = dataset_id		# Identifies comparative model
        self.db_templates = db_templates	# List of (db_name, db_code, db_asym_id), e.g. ('PDB', '1xyz', 'B')
        self.template_models = []		# Filled in after templates fetched.
        self.comparative_model = None
        self.alignment = None
        from os.path import basename
        Model.__init__(self, basename(alignment_file), session)
        self.display = False

    def _get_display(self):
        a = self.alignment
        if a is not None:
            for v in a.viewers:
                if v.displayed():
                    return True
        return False
    def _set_display(self, display):
        a = self.alignment
        if display:
            if a is None and len(self.template_models) == 0:
                self.fetch_template_models()
            self.show_alignment()
            if a is None:
                self.align_templates()

        elif a:
            for v in a.viewers:
                v.display(False)
    display = property(_get_display, _set_display)

    def show_alignment(self):
        a = self.alignment
        if a is None:
            from chimerax.seqalign.parse import open_file
            a = open_file(self.session, None, self.alignment_file,
                          auto_associate=False, return_vals='alignments')[0]
            self.alignment = a
            # Associate templates with sequences in alignment.
            tmap = {'%s%s' % (tm.pdb_id.lower(), tm.pdb_chain_id) : tm
                    for tm in self.template_models}
            if tmap:
                for seq in a.seqs:
                    tm = tmap.get(seq.name)
                    if tm:
                        a.associate(tm.chains[0], seq, force = True)
                        tm._associated_sequence = seq
            cm = self.comparative_model
            if cm:
                a.associate(cm.chains[0], a.seqs[-1], force = True)
        else:
            for v in a.viewers:
                v.display(True)
        return a

    def align_templates(self):
        a = self.alignment
        cm = self.comparative_model
        if a and cm:
            for tm in self.template_models:
                if tm._associated_sequence:
                    results = a.match(cm.chains[0], [tm.chains[0]], iterate=None)
                    if results:
                        # Show only matched residues
                        # TODO: Might show full interval of residues with unused
                        #       insertions colored gray
                        tmatoms = results[0][0]
                        tm.residues.ribbon_displays = False
                        tmatoms.unique_residues.ribbon_displays = True

    def fetch_template_models(self):
        for db_name, db_code, db_asym_id in self.db_templates:
            if db_name == 'PDB' and len(db_code) == 4:
                from chimerax.core.atomic.mmcif import fetch_mmcif
                models, msg = fetch_mmcif(self.session, db_code, smart_initial_display = False)
                name = '%s %s' % (db_code, db_asym_id)
                for m in models:
                    m.pdb_id = db_code
                    m.pdb_chain_id = db_asym_id
                    m.asym_id = self.asym_id
                    m.dataset_id = self.dataset_id	# For locating comparative model
                    keep_one_chain(m, db_asym_id)
                    show_colored_ribbon(m, self.asym_id, color_offset = 80)
                self.add(models)
                self.template_models.extend(models)
                
# -----------------------------------------------------------------------------
#
def assign_comparative_models_to_sequences(cmodels, seqmodels):
    cmap = {(cm.dataset_id, cm.asym_id):cm for cm in cmodels}
    for sam in seqmodels:
        ckey = (sam.dataset_id, sam.asym_id)
        if ckey in cmap:
            sam.comparative_model = cmap[ckey]
    
# -----------------------------------------------------------------------------
#
def keep_one_chain(s, chain_id):
    atoms = s.atoms
    cids = atoms.residues.chain_ids
    dmask = (cids != chain_id)
    dcount = dmask.sum()
    if dcount > 0 and dcount < len(atoms):	# Don't delete all atoms if chain id not found.
        datoms = atoms.filter(dmask)
        datoms.delete()
    
# -----------------------------------------------------------------------------
#
def read_linked_datasets(session, datasets_table, acomp, dataset_entities, ihm_dir):
    '''Read linked data from ihm_dataset_other table'''
    lmodels = []
    fields = ['dataset_list_id', 'data_type', 'doi', 'content_filename', 'file']
    for did, data_type, doi, content_filename, file in datasets_table.fields(fields, allow_missing_fields = True):
        if data_type == 'Comparative model' and (content_filename.endswith('.pdb') or file.endswith('.cif')):
            from os.path import basename, isfile, join
            if file.endswith('.cif'):
                path = join(ihm_dir, file)
                name = basename(file)
                from chimerax.core.atomic.mmcif import open_mmcif
                models, msg = open_mmcif(session, path, name, smart_initial_display = False)
            else:
                from .doi_fetch import fetch_doi_archive_file
                pdbf = fetch_doi_archive_file(session, doi, content_filename)
                name = basename(content_filename)
                from chimerax.core.atomic.pdb import open_pdb
                models, msg = open_pdb(session, pdbf, name, smart_initial_display = False)
                pdbf.close()
            eid, asym_id = dataset_entities[did] if did in dataset_entities else (None, None)
            for m in models:
                m.dataset_id = did
                m.entity_id = eid
                m.asym_id = asym_id
                show_colored_ribbon(m, asym_id)
            lmodels.extend(models)
      
    return lmodels

# -----------------------------------------------------------------------------
#
def group_template_models(session, template_models, comparative_models):
    '''Place template models in groups named after their comparative model.'''
    mm = []
    tmodels = template_models
    from chimerax.core.models import Model
    for cm in comparative_models:
        did = cm.dataset_id
        templates = [tm for tm in tmodels if tm.dataset_id == did]
        if templates:
            tm_group = Model(cm.name, session)
            tm_group.add(templates)
            cm.template_models = templates
            mm.append(tm_group)
            tmodels = [tm for tm in tmodels if tm.dataset_id != did]
    if tmodels:
        from chimerax.core.models import Model
        tm_group = Model('extra templates', session)
        tm_group.add(tmodels)
        mm.append(tm_group)
    tg = Model('Template models', session)
    tg.display = False
    tg.add(mm)
    return tg

# -----------------------------------------------------------------------------
#
def align_template_models(session, comparative_models):
    for cm in comparative_models:
        tmodels = getattr(cm, 'template_models', [])
        if tmodels is None:
            continue
        catoms = cm.atoms
        rnums = catoms.residues.numbers
        for tm in tmodels:
            # Find range of comparative model residues that template was matched to.
            from numpy import logical_and
            cratoms = catoms.filter(logical_and(rnums >= tm.seq_begin, rnums <= tm.seq_end))
            print('match maker', tm.name, 'to', cm.name, 'residues', tm.seq_begin, '-', tm.seq_end)
            from chimerax.match_maker.match import cmd_match
            matches = cmd_match(session, tm.atoms, cratoms, iterate = False)
            fatoms, toatoms, rmsd, full_rmsd, tf = matches[0]
            # Color unmatched template residues gray.
            mres = fatoms.residues
            tres = tm.residues
            nonmatched_res = tres.subtract(mres)
            nonmatched_res.ribbon_colors = (170,170,170,255)
            # Hide unmatched template beyond ends of matching residues
            mnums = mres.numbers
            tnums = tres.numbers
            tres.filter(tnums < mnums.min()).ribbon_displays = False
            tres.filter(tnums > mnums.max()).ribbon_displays = False

# -----------------------------------------------------------------------------
#
def show_colored_ribbon(m, asym_id, color_offset = None):
    if asym_id is None:
        from numpy import random, uint8
        color = random.randint(128,255,(4,),uint8)
        color[3] = 255
    else:
        from chimerax.core.atomic.colors import chain_rgba8
        color = chain_rgba8(asym_id)
        if color_offset:
            from numpy.random import randint
            offset = randint(-color_offset,color_offset,(3,))
            for a in range(3):
                color[a] = max(0, min(255, color[a] + offset[a]))
    r = m.residues
    r.ribbon_colors = color
    r.ribbon_displays = True
    a = m.atoms
    a.colors = color
    a.displays = False

# -----------------------------------------------------------------------------
#
def align_atomic_models_to_spheres(amodels, smodels):
    asmodels = smodels[0].asym_model_map()
    for m in amodels:
        sm = asmodels.get(m.asym_id)
        if sm is None:
            continue
        # Align comparative model residue centers to sphere centers
        res = m.residues
        rnums = res.numbers
        rc = res.centers
        mxyz = []
        sxyz = []
        for rn, c in zip(rnums, rc):
            s = sm.residue_sphere(rn)
            if s:
                mxyz.append(c)
                sxyz.append(s.coord)
                # TODO: For spheres with multiple residues use average residue center
        if len(mxyz) >= 3:
            from chimerax.core.geometry import align_points
            from numpy import array, float64
            p, rms = align_points(array(mxyz,float64), array(sxyz,float64))
            m.position = p
            # print ('aligned %s, %d residues, rms %.4g' % (m.name, len(mxyz), rms))
    
# -----------------------------------------------------------------------------
#
def atom_lookup(models):
    amap = {}
    for m in models:
        res = m.residues
        for res_num, atom in zip(res.numbers, res.principal_atoms):
            amap[(m.asym_id, res_num)] = atom
    def lookup(asym_id, res_num, amap=amap):
        return amap.get((asym_id, res_num))
    return lookup
    
# -----------------------------------------------------------------------------
#
def ensemble_sphere_lookup(emodel, aindex):
    def lookup(asym_id, res_num, atoms=emodel.atoms, aindex=aindex):
        i = aindex.get((asym_id, res_num))
        return None if i is None else atoms[i]
    return lookup


# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class SphereModel(Model):
    def __init__(self, name, ihm_model_id, session):
        Model.__init__(self, name, session)
        self.ihm_model_id = ihm_model_id
        self._asym_models = {}
        self.sphere_index = {}	# Map chain_id, res_num to ensemble sphere index

    def add_asym_models(self, models):
        Model.add(self, models)
        am = self._asym_models
        for m in models:
            am[m.asym_id] = m

        si = self.sphere_index
        io = 0
        for m in models:
            for res_num, a in m._res_sphere.items():
                si[(m.asym_id, res_num)] = a.coord_index+io
            io += m.num_atoms
            
    def asym_model(self, asym_id):
        return self._asym_models.get(asym_id)

    def asym_model_map(self):
        return self._asym_models

    def residue_sphere(self, asym_id, res_num):
        return self.asym_model(asym_id).residue_sphere(res_num)
    
# -----------------------------------------------------------------------------
# Map chain id, res number to atom index for ensemble sphere models.
#
def ensemble_sphere_map(smodel):
    sindex = {}
    io = 0
    for smodel in smodels:
                    sindex.update({cr:i+io for cr,i in smodel.index_map.items()})
                    io += smodel.num_atoms
    
# -----------------------------------------------------------------------------
#
from chimerax.core.atomic import Structure
class SphereAsymModel(Structure):
    def __init__(self, session, name, asym_id, model_id, sphere_list):
        Structure.__init__(self, session, name = name, smart_initial_display = False)

        self.ihm_model_id = model_id
        self.asym_id = asym_id
        self._res_sphere = rs = {}	# res_num -> sphere atom
        
        from chimerax.core.atomic.colors import chain_rgba8
        color = chain_rgba8(asym_id)
        for (sb,se,xyz,r) in sphere_list:
            aname = 'CA'
            a = self.new_atom(aname, 'C')
            a.coord = xyz
            a.radius = r
            a.draw_mode = a.SPHERE_STYLE
            a.color = color
            rname = '%d' % (se-sb+1)
            # Convention on ensemble PDB files is beads get middle residue number of range
            rnum = sb + (sb-se+1)//2
            r = self.new_residue(rname, asym_id, rnum)
            r.add_atom(a)
            for s in range(sb, se+1):
                rs[s] = a
        self.new_atoms()

    def residue_sphere(self, res_num):
        return self._res_sphere.get(res_num)
    
