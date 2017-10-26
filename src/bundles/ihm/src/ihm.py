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
def read_ihm(session, filename, name, *args, load_ensembles = False, load_linked_files = True,
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

    m = IHMModel(session, filename,
                 load_ensembles = load_ensembles,
                 load_linked_files = load_linked_files,
                 show_sphere_crosslinks = show_sphere_crosslinks,
                 show_atom_crosslinks = show_atom_crosslinks)

    return [m], m.description

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class IHMModel(Model):
    def __init__(self, session, filename,
                 load_ensembles = False,
                 load_linked_files = True,
                 show_sphere_crosslinks = True,
                 show_atom_crosslinks = False):
    
        self.filename = filename
        from os.path import basename, splitext, dirname
        self.ihm_directory = dirname(filename)
        name = splitext(basename(filename))[0]
        self._file_info = None
        self._data_sets = None	# Map dataset_list_id to DataSet

        Model.__init__(self, name, session)

        self.tables = self.read_tables(filename)

        # Starting atomic models, including experimental and comparative structures and templates.
        stmodels, seqmodels = self.read_starting_models(load_linked_files)
        self.starting_models = stmodels
        self.sequence_alignment_models = seqmodels

        # Crosslinks
        xlinks, xlmodels = self.read_crosslinks()
        self.crosslink_models = xlinks

        # 2D and 3D electron microscopy projections
        em2d = self.read_2d_electron_microscopy_maps()
        em3d = self.read_3d_electron_microscopy_maps()
        emmodels = em2d + em3d
        self.electron_microscopy_models = emmodels

        # Make restraint model groups
        rmodels = xlmodels + emmodels
        if rmodels:
            r_group = Model('Restraints', self.session)
            r_group.add(rmodels)
            self.add([r_group])

        # Map model id to group id.
        mgroup = self.model_id_to_group_id()
        
        # Sphere models, ensemble models, atomic models
        smodels, emodels = self.make_sphere_models(mgroup, load_ensembles = load_ensembles)
        self.sphere_models = smodels
        self.ensemble_sphere_models = emodels
        amodels = self.read_atomic_models(filename, mgroup)
        self.atomic_models = amodels

        # Align 2DEM to projection position for first sphere model
        if smodels:
            s0 = smodels[0]
            for v in em2d:
                if hasattr(v, 'ihm_model_projections'):
                    p = v.ihm_model_projections.get(s0.ihm_model_ids[0])
                    if p:
                        v.position = p
                        
        # Add crosslinks to sphere models
        if show_sphere_crosslinks:
            self.create_result_model_crosslinks(xlinks, smodels, emodels, amodels, xlmodels)
            self.set_initial_sphere_display(smodels)
        if show_atom_crosslinks:
            self.create_starting_model_crosslinks(xlinks, stmodels, xlmodels)
    
        # Align starting models to first sphere model
        if smodels:
            # TODO: Align to first result model, could be spheres or atomic
            align_starting_models_to_spheres(stmodels, smodels[0])
    
        # Ensemble localization
        self.localization_models = lmaps = self.read_localization_maps()

        # Put sphere, ensemble, atomic models and localization maps into parent group models.
        self.group_result_models(smodels, emodels, amodels, lmaps, mgroup)

    def read_tables(self, filename):
        # Read ihm tables
        table_names = ['ihm_struct_assembly',  		# Asym ids, entity ids, and entity names
                       'ihm_model_list',		# Model groups
                       'ihm_sphere_obj_site',		# Bead model for each cluster
                       'ihm_cross_link_restraint',	# Crosslinks
                       'ihm_ensemble_info',		# Names of ensembles, e.g. cluster 1, 2, ...
                       'ihm_gaussian_obj_ensemble',	# Distribution of ensemble models
                       'ihm_localization_density_files', # Spatial distribution of ensemble models
                       'ihm_dataset_related_db_reference', # Starting model database ids
                       'ihm_dataset_external_reference', # Comparative models, EM data
                       'ihm_external_files',		# files in DOI archives, at URL or on local disk
                       'ihm_external_reference_info',	# DOI archive and URL external files
                       'ihm_starting_model_details', 	# Starting models
                       'ihm_starting_comparative_models', # Compararative model templates
                       'ihm_2dem_class_average_restraint', # 2D EM constraing projectionn of atomic model
                       'ihm_2dem_class_average_fitting', # 2D EM orientation relative to model
                       'ihm_3dem_restraint',		# 3d electron microscopy
                       ]
        from os.path import basename
        from chimerax.core.atomic import mmcif
        table_list = mmcif.get_mmcif_tables(filename, table_names)
        tables = dict(zip(table_names, table_list))
        return tables

    # -----------------------------------------------------------------------------
    #
    def asym_id_names(self):
        sat = self.tables['ihm_struct_assembly']
        sa_fields = [
            'entity_description',
            'asym_id',
            ]
        sa = sat.fields(sa_fields)
        anames = {asym_id : edesc for edesc, asym_id in sa}
        return anames

    # -----------------------------------------------------------------------------
    #
    def read_starting_models(self, load_linked_files):

        # Read experimental starting models
        xmodels = self.read_experimental_models()

        # Read comparative models
        cmodels = self.read_comparative_models() if load_linked_files else []

        # Read comparative model templates
        tmodels, seqmodels = self.read_template_models()

        # Associate comparative models with sequence alignments.
        if seqmodels:
            assign_comparative_models_to_sequences(cmodels, seqmodels)

        # Group starting models, sequence alignment and templates by asym id.
        models = xmodels + cmodels + seqmodels + tmodels
        if models:
            sm_group = Model('Starting models', self.session)
            sma = {}
            for m in models:
                sma.setdefault(m.asym_id, []).append(m)
            smg = []
            anames = self.asym_id_names()
            for asym_id in sorted(sma.keys()):
                am = sma[asym_id]
                name = '%s %s' % (anames[asym_id], asym_id)
                a_group = Model(name, self.session)
                a_group.add(am)
                smg.append(a_group)
                a_group.color = am[0].single_color	# Group color is first model color
            sm_group.add(smg)
            self.add([sm_group])

        return xmodels+cmodels, seqmodels

    # -----------------------------------------------------------------------------
    #
    def read_experimental_models(self):
        '''Read crystallography and NMR atomic starting models.'''
        xmodels = []

        starting_models = self.tables['ihm_starting_model_details']
        if not starting_models:
            return xmodels

        fields = ['asym_id', 'seq_id_begin', 'seq_id_end', 'starting_model_source',
                  'starting_model_auth_asym_id', 'dataset_list_id']
        rows = starting_models.fields(fields, allow_missing_fields = True)

        for asym_id, seq_beg, seq_end, source, auth_asym_id, did in rows:
            if source != 'experimental model':
                continue
            d = self.data_set(did, 'ihm_starting_model_details')
            if d is None:
                continue
            models = d.models(self.session)
            for m in models:
                keep_one_chain(m, auth_asym_id)
                m.name += ' ' + auth_asym_id
                show_colored_ribbon(m, asym_id)
            xmodels.extend(models)
            for m in models:
                m.asym_id = asym_id
                m.seq_begin, m.seq_end = int(seq_beg), int(seq_end)
                m.dataset_id = did
                m.comparative_model = False

        return xmodels
    
    # -----------------------------------------------------------------------------
    #
    def data_set(self, dataset_list_id, table_name):
        ds = self._data_sets
        if ds is None:
            self._data_sets = ds = {}
            dref = self.tables['ihm_dataset_related_db_reference']
            fields = ['dataset_list_id', 'db_name', 'accession_code']
            rows = dref.fields(fields, allow_missing_fields = True)
            for did, db_name, db_code in rows:
                ds[did] = DatabaseDataSet(db_name, db_code)
            deref = self.tables['ihm_dataset_external_reference']
            fields = ['dataset_list_id', 'file_id']
            rows = deref.fields(fields, allow_missing_fields = True)
            for did, file_id in rows:
                finfo = self.file_info(file_id)
                if finfo:
                    ds[did] = FileDataSet(finfo)
        if dataset_list_id not in ds:
            self.session.logger.warning('Data set id %s listed in table %s was not found '
                                        'in ihm_dataset_external_reference or ihm_dataset_related_db_reference tables'
                                        % (dataset_list_id, table_name))
            raise ValueError('bad data set id')
        return ds.get(dataset_list_id, None)
    
    # -----------------------------------------------------------------------------
    #
    def read_comparative_models(self):
        '''Read comparative models from the ihm_starting_model_details table'''
        lmodels = []
        sm_table = self.tables['ihm_starting_model_details']
        if not sm_table:
            return lmodels
        fields = ['starting_model_id', 'asym_id', 'starting_model_source', 'starting_model_auth_asym_id',
                  'dataset_list_id']
        rows = sm_table.fields(fields, allow_missing_fields = True)
        # TODO: Starting model can appear multiple times in table, with different sequence ranges.  Seems wrong.
        smfound = set()
        for smid, asym_id, data_type, auth_asym_id, did in rows:
            if data_type != 'comparative model':
                continue
            d = self.data_set(did, 'ihm_starting_model_details')
            if d is None:
                continue
            if smid in smfound:
                continue
            smfound.add(smid)
            models = d.models(self.session)
            for m in models:
                keep_one_chain(m, auth_asym_id)
                m.name += ' ' + auth_asym_id
                m.dataset_id = did
                m.asym_id = asym_id
                m.comparative_model = True
                show_colored_ribbon(m, asym_id)
            lmodels.extend(models)
      
        return lmodels

    # -----------------------------------------------------------------------------
    #
    def read_template_models(self):
        '''Read crystallography and NMR atomic starting models.'''
        tmodels = []
        seqmodels = []

        # Get info about comparative models.
        starting_models = self.tables['ihm_starting_model_details']
        if not starting_models:
            return tmodels, seqmodels
        fields = ['starting_model_id', 'asym_id', 'seq_id_begin', 'seq_id_end',
                  'starting_model_source', 'dataset_list_id']
        rows = starting_models.fields(fields, allow_missing_fields = True)
        smdetails = {sm_id:(asym_id, seq_beg, seq_end, did)
                     for sm_id, asym_id, seq_beg, seq_end, source, did in rows
                     if source == 'comparative model'}

        # Get info about templates
        comp_models = self.tables['ihm_starting_comparative_models']
        if not comp_models:
            return tmodels, seqmodels
        fields = ['starting_model_id','template_auth_asym_id',
                  'template_seq_id_begin', 'template_seq_id_end', 'template_dataset_list_id',
                  'alignment_file_id']
        rows = comp_models.fields(fields, allow_missing_fields = True)

        from collections import OrderedDict
        alignments = OrderedDict()  # Sequence alignments for comparative models
        for sm_id, auth_asym_id, tseq_beg, tseq_end, tdid, alignment_file_id in rows:
            d = self.data_set(tdid, 'ihm_starting_comparative_models')
            if d is None:
                continue
            if sm_id not in smdetails:
                continue
            # Template for a comparative model.
            asym_id, seq_beg, seq_end, cdid = smdetails[sm_id]
            tm = TemplateModel(self.session, asym_id, int(seq_beg), int(seq_end),
                               auth_asym_id, int(tseq_beg), int(tseq_end), d)
            tmodels.append(tm)
            if alignment_file_id != '.':
                sfinfo = self.file_info(alignment_file_id)
                if sfinfo is not None:
                    a = (sfinfo, asym_id, cdid)
                    sam = alignments.get(a)
                    if sam is None:
                        # Make sequence alignment model for comparative model
                        alignments[a] = sam = SequenceAlignmentModel(self.session, sfinfo, asym_id, cdid)
                    sam.add_template_model(tm)

        seqmodels = list(alignments.values())
        return tmodels, seqmodels

    # -----------------------------------------------------------------------------
    #
    def file_info(self, file_id):
        fmap = self._file_info
        if fmap is None:
            refs = {}
            ref_table = self.tables['ihm_external_reference_info']
            if ref_table:
                ref_fields = ['reference_id', 'reference_type', 'reference', 'refers_to', 'associated_url']
                rows = ref_table.fields(ref_fields, allow_missing_fields = True)
                for ref_id, ref_type, ref, content, url in rows:
                    refs[ref_id] = ExternalReference(ref_id, ref_type, ref, content, url)

            self._file_info = fmap = {}
            files_table = self.tables['ihm_external_files']
            if files_table:
                files_fields = ['id', 'reference_id', 'file_path']
                rows = files_table.fields(files_fields, allow_missing_fields = True)
                for f_id, ref_id, file_path in rows:
                    fpath = None if file_path == '.' else file_path
                    fmap[f_id] = FileInfo(f_id, refs.get(ref_id,None), fpath, self.ihm_directory)

        fi = fmap.get(file_id, None)
        return fi

    # -----------------------------------------------------------------------------
    #
    def model_names(self):
        mlt = self.tables['ihm_model_list']
        ml_fields = ['model_id', 'model_name']
        ml = mlt.fields(ml_fields, allow_missing_fields = True)
        mnames = {mid:mname for mid,mname in ml if mname}
        return mnames

    # -----------------------------------------------------------------------------
    #
    def make_sphere_models(self, model_group, group_coordsets = True, load_ensembles = False):
        mnames = self.model_names()

        sost = self.tables['ihm_sphere_obj_site']
        if sost is None:
            smodels = []
        else:
            sos_fields = [
                'seq_id_begin',
                'seq_id_end',
                'asym_id',
                'cartn_x',
                'cartn_y',
                'cartn_z',
                'object_radius',
                'model_id']
            spheres = sost.fields(sos_fields)
            mspheres = {}
            for seq_beg, seq_end, asym_id, x, y, z, radius, model_id in spheres:
                sb, se = int(seq_beg), int(seq_end)
                xyz = float(x), float(y), float(z)
                r = float(radius)
                mspheres.setdefault(model_id, []).append((asym_id,sb,se,xyz,r))
            smodels = self.make_sphere_models_by_group(mspheres, mnames, model_group, group_coordsets)

        # Open ensemble sphere models that are not included in ihm sphere obj table.
        emodels = self.load_sphere_model_ensembles(smodels) if load_ensembles else []

        return smodels, emodels

    # -----------------------------------------------------------------------------
    #
    def make_sphere_models_by_group(self, mspheres, mnames, model_group, group_coordsets):
        # Find sphere models by group
        msg = {}
        for mid, slist in mspheres.items():
            msg.setdefault(model_group[mid],[]).append((mid,slist))

        # Sort models in a group by model id.
        for ms in msg.values():
            ms.sort(key = lambda ms: ms[0])

        # Sort groups by id
        gs = list(msg.keys())
        gs.sort()
        
        smodels = []
        for g in gs:
            ms = msg[g]
            if group_coordsets and self.same_sphere_atoms(ms):
                # For groups with matching residue / atom names use coordinate set.
                mid, slist = ms[0]
                sm = SphereModel(self.session, mnames.get(mid, 'sphere model'), mid, slist)
                sm.ihm_group_id = g
                for mid, slist in ms[1:]:
                    sm.add_coordinates(mid, slist)
                smodels.append(sm)
                if len(ms) > 1:
                    sm.name = '%d models' % len(ms)
            else:
                # Make separate sphere models, do not use coordinate sets.
                for i, (mid, slist) in enumerate(ms):
                    sm = SphereModel(self.session, mnames.get(mid, 'sphere model'), mid, slist)
                    sm.ihm_group_id = g
                    sm.display = (i == 0)            # Undisplay all but first sphere model in each group
                    smodels.append(sm)

        return smodels

    # -----------------------------------------------------------------------------
    #
    def same_sphere_atoms(self, mslist):
        # Check if all sphere models have identical atoms in same order so that
        # a coordinate set could be used to represent them.
        sphere_ids = None
        for model_id, sphere_list in mslist:
            sids = [(asym_id,sb,se) for (asym_id, sb,se,xyz,r) in sphere_list]
            if sphere_ids is None:
                sphere_ids = sids
            elif sids != sphere_ids:
                return False
        return True
    
    # -----------------------------------------------------------------------------
    # Note ensemble models are AtomicStructure models, not SphereModel.
    #
    def load_sphere_model_ensembles(self, smodels):
        eit = self.tables['ihm_ensemble_info']
        ei_fields = ['ensemble_name', 'model_group_id', 'ensemble_file_id']
        ei = eit.fields(ei_fields, allow_missing_fields = True)
        emodels = []
        for mname, gid, file_id in ei:
            finfo = self.file_info(file_id)
            if finfo is None:
                continue
            fname = finfo.file_name
            if fname.endswith('.dcd'):
                gsm = [sm for sm in smodels if sm.ihm_group_id == gid]
                if len(gsm) != 1:
                    continue  # Don't have exactly one sphere model for this group id
                sm = gsm[0].copy(name = mname)
                dcd_path = finfo.path(self.session)
                from chimerax.md_crds.read_coords import read_coords
                read_coords(self.session, dcd_path, sm, format_name = 'dcd', replace=True)
                sm.active_coordset_id = 1
            elif fname.endswith('.pdb') or fname.endswith('.pdb.gz'):
                fstream = finfo.stream(self.session, uncompress = True)
                if fstream is None:
                    continue
                from chimerax.core.atomic.pdb import open_pdb
                mlist,msg = open_pdb(self.session, fstream, mname,
                                     auto_style = False, coordsets = True)
                sm = mlist[0]
            sm.ihm_group_id = gid
            sm.display = False
            sm.name += ' %d models' % sm.num_coordsets
            sm.ss_assigned = True	# Don't assign secondary structure to sphere model
            atoms = sm.atoms
            from chimerax.core.atomic.colors import chain_colors
            atoms.colors = chain_colors(atoms.residues.chain_ids)
            emodels.append(sm)

        # Copy bead radii from best score model to ensemble models
        if smodels and emodels:
            r = smodels[0].atoms.radii
            for em in emodels:
                em.atoms.radii = r

        return emodels

    # -----------------------------------------------------------------------------
    #
    def read_atomic_models(self, path, mgroup):
        from chimerax.core.atomic import open_mmcif
        models, msg = open_mmcif(self.session, path)

        # Assign IHM model ids.
        if models:
            mnames = self.model_names()
        for i,m in enumerate(models):
            # TODO: Need to read model id from the ihm_model_id field in atom_site table.
            m.display = (i == 0)	# Show only first atomic model
            mid = str(i+1)
            m.ihm_model_ids = [mid]
            m.ihm_group_id = mgroup[mid]
            if mid in mnames:
                m.name = mnames[mid]
        if models:
            self.session.logger.warning('Warning: ihm_model_id in atom_site table currently ignored.  '
                                        'Assuming ihm model ids in atom_site table are 1,2,3,...')
        return models
        
    # -----------------------------------------------------------------------------
    #
    def group_result_models(self, smodels, emodels, amodels, lmaps, model_group):

        group_models = self.make_model_groups()
        group = {g.ihm_group_id:g for g in group_models}

        # Add models to groups.
        for m in smodels + emodels + amodels + lmaps:
            group[m.ihm_group_id].add([m])

        # Warn about groups with missing models.
        smids = set(sum((m.ihm_model_ids for m in smodels+amodels), []))
        for g in group_models:
            missing = [mid for mid in g.ihm_model_ids if mid not in smids]
            if missing:
                msg = ('Missing sphere models (id %s) for group %s (id %s)'
                       % (','.join('%s' % mid for mid in missing), g.name, g.ihm_group_id))
                self.session.logger.info(msg)
        
        # Create results model group
        if group_models:
            if len(group_models) == 1:
                group_models[0].name = 'Result models'
                self.add(group_models)
            else:
                rs_group = Model('Result models', self.session)
                rs_group.add(group_models)
                self.add([rs_group])

        return group_models

    # -----------------------------------------------------------------------------
    #
    def model_id_to_group_id(self):
        mlt = self.tables['ihm_model_list']
        ml_fields = [
            'model_id',
            'model_group_id',]
        ml = mlt.fields(ml_fields)
        mgroup = {mid:gid for mid, gid in ml}
        return mgroup

    # -----------------------------------------------------------------------------
    #
    def make_model_groups(self):
        mlt = self.tables['ihm_model_list']
        ml_fields = [
            'model_id',
            'model_group_id',
            'model_group_name',]
        ml = mlt.fields(ml_fields)
        gm = {}
        for mid, gid, gname in ml:
            gm.setdefault((gid, gname), []).append(mid)
        gmodels = []
        for (gid, gname), mid_list in gm.items():
            g = Model(gname, self.session)
            g.ihm_group_id = gid
            g.ihm_model_ids = mid_list
            gmodels.append(g)

        gmodels.sort(key = lambda g: g.ihm_group_id)

        for g in gmodels[1:]:
            g.display = False	# Only show first group.
            
        self.add(gmodels)
        
        return gmodels
        
    # -----------------------------------------------------------------------------
    #
    def read_crosslinks(self):
        clrt = self.tables['ihm_cross_link_restraint']
        if clrt is None:
            return [], []

        clrt_fields = [
            'asym_id_1',
            'seq_id_1',
            'atom_id_1',
            'asym_id_2',
            'seq_id_2',
            'atom_id_2',
            'restraint_type',
            'distance_threshold'
            ]
        # restraint_type and distance_threshold can be missing
        clrt_rows = clrt.fields(clrt_fields, allow_missing_fields = True)
        xlinks = {}
        for asym_id_1, seq_id_1, atom_id_1, asym_id_2, seq_id_2, atom_id_2, rtype, distance_threshold in clrt_rows:
            d = float(distance_threshold) if distance_threshold is not None else None
            xl = Crosslink(asym_id_1, int(seq_id_1), atom_id_1, asym_id_2, int(seq_id_2), atom_id_2, d)
            xlinks.setdefault(rtype, []).append(xl)

        xlmodels = [CrossLinkModel(self.session, xltype, len(xllist))
                    for xltype, xllist in xlinks.items()]

        return xlinks, xlmodels

    # -----------------------------------------------------------------------------
    #
    def set_initial_sphere_display(self, smodels):
        if len(smodels) == 0:
            return

        # Hide spheres of first model except multi-residue beads and pseudobond endpoints
        # Other parts of structure are depicted using starting models.
        # TODO: Don't hide spheres if there are no starting models.
        smodel = smodels[0]
    
        # Show only multi-residue spheres and crosslink end-point spheres
        satoms = smodel.atoms
        satoms.displays = False
        satoms.filter(satoms.residues.names != '1').displays = True

        # Show pseudobond endpoints
        from chimerax.core.atomic import PseudobondGroup
        pbgs = [pbg for pbg in smodel.child_models() if isinstance(pbg, PseudobondGroup)]
        for pbg in pbgs:
            a1,a2 = pbg.pseudobonds.atoms
            a1.displays = True
            a2.displays = True

    # -----------------------------------------------------------------------------
    #
    def create_result_model_crosslinks(self, xlinks, smodels, emodels, amodels, xlmodels):
        xpbgs = []
        # Create cross links for sphere models
        for smodel in smodels:
            pbgs = make_crosslink_pseudobonds(self.session, xlinks, smodel.residue_sphere,
                                              name = smodel.ihm_group_id,
                                              parent = smodel)
            xpbgs.extend(pbgs)

        if emodels and smodels:
            for emodel in emodels:
                pbgs = make_crosslink_pseudobonds(self.session, xlinks,
                                                  ensemble_sphere_lookup(emodel, smodels[0]),
                                                  parent = emodel)
                xpbgs.extend(pbgs)

        for amodel in amodels:
            # TODO: Ignoring atom specification in crosslink.  Uses principle atom.
            pbgs = make_crosslink_pseudobonds(self.session, xlinks, atom_lookup([amodel]),
                                              radius = 0.2, parent = amodel)
            xpbgs.extend(pbgs)
            
        # TODO: Add crosslinks to result atomic models
        if amodels and xlinks:
            self.session.logger.warning('Crosslinks not yet shown for atomic models.')

        # Allow hiding pseudobond groups for multiple result models.
        for xlm in xlmodels:
            pbgs = [pbg for pbg in xpbgs if pbg.crosslink_type == xlm.crosslink_type]
            xlm.add_pseudobond_models(pbgs)

    # -----------------------------------------------------------------------------
    # Create cross links for starting atomic models.
    #
    def create_starting_model_crosslinks(self, xlinks, amodels, xlmodels):
        if amodels:
            # Starting models may not have disordered regions, so crosslinks will be omitted.
            pbgs = make_crosslink_pseudobonds(self.session, xlinks, atom_lookup(amodels))
            for xlm in xlmodels:
                pbgs = [pbg for pbg in xpbgs if pbg.crosslink_type == xlm.crosslink_type]
                xlm.add_pseudobond_models(pbgs)

    # -----------------------------------------------------------------------------
    #
    def read_2d_electron_microscopy_maps(self):
        emmodels = []
        dot = self.tables['ihm_2dem_class_average_restraint']
        if dot is None:
            return emmodels

        rt = {}	# Orientations of 2D EM for best projection
        caf = self.tables['ihm_2dem_class_average_fitting']
        if caf:
            fields = ['restraint_id', 'model_id',
                      'rot_matrix[1][1]', 'rot_matrix[2][1]', 'rot_matrix[3][1]',
                      'rot_matrix[1][2]', 'rot_matrix[2][2]', 'rot_matrix[3][2]',
                      'rot_matrix[1][3]', 'rot_matrix[2][3]', 'rot_matrix[3][3]',
                      'tr_vector[1]', 'tr_vector[2]', 'tr_vector[3]']
            rows = caf.fields(fields)
            from chimerax.core.geometry import Place
            for rid, mid, r11s,r21s,r31s,r12s,r22s,r32s,r13s,r23s,r33s,t1s,t2s,t3s in rows:
                r11,r21,r31,r12,r22,r32,r13,r23,r33,t1,t2,t3 = \
                    [float(x) for x in (r11s,r21s,r31s,r12s,r22s,r32s,r13s,r23s,r33s,t1s,t2s,t3s)]
#                rt.setdefault(rid,{})[mid] = Place(((r11,r12,r13,t1),(r21,r22,r23,t2),(r31,r32,r33,t3)))
                rt.setdefault(rid,{})[mid] = Place(((r11,r12,r13,t1),(r21,r22,r23,t2),(r31,r32,r33,t3))).inverse()
#                rt.setdefault(rid,{})[mid] = Place(((r11,r21,r31,t1),(r12,r22,r32,t2),(r13,r23,r33,t3)))
            
        fields = ['id', 'dataset_list_id', 'pixel_size_width', 'pixel_size_height']
        rows = dot.fields(fields, allow_missing_fields = True)
        for rid, did, pwidth, pheight in rows:
            d = self.data_set(did, 'ihm_2dem_class_average_restraint')
            if d:
                v = d.volume_model(self.session)
                if v:
                    v.name += ' %dD electron microscopy' % (3 if v.data.size[2] > 1 else 2)
                    v.data.set_step((float(pwidth), float(pheight), v.data.step[2]))
                    if rid in rt:
                        v.ihm_model_projections = rt[rid]
                    v.initialize_thresholds(vfrac = (0.01,1), replace = True)
                    v.show()
                    emmodels.append(v)
        return emmodels

    # -----------------------------------------------------------------------------
    #
    def read_3d_electron_microscopy_maps(self):
        emmodels = []
        dot = self.tables['ihm_3dem_restraint']
        if dot is None:
            return emmodels
        fields = ['dataset_list_id']
        rows = dot.fields(fields, allow_missing_fields = True)
        dfound = set()
        for did, in rows:
            d = self.data_set(did, 'ihm_3dem_restraint')
            if d:
                if d in dfound:
                    # Show one copy of map even if it is used to constrain multiple models (e.g. mediator.cif)
                    continue
                dfound.add(d)
                v = d.volume_model(self.session)
                if v:
                    v.name += ' %dD electron microscopy' % (3 if v.data.size[2] > 1 else 2)
                    v.initialize_thresholds(vfrac = (0.01,1), replace = True)
                    v.show()
                    emmodels.append(v)
        return emmodels

    # -----------------------------------------------------------------------------
    #
    def read_localization_maps(self):

        lmaps = self.read_ensemble_localization_maps()
        if len(lmaps) == 0:
            lmaps = self.read_gaussian_localization_maps()
        return lmaps

    # -----------------------------------------------------------------------------
    #
    def read_ensemble_localization_maps(self, level = 0.2, opacity = 0.5):
        '''Level sets surface threshold so that fraction of mass is outside the surface.'''

        eit = self.tables['ihm_ensemble_info']
        elt = self.tables['ihm_localization_density_files']
        if eit is None or elt is None:
            return []

        ensemble_fields = ['ensemble_id', 'model_group_id', 'num_ensemble_models']
        rows = eit.fields(ensemble_fields)
        ens_group = {id:(gid,int(n)) for id, gid, n in rows}

        loc_fields = ['asym_id', 'ensemble_id', 'file_id']
        loc = elt.fields(loc_fields)
        ens = {}
        for asym_id, ensemble_id, file_id in loc:
            if not ensemble_id in ens_group:
                self.session.logger.warning('Ensemble id "%s" from ihm_localization_density_files table'
                                            'is not present in the ihm_ensemble_info table' % ensemble_id)
                continue
            ens.setdefault(ensemble_id, []).append((asym_id, file_id))

        pmods = []
        from chimerax.core.map.volume import open_map
        from chimerax.core.atomic.colors import chain_rgba
        from os.path import join
        for ensemble_id in sorted(ens.keys()):
            asym_loc = ens[ensemble_id]
            gid, n = ens_group[ensemble_id]
            name = 'Localization map ensemble %s' % ensemble_id
            m = Model(name, self.session)
            m.ihm_group_id = gid
            pmods.append(m)
            for asym_id, file_id in sorted(asym_loc):
                finfo = self.file_info(file_id)
                if finfo is None:
                    continue
                map_path = finfo.path(self.session)
                if map_path is None:
                    # TODO: Warn map file not found.
                    continue
                maps,msg = open_map(self.session, map_path, show = False, show_dialog=False)
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
    def read_gaussian_localization_maps(self, level = 0.2, opacity = 0.5):
        '''Level sets surface threshold so that fraction of mass is outside the surface.'''

        eit = self.tables['ihm_ensemble_info']
        goet = self.tables['ihm_gaussian_obj_ensemble']
        if eit is None or goet is None:
            return []

        ensemble_fields = ['ensemble_id', 'model_group_id', 'num_ensemble_models']
        ens = eit.fields(ensemble_fields)
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
        gauss_rows = goet.fields(gauss_fields)
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
        from chimerax.core.map import volume_from_grid_data
        from chimerax.core.atomic.colors import chain_rgba

        for ensemble_id in sorted(cov.keys()):
            asym_gaussians = cov[ensemble_id]
            gid, n = ens_group[ensemble_id]
            m = Model('Localization map ensemble %s of %d models' % (ensemble_id, n), self.session)
            m.ihm_group_id = gid
            pmods.append(m)
            for asym_id in sorted(asym_gaussians.keys()):
                g = probability_grid(asym_gaussians[asym_id])
                g.name = '%s Gaussians' % asym_id
                g.rgba = chain_rgba(asym_id)[:3] + (opacity,)
                v = volume_from_grid_data(g, self.session, show_data = False,
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
    @property
    def description(self):
        # Report what was read in
        nc = len([m for m in self.starting_models if m.comparative_model])
        nx = len([m for m in self.starting_models if not m.comparative_model])
        nsa = len(self.sequence_alignment_models)
        nt = sum([len(sqm.template_models) for sqm in self.sequence_alignment_models], 0)
        nem = len(self.electron_microscopy_models)
        ns = len(self.sphere_models)
        na = len(self.atomic_models)
        nse = len(self.ensemble_sphere_models)
        nl = sum([len(lm.child_models()) for lm in self.localization_models], 0)
        xldesc = ', '.join('%d %s crosslinks' % (len(xls),type)
                           for type,xls in self.crosslink_models.items())
        esizes = ' and '.join('%d'%em.num_coordsets for em in self.ensemble_sphere_models)
        msg = ('Opened IHM file %s\n'
               ' %d xray/nmr models, %d comparative models, %d sequence alignments, %d templates\n'
               ' %s, %d electron microscopy images\n'
               ' %d atomic models, %d sphere models, %d ensembles with %s models, %d localization maps' %
               (self.filename, nx, nc, nsa, nt, xldesc, nem, na, ns, nse, esizes, nl))
        return msg


# -----------------------------------------------------------------------------
#
class FileInfo:
    def __init__(self, file_id, ref, file_path, ihm_dir):
        self.file_id = file_id
        self.ref = ref		# ExternalReference object or None
        self.file_path = file_path
        self.ihm_dir = ihm_dir

    def stream(self, session, mode = 'r', uncompress = False):
        r = self.ref
        if r is None:
            # Local file
            path = join(self.ihm_dir, self.file_path)
            if uncompress and path.endswith('.gz'):
                import gzip
                f = gzip.open(path, mode)
            else:
                f = open(path, mode)
        elif r.ref_type == 'DOI':
            if r.content == 'Archive':
                from .doi_fetch import fetch_doi_archive_file
                f = fetch_doi_archive_file(session, r.ref, r.url, self.file_path)
                # TODO: Handle gzip decompression of archive files.
            elif r.content == 'File':
                from .doi_fetch import fetch_doi
                path = fetch_doi(session, r.ref, r.url)
                if uncompress and path.endswith('.gz'):
                    import gzip
                    f = gzip.open(path, mode)
                else:
                    f = open(path, mode)
            else:
                f = None
        else:
            f = None
        return f

    @property
    def file_name(self):
        if self.file_path is not None:
            from os.path import basename
            fname = basename(self.file_path)
        elif self.ref and self.ref.url:
            from os.path import basename
            fname = basename(self.ref.url)
        else:
            fname = None
        return fname

    # -----------------------------------------------------------------------------
    #
    def path(self, session):
        if self.file_path:
            from os.path import join, isfile
            path = join(self.ihm_dir, self.file_path)
            if isfile(path):
                return path
            
        r = self.ref
        if r and r.ref_type == 'DOI':
            if r.content == 'Archive':
                from .doi_fetch import unzip_archive
                unzip_archive(session, r.ref, r.url, self.ihm_dir)
                if not isfile(path):
                    session.logger.warning('Failed to find map file in zip archive DOI "%s", url "%s", path "%s"'
                                           % (r.ref, r.url, path))
                    path = None
            elif r.content == 'File':
                from .doi_fetch import fetch_doi
                path = fetch_doi(session, r.ref, r.url)
            else:
                path = None

        return path

# -----------------------------------------------------------------------------
#
class ExternalReference:
    def __init__(self, ref_id, ref_type, ref, content, url):
        self.ref_id = ref_id
        self.ref_type = ref_type	# "DOI"
        self.ref = ref 			# DOI identifier
        self.content = content		# "Archive" or "File"
        self.url = url			# URL to zip archive for a DOI, or file

# -----------------------------------------------------------------------------
#
class DataSet:
    def __init__(self, name):
        self.name = name
    def models(self, session):
        return []
    def volume_model(self, session):
        return None
    
# -----------------------------------------------------------------------------
#
class FileDataSet(DataSet):
    def __init__(self, file_info):
        DataSet.__init__(self, file_info.file_name)
        self.file_info = file_info
    def models(self, session):
        # TODO: use data set instead of file_id.
        finfo = self.file_info
        open_model = atomic_model_reader(finfo.file_path)
        if open_model:
            fs = finfo.stream(session)
            models, msg = open_model(session, fs, finfo.file_name, auto_style = False)
            fs.close()
        else:
            models = []	# Don't know how to read atomic model file
        return models
    def volume_model(self, session):
        finfo = self.file_info
        filename = finfo.file_name
        image_path = finfo.path(session)
        if image_path:
            from chimerax.core.map.volume import open_map
            from chimerax.core.map.data import Unknown_File_Type
            try:
                maps,msg = open_map(session, image_path)
            except Unknown_File_Type:
                return None
            v = maps[0]
            return v
        return None
    
# -----------------------------------------------------------------------------
#
class DatabaseDataSet(DataSet):
    def __init__(self, db_name, db_code):
        DataSet.__init__(self, db_code)
        self.db_name = db_name
        self.db_code = db_code
    def models(self, session):
        if self.db_name == 'PDB' and self.db_code != '?':
            from chimerax.core.atomic.mmcif import fetch_mmcif
            models, msg = fetch_mmcif(session, self.db_code, auto_style = False)
        else:
            models = []
        return models
    def volume_model(self, session):
        if self.db_name == 'EMDB' and self.db_code != '?':
            from chimerax.core.map.emdb_fetch import fetch_emdb
            models, status = fetch_emdb(session, self.db_code)
            return models[0]
        return None

# -----------------------------------------------------------------------------
#
class Crosslink:
    def __init__(self, asym1, seq1, atom1, asym2, seq2, atom2, dist):
        self.asym1 = asym1	# Chain id
        self.seq1 = seq1	# Residue number, integer
        self.atom1 = atom1 	# Atom name, can be None
        self.asym2 = asym2
        self.seq2 = seq2
        self.atom2 = atom2
        self.distance = dist

# -----------------------------------------------------------------------------
# Crosslink model controls display of pseudobond groups but does not display
# anything itself.  The controlled pseudobond groups are not generally child models.
#
class CrossLinkModel(Model):
    def __init__(self, session, crosslink_type, count):
        name = '%d %s crosslinks' % (count, crosslink_type)
        Model.__init__(self, name, session)
        self.crosslink_type = crosslink_type if crosslink_type else ''
        self._pseudobond_groups = []

    def add_pseudobond_models(self, pbgs):
        self._pseudobond_groups.extend(pbgs)
        
    def _get_display(self):
        for pbg in self._pseudobond_groups:
            if pbg.display:
                return True
        return False
    def _set_display(self, display):
        for pbg in self._pseudobond_groups:
            pbg.display = display
    display = property(_get_display, _set_display)

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
        g.crosslink_type = type
        pbgs.append(g)
        missing = []
        apairs = {}
        for xl in xlist:
            a1 = atom_lookup(xl.asym1, xl.seq1, xl.atom1)
            a2 = atom_lookup(xl.asym2, xl.seq2, xl.atom2)
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
            smiss = ','.join('/%s:%d' % (asym_id, seq_num) for asym_id, seq_num in missing[:3])
            if len(missing) > 3:
                smiss += '...'
            msg = 'Missing %d %s crosslink residues %s' % (len(missing), type, smiss)
            if parent is not None and hasattr(parent, 'name'):
                msg = parent.name + ' ' + msg
            session.logger.info(msg)
                
    return pbgs

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
class TemplateModel(Model):
    def __init__(self, session,
                 asym_id, seq_begin, seq_end,
                 template_asym_id, template_seq_begin, template_seq_end,
                 data_set):
        name = 'Template %s %s' % (data_set.name, template_asym_id)
        Model.__init__(self, name, session)
        self.asym_id = asym_id
        self.seq_begin, self.seq_end = seq_begin, seq_end
        self.template_asym_id = template_asym_id
        self.template_seq_begin, self.template_seq_end = template_seq_begin, template_seq_end
        self.data_set = data_set    		# Template model database reference or file
        self.sequence_alignment_model = None
        
    def _get_display(self):
        return False
    def _set_display(self, display):
        if display:
            self.fetch_model()
    display = property(_get_display, _set_display)

    def fetch_model(self):
        if hasattr(self.data_set, 'db_code'):
            # TODO: This is a guess at the name of the template in the sequence alignment.
            sa_name = '%s%s' % (self.data_set.db_code.lower(), self.template_asym_id)
        else:
            sa_name = None
            
        models = self.data_set.models(self.session)
        for i,m in enumerate(models):
            m.name = 'Template %s %s' % (m.name, self.template_asym_id)
            m.sequence_alignment_name = sa_name
            m.asym_id = self.asym_id
            keep_one_chain(m, self.template_asym_id)
            show_colored_ribbon(m, self.asym_id, color_offset = 80)
            if i == 0:
                m.id = self.id

        # Replace TemplateModel with AtomicStructure
        p = self.parent
        self.session.models.remove([self])
        p.add(models)

        sam = self.sequence_alignment_model
        if sam:
            sam.associate_structures(models)
            sam.align_structures(models)
        
# -----------------------------------------------------------------------------
#
class SequenceAlignmentModel(Model):
    def __init__(self, session, alignment_file_info, asym_id, dataset_id):
        self.alignment_file_info = alignment_file_info	# FileInfo
        self.asym_id = asym_id			# IHM asym_id
        self.dataset_id = dataset_id		# Comparative model id
        self.template_models = []		# Filled in after templates fetched.
        self.comparative_model = None
        self.alignment = None
        Model.__init__(self, 'Alignment ' + alignment_file_info.file_name, session)
        self.display = False

    def add_template_model(self, model):
        self.template_models.append(model)
        model.sequence_alignment_model = self
        
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
            self.show_alignment()
        elif a:
            for v in a.viewers:
                v.display(False)
    display = property(_get_display, _set_display)

    def show_alignment(self):
        a = self.alignment
        if a is None:
            fi = self.alignment_file_info
            astream = fi.stream(self.session)
            from chimerax.core.io import deduce_format
            fmt = deduce_format(fi.file_name, no_raise=True)[0]
            if fmt is None:
                print ('Unknown alignment file suffix', fi.file_name)
                return None
            from chimerax.seqalign.parse import open_file
            a = open_file(self.session, astream, fi.file_name, format_name = fmt.name,
                          auto_associate=False, return_vals='alignments')[0]
            self.alignment = a
            self.associate_structures(self.template_models)
            cm = self.comparative_model
            if cm:
                a.associate(cm.chains[0], a.seqs[-1], force = True)
        else:
            for v in a.viewers:
                v.display(True)
        return a

    def associate_structures(self, models):
        # Associate templates with sequences in alignment.
        a = self.alignment
        if a is None:
            a = self.show_alignment()
            if a is None:
                return
        from chimerax.core.atomic import AtomicStructure
        tmap = {tm.sequence_alignment_name : tm for tm in models if isinstance(tm, AtomicStructure)}
        if tmap:
            for seq in a.seqs:
                tm = tmap.get(seq.name)
                if tm:
                    a.associate(tm.chains[0], seq, force = True)
                    tm._associated_sequence = seq

    def align_structures(self, models):
        a = self.alignment
        if a is None:
            a = self.show_alignment()
        cm = self.comparative_model
        if a and cm:
            for m in models:
                if m._associated_sequence:
                    results = a.match(cm.chains[0], [m.chains[0]], iterate=None)
                    if results:
                        # Show only matched residues
                        # TODO: Might show full interval of residues with unused
                        #       insertions colored gray
                        matoms = results[0][0]
                        m.residues.ribbon_displays = False
                        matoms.unique_residues.ribbon_displays = True
                
# -----------------------------------------------------------------------------
#
def atomic_model_reader(filename):
    if filename.endswith('.cif'):
        from chimerax.core.atomic.mmcif import open_mmcif
        return open_mmcif
    elif filename.endswith('.pdb'):
        from chimerax.core.atomic.pdb import open_pdb
        return open_pdb
    return None
                
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
    elif dcount == len(atoms):
        print ('No chain %s in %s' % (chain_id, s.name))

# -----------------------------------------------------------------------------
#
def group_template_models(session, template_models, seq_alignment_models, sa_group):
    '''Place template models in groups under their sequence alignment model.'''
    for sam in seq_alignment_models:
        sam.add(sam.template_models)
    tmodels = [tm for tm in template_models if tm.sequence_alignment_model is None]
    if tmodels:
        et_group = Model('extra templates', session)
        et_group.add(tmodels)
        sa_group.add(et_group)

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
def align_starting_models_to_spheres(amodels, smodel):
    if len(amodels) == 0:
        return
    for m in amodels:
        # Align comparative model residue centers to sphere centers
        res = m.residues
        rnums = res.numbers
        rc = res.centers
        mxyz = []
        sxyz = []
        for rn, c in zip(rnums, rc):
            s = smodel.residue_sphere(m.asym_id, rn)
            if s:
                mxyz.append(c)
                sxyz.append(s.coord)
                # TODO: For spheres with multiple residues use average residue center
        if len(mxyz) >= 3:
            from chimerax.core.geometry import align_points
            from numpy import array, float64
            p, rms = align_points(array(mxyz,float64), array(sxyz,float64))
            m.position = p
            print ('aligned %s, %d residues, rms %.4g' % (m.name, len(mxyz), rms))
        else:
            print ('could not align aligned %s to spheres, %d matching residues' % (m.name, len(mxyz)))
            
# -----------------------------------------------------------------------------
#
def atom_lookup(models):
    amap = {}
    for m in models:
        for a in m.atoms:
            res = a.residue
            amap[(res.chain_id, res.number, a.name)] = a
        for r in m.residues:
            amap[(res.chain_id, res.number, None)] = r.principal_atom
    def lookup(asym_id, res_num, atom_name, amap=amap):
        return amap.get((asym_id, res_num, atom_name))
    return lookup
    
# -----------------------------------------------------------------------------
#
def ensemble_sphere_lookup(emodel, smodel):
    def lookup(asym_id, res_num, atom_name, atoms=emodel.atoms, smodel=smodel):
        a = smodel.residue_sphere(asym_id, res_num)
        return None if a is None else atoms[a.coord_index]
    return lookup

# -----------------------------------------------------------------------------
#
from chimerax.core.atomic import Structure
class SphereModel(Structure):
    def __init__(self, session, name, ihm_model_id, sphere_list):
        Structure.__init__(self, session, name = name, auto_style = False)
        self.ihm_model_ids = [ihm_model_id]
        self.ihm_group_id = None
        
        self._asym_models = {}
        self._sphere_atom = sa = {}	# (asym_id, res_num) -> sphere atom
        
        from chimerax.core.atomic.colors import chain_rgba8
        for (asym_id, sb,se,xyz,r) in sphere_list:
            aname = 'CA'
            a = self.new_atom(aname, 'C')
            a.coord = xyz
            a.radius = r
            a.draw_mode = a.SPHERE_STYLE
            a.color = chain_rgba8(asym_id)
            rname = '%d' % (se-sb+1)
            # Convention on ensemble PDB files is beads get middle residue number of range
            rnum = sb + (sb-se+1)//2
            r = self.new_residue(rname, asym_id, rnum)
            r.add_atom(a)
            for s in range(sb, se+1):
                sa[(asym_id,s)] = a
        self.new_atoms()

    def copy(self, name = None):
        # Copy only the Structure, not the SphereModel
        if name is None:
            name = self.name
        from chimerax.core.atomic.molobject import StructureData
        m = Structure(self.session, name = name, c_pointer = StructureData._copy(self),
                           auto_style = False, log_info = False)
        m.positions = self.positions
        return m
    
    def residue_sphere(self, asym_id, res_num, atom_name=None):
        return self._sphere_atom.get((asym_id,res_num))

    def add_coordinates(self, model_id, sphere_list):
        self.ihm_model_ids.append(model_id)
        from numpy import array, float64
        cxyz = array(tuple(xyz for (asym_id, sb,se,xyz,r) in sphere_list), float64)
        id = len(self.ihm_model_ids)
        self.add_coordset(id, cxyz)
        # TODO: What if sphere radius values differ from one coordinate set to another?
