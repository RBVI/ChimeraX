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

import os.path
import ihm.reader
import ihm.location

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
    SESSION_SAVE = False
    SESSION_WARN = True
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
        self._asym_colors = self._asym_colors8 = None

        Model.__init__(self, name, session)

        # Get python-ihm System object for this file
        self.system = self.read_ihm_system(filename)

        # Starting atomic models, including experimental and comparative structures and templates.
        stmodels, seqmodels = self.read_starting_models(load_linked_files)
        self.starting_models = stmodels
        self.sequence_alignment_models = seqmodels

        # Crosslinks and predicted contacts
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

        # Align 2DEM to projection position for first sphere or atomic model
        if smodels or amodels:
            s0 = smodels[0] if smodels else amodels[0]
            for v in em2d:
                if hasattr(v, 'ihm_model_projections'):
                    p = v.ihm_model_projections.get(s0.ihm_model_ids[0])
                    if p:
                        v.position = p
                    else:
                        # No alignment provided for map so hide it.
                        v.display = False
                else:
                    # No alignment provided for map so hide it.
                    v.display = False
                        
        # Add crosslinks to sphere models
        if show_sphere_crosslinks:
            self.create_result_model_crosslinks(xlinks, smodels, emodels, amodels, xlmodels)
        if show_atom_crosslinks:
            self.create_starting_model_crosslinks(xlinks, stmodels, xlmodels)

        # Show only spheres and atoms that have crosslink restraints
        from chimerax.atomic import AtomicStructure
        hidden_asyms = set(m.asym_id for m in stmodels
                           if isinstance(m, AtomicStructure) and hasattr(m, 'asym_id'))
        self.set_initial_atom_display(smodels, amodels, hidden_asyms)
    
        # Align starting models to first sphere model
        if smodels:
            align_starting_models_to_spheres(stmodels, smodels[0])
        elif amodels:
            align_starting_models_to_atoms(stmodels, amodels[0])
    
        # Ensemble localization
        self.localization_models = lmaps = self.read_localization_maps()

        # Put sphere, ensemble, atomic models and localization maps into parent group models.
        self.group_result_models(smodels, emodels, amodels, lmaps, mgroup)

    def read_ihm_system(self, filename):
        with open(filename) as fh:
            # If multiple data blocks in the file, return just the first one.
            # We also don't use starting model coordinates in the mmCIF file,
            # so don't have the reader read them and waste time & memory.
            return ihm.reader.read(fh, read_starting_model_coord=False,
                                   reject_old_file=True)[0]

    # -----------------------------------------------------------------------------
    #
    def added_to_session(self, session):
        super().added_to_session(session)

        # Write into log table of entities and chains in result model
        self.log_entity_table()
        
    # -----------------------------------------------------------------------------
    #
    def log_entity_table(self):
        if not hasattr(self, 'results_model'):
            return

        # Group chains by entity name.
        anames = self.asym_entity_names()
        ea = {}
        for asym_id, edesc in anames.items():
            if asym_id not in ('.', '?'):
                ea.setdefault(edesc,[]).append(asym_id)

        # Create html table of entities with chains for each entity.
        rid = self.results_model.id_string
        from chimerax.core.logger import html_table_params
        summary = '\n<table %s>\n' % html_table_params
        summary += '  <thead>\n'
        summary += '    <tr>\n'
        summary += '      <th colspan="2">Entities and chains for %s</th>\n' % self.name
        summary += '    </tr>\n'
        summary += '    <tr>\n'
        summary += '      <th>Entity</th>\n'
        summary += '      <th>Chains</th>\n'
        summary += '    </tr>\n'
        summary += '  </thead>\n'
        summary += '  <tbody>\n'
        edescs = sorted(ea.keys())
        for edesc in edescs:
            asym_ids = ea[edesc]
            summary += '    <tr>\n'
            summary += '      <td>'
            elink = '<a title="Select entity" href="cxcmd:select #%s/%s">%s</a>' % (
                rid, ','.join(asym_ids), edesc)
            summary += elink
            summary += '      </td>'
            summary += '      <td style="text-align:center">'
            asym_id_links = ['<a title="Select chain" href="cxcmd:select #%s/%s">%s</a>'
                             % (rid, asym_id, asym_id) for asym_id in asym_ids]
            summary += ', '.join(asym_id_links)
            summary += '      </td>'
            summary += '    </tr>\n'
        summary += '  </tbody>\n'
        summary += '</table>'
        self.session.logger.info(summary, is_html=True)
        
    # -----------------------------------------------------------------------------
    #
    def asym_entity_names(self):
        return {asym._id : asym.entity.description
                for asym in self.system.asym_units}

    # -----------------------------------------------------------------------------
    #
    def asym_colors(self):
        '''
        Use the standard ChimeraX chain color based on chain id, but color chains
        with the same entity using the same color as the first chain id for that entity.
        '''
        if self._asym_colors is not None:
            return self._asym_colors
        
        easyms = {}
        for asym_id, ename in self.asym_entity_names().items():
            easyms.setdefault(ename,[]).append(asym_id)

        self._asym_colors = asym_colors = {}
        from chimerax.atomic.colors import chain_rgba
        for asym_ids in easyms.values():
            color = chain_rgba(asym_ids[0])
            for asym_id in asym_ids:
                asym_colors[asym_id] = color

        return asym_colors

    # -----------------------------------------------------------------------------
    #
    def asym_colors8(self):
        if self._asym_colors8 is not None:
            return self._asym_colors8
        from chimerax.core.colors import rgba_to_rgba8
        self._asym_colors8 = {asym_id:rgba_to_rgba8(color)
                              for asym_id, color in self.asym_colors().items()}
        return self._asym_colors8

    # -----------------------------------------------------------------------------
    #
    def asym_detail_text(self):
        return {asym._id : asym.details
                for asym in self.system.asym_units}

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
            anames = self.asym_entity_names()
            for asym_id in sorted(sma.keys()):
                am = sma[asym_id]
                name = '%s %s' % (anames[asym_id], asym_id)
                a_group = Model(name, self.session)
                a_group.add(am)
                smg.append(a_group)
                a_group.color = am[0].model_color	# Group color is first model color
            sm_group.add(smg)
            self.add([sm_group])

        return xmodels+cmodels, seqmodels

    # -----------------------------------------------------------------------------
    #
    def read_experimental_models(self):
        '''Read crystallography and NMR atomic starting models.'''
        xmodels = []

        asym_colors = self.asym_colors8()

        for sm in self.system.orphan_starting_models:
            if not isinstance(sm.dataset, ihm.dataset.PDBDataset):
                continue
            d = self.data_set(sm.dataset)
            models = d.models(self.session)
            for m in models:
                keep_one_chain(m, sm.asym_id)
                m.name += ' ' + sm.asym_id
                show_colored_ribbon(m, asym_colors.get(sm.asym_unit._id))
            xmodels.extend(models)
            for m in models:
                m.asym_id = sm.asym_unit._id
                m.seq_begin, m.seq_end = sm.asym_unit.seq_id_range
                m.dataset_id = sm.dataset._id
                m.comparative_model = False

        return xmodels
    
    # -----------------------------------------------------------------------------
    #
    def data_set(self, ihm_dataset):
        ds = self._data_sets
        if ds is None:
            self._data_sets = ds = {}
            for d in self.system.orphan_datasets:
                if isinstance(d.location, ihm.location.DatabaseLocation):
                    ds[d._id] = DatabaseDataSet(d.location.db_name,
                                                d.location.access_code)
                elif d.location is not None:
                    finfo = self.file_info(d.location)
                    if finfo:
                        ds[d._id] = FileDataSet(finfo)
                    else:
                        raise ValueError('bad data set id')
        return ds.get(ihm_dataset._id, None)
    
    # -----------------------------------------------------------------------------
    #
    def read_comparative_models(self):
        '''Read comparative models from the ihm_starting_model_details table'''
        lmodels = []

        smfound = set()
        asym_colors = self.asym_colors8()

        for sm in self.system.orphan_starting_models:
            if not isinstance(sm.dataset, ihm.dataset.ComparativeModelDataset):
                continue
            d = self.data_set(sm.dataset)
            if d is None:
                continue
            if sm._id in smfound:
                continue
            smfound.add(sm._id)
            models = d.models(self.session)
            for m in models:
                keep_one_chain(m, sm.asym_id)
                m.name += ' ' + sm.asym_id
                m.dataset_id = sm.dataset._id
                m.asym_id = sm.asym_unit._id
                m.comparative_model = True
                show_colored_ribbon(m, asym_colors.get(sm.asym_unit._id))
            lmodels.extend(models)
      
        return lmodels

    # -----------------------------------------------------------------------------
    #
    def read_template_models(self):
        '''Read crystallography and NMR atomic starting models.'''
        from collections import OrderedDict
        tmodels = []
        seqmodels = []
        alignments = OrderedDict()  # Sequence alignments for comparative models
        asym_colors = self.asym_colors8()

        # Get info about comparative models.
        for sm in self.system.orphan_starting_models:
            if not isinstance(sm.dataset, ihm.dataset.ComparativeModelDataset):
                continue
            asym_id = sm.asym_unit._id
            seq_beg, seq_end = sm.asym_unit.seq_id_range
            # Get info about templates
            for t in sm.templates:
                d = self.data_set(t.dataset)
                if d is None:
                    continue
                # Template for a comparative model.
                tseq_beg, tseq_end = t.template_seq_id_range
                tm = TemplateModel(self.session, asym_id, seq_beg, seq_end,
                                   t.asym_id, tseq_beg, tseq_end, d)
                tm.base_color = asym_colors[asym_id]
                tmodels.append(tm)
                if t.alignment_file:
                    sfinfo = self.file_info(t.alignment_file)
                    if sfinfo is not None:
                        a = (sfinfo, asym_id, sm.dataset._id)
                        sam = alignments.get(a)
                        if sam is None:
                            # Make sequence alignment model for
                            # comparative model
                            alignments[a] = sam = SequenceAlignmentModel(
                                        self.session, sfinfo, asym_id,
                                        sm.dataset._id)
                        sam.add_template_model(tm)

        seqmodels = list(alignments.values())
        return tmodels, seqmodels

    # -----------------------------------------------------------------------------
    #
    def file_info(self, ihm_location):
        fmap = self._file_info
        if fmap is None:
            refs = {}
            self._file_info = fmap = {}
            for loc in self.system._all_locations():
                if not isinstance(loc, ihm.location.FileLocation):
                    continue
                repo = loc.repo
                if repo and repo._id not in refs:
                    refs[repo._id] = ExternalReference(repo._id,
                                            repo.reference_type,
                                            repo.reference,
                                            repo.refers_to,
                                            repo.url)
                fmap[loc._id] = FileInfo(loc._id,
                        refs[repo._id] if repo else None,
                        loc.path, self.ihm_directory)
        fi = fmap.get(ihm_location._id, None)
        return fi

    # -----------------------------------------------------------------------------
    #
    def model_names(self):
        # Work around python-ihm issue #42
        return {m._id: (m.name if m.name and m.name != ihm.unknown
                        else 'result %s' % m._id)
                for mg in self.all_model_groups() for m in mg}

    # -----------------------------------------------------------------------------
    #
    def make_sphere_models(self, model_group, group_coordsets = True, load_ensembles = False):
        # todo: this is really inefficient since it duplicates Python objects.
        # Better would be to override ihm.model.Model to populate ChimeraX
        # data structures directly, or use the Sphere objects unmodified.
        mnames = self.model_names()
        mspheres = {}

        for mg in self.all_model_groups():
            for m in mg:
                for s in m._spheres:
                    sb, se = s.seq_id_range
                    xyz = s.x, s.y, s.z
                    mspheres.setdefault(m._id, []).append(
                            (s.asym_unit._id,sb,se,xyz,s.radius))
        smodels = self.make_sphere_models_by_group(mspheres, mnames,
                                      model_group, group_coordsets)

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

        # Associate entity names with asym ids.
        anames = self.asym_entity_names()
        adetail = self.asym_detail_text()
        asym_colors = self.asym_colors8()

        smodels = []
        for g in gs:
            ms = msg[g]
            if group_coordsets and self.same_sphere_atoms(ms):
                # For groups with matching residue / atom names use coordinate set.
                mid, slist = ms[0]
                mname = mnames.get(mid, 'sphere model')
                sm = SphereModel(self.session, mname, mid, slist, anames, adetail, asym_colors)
                sm.ihm_group_id = g
                for mid, slist in ms[1:]:
                    sm.add_coordinates(mid, slist)
                smodels.append(sm)
                if len(ms) > 1:
                    sm.name = '%d models' % len(ms)
            else:
                # Make separate sphere models, do not use coordinate sets.
                for i, (mid, slist) in enumerate(ms):
                    mname = mnames.get(mid, 'sphere model')
                    sm = SphereModel(self.session, mname, mid, slist, anames, adetail, asym_colors)
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
        def _all_samples():  # iterate over all ensembles and subsamples
            for ensemble in self.system.ensembles:
                mg = ensemble.model_group
                yield ensemble, mg
                for subsample in ensemble.subsamples:
                    yield subsample, subsample.model_group or mg
        emodels = []
        for sample, model_group in _all_samples():
            # Can't match without a model group ID
            if model_group is None:
                continue
            gid = model_group._id
            if sample.file is None:
                continue
            finfo = self.file_info(sample.file)
            if finfo is None:
                continue
            fname = finfo.file_name
#            print("looked up", sample.file, "got", fname)
            if fname.endswith('.dcd'):
                gsm = [sm for sm in smodels if sm.ihm_group_id == gid]
                if len(gsm) != 1:
                    continue  # Don't have exactly one sphere model for this group id
                sm = gsm[0].copy(name = sample.name)
                dcd_path = finfo.path(self.session)
                from chimerax.md_crds.read_coords import read_coords
                read_coords(self.session, dcd_path, sm, format_name = 'dcd', replace=True)
                sm.active_coordset_id = 1
            elif fname.endswith('.pdb') or fname.endswith('.pdb.gz'):
                fstream = finfo.stream(self.session, uncompress = True)
                if fstream is None:
                    continue
                from chimerax.pdb import open_pdb
                mlist,msg = open_pdb(self.session, fstream, sample.name,
                                     auto_style = False, coordsets = True)
                sm = mlist[0]
            sm.ihm_group_id = gid
            sm.display = False
            sm.name += ' %d models' % sm.num_coordsets
            sm.ss_assigned = True	# Don't assign secondary structure to sphere model
            atoms = sm.atoms
            from chimerax.atomic.colors import chain_colors
            atoms.colors = chain_colors(atoms.residues.chain_ids)
            emodels.append(sm)
            
        self.add_entity_names(emodels)
        self.set_asym_colors(emodels)
        self.copy_sphere_radii(emodels, smodels)

        return emodels

    # -----------------------------------------------------------------------------
    #
    def add_entity_names(self, models):
        if len(models) == 0:
            return
        
        entity_names = self.asym_entity_names()
        asym_details = self.asym_detail_text()
        for m in models:
            for r in m.residues:
                asym_id = r.chain_id
                r.entity_name = entity_names.get(asym_id, '?')
                r.asym_detail = asym_details.get(asym_id, '')


    # -----------------------------------------------------------------------------
    #
    def set_asym_colors(self, models):
        if len(models) == 0:
            return
        
        asym_colors = self.asym_colors8()
        for m in models:
            for s, asym_id, atoms in m.atoms.by_chain:
                atoms.colors = asym_colors.get(asym_id, (200,200,200,255))

    # -----------------------------------------------------------------------------
    #
    def copy_sphere_radii(self, emodels, smodels):
        # Copy bead radii from best score model to ensemble models
        for em in emodels:
            esm = [sm for sm in smodels if sm.ihm_group_id == em.ihm_group_id]
            if len(esm) == 1:
                sm = esm[0]
                if sm.num_atoms == em.num_atoms:
                    em.atoms.radii = sm.atoms.radii

    # -----------------------------------------------------------------------------
    #
    def read_atomic_models(self, path, mgroup):
        from chimerax.mmcif import open_mmcif
        models, msg = open_mmcif(self.session, path, auto_style = False)

        # Assign IHM model ids.
        if models:
            mnames = self.model_names()
        group_ids = set()
        for i,m in enumerate(models):
            # TODO: Need to read model id from the ihm_model_id field in atom_site table.
            mid = str(i+1)
            m.ihm_model_ids = [mid]
            gid = mgroup[mid]
            m.ihm_group_id = gid
            if mid in mnames:
                m.name = mnames[mid]
            m.display = (gid not in group_ids)	# Show only first atomic model in each group
            group_ids.add(gid)
            m.apply_auto_styling(self.session)
            
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
                self.results_model = gm0 = group_models[0]
                gm0.name = 'Result models'
                self.add(group_models)
            else:
                rs_group = Model('Result models', self.session)
                rs_group.add(group_models)
                self.results_model = rs_group
                self.add([rs_group])

        return group_models

    # -----------------------------------------------------------------------------
    #
    def all_model_groups(self):
        for state_group in self.system.state_groups:
            for state in state_group:
                for model_group in state:
                    yield model_group

    def model_id_to_group_id(self):
        return {model._id: model_group._id
                for model_group in self.all_model_groups()
                for model in model_group}

    # -----------------------------------------------------------------------------
    #
    def make_model_groups(self):
        gmodels = []
        for mg in self.all_model_groups():
            g = Model(mg.name if mg.name else 'Group ' + mg._id,
                      self.session)
            g.ihm_group_id = mg._id
            g.ihm_model_ids = [m._id for m in mg]
            gmodels.append(g)

        gmodels.sort(key = lambda g: g.ihm_group_id)

        for g in gmodels[1:]:
            g.display = False	# Only show first group.
            
        self.add(gmodels)
        
        return gmodels
        
    # -----------------------------------------------------------------------------
    #
    def read_crosslinks(self):
        xlinks = {}
        for xlrestraint in self.system.restraints:
            if not isinstance(xlrestraint, ihm.restraint.CrossLinkRestraint):
                continue

            ct = xlrestraint.linker.auth_name or ''
            for x in xlrestraint.cross_links:
                exx = x.experimental_cross_link
                xl = Crosslink(x.asym1._id, exx.residue1.seq_id, x.atom1,
                               x.asym2._id, exx.residue2.seq_id, x.atom2,
                               x.distance.distance,
                               x.distance.distance_lower_limit)
                xlinks.setdefault(ct, []).append(xl)

        pc = self.read_predicted_contacts()
        if pc:
            xlinks['predicted contacts'] = pc
        
        xlmodels = [CrossLinkModel(self.session, xltype, len(xllist))
                    for xltype, xllist in xlinks.items()]

        return xlinks, xlmodels

    # -----------------------------------------------------------------------------
    #
    def read_predicted_contacts(self):
        xlinks = []
        for x in self.system.restraints:
            if not isinstance(x, ihm.restraint.PredictedContactRestraint):
                continue
            xl = Crosslink(x.resatom1.asym._id, x.resatom1.seq_id,
                           x.resatom1.id, x.resatom2.asym._id,
                           x.resatom2.seq_id, x.resatom2.id,
                           x.distance.distance,
                           x.distance.distance_lower_limit)
            xlinks.append(xl)

        return xlinks
    
    # -----------------------------------------------------------------------------
    #
    def set_initial_atom_display(self, smodels, amodels, hidden_asym_ids):
        if smodels:
            # Hide spheres of first model except multi-residue beads and pseudobond endpoints
            # Other parts of structure are depicted using starting models (in hidden_asym_ids).
            smodel = smodels[0]

            # Show only multi-residue spheres
            for m, cid, satoms in smodel.atoms.by_chain:
                if cid in hidden_asym_ids:
                    satoms.displays = False
                    satoms.filter(satoms.residues.names != '1').displays = True

        # Show crosslink end-point atoms and spheres
        from chimerax.atomic import PseudobondGroup
        pbgs = sum([[pbg for pbg in m.child_models()
                     if isinstance(pbg, PseudobondGroup) and pbg.name != 'missing structure']
                    for m in smodels[:1] + amodels], [])
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

        for emodel in emodels:
            esm = [sm for sm in smodels if sm.ihm_group_id == emodel.ihm_group_id]
            if len(esm) == 1:
                pbgs = make_crosslink_pseudobonds(self.session, xlinks,
                                                  ensemble_sphere_lookup(emodel, esm[0]),
                                                  parent = emodel)
                xpbgs.extend(pbgs)

        for amodel in amodels:
            # TODO: Ignoring atom specification in crosslink.  Uses principle atom.
            pbgs = make_crosslink_pseudobonds(self.session, xlinks, atom_lookup([amodel]),
                                              radius = 0.2, parent = amodel)
            xpbgs.extend(pbgs)

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
        rt = {}  # Orientations of 2D EM for best projection
        for r in self.system.restraints:
            if not isinstance(r, ihm.restraint.EM2DRestraint):
                continue

            for model, fit in r.fits.items():
                rm = fit.rot_matrix
                t = fit.tr_vector
                from chimerax.geometry import Place
                rt.setdefault(r._id,{})[model._id] = Place(((rm[0][0],rm[0][1],rm[0][2],t[0]),(rm[1][0],rm[1][1],rm[1][2],t[1]),(rm[2][0],rm[2][1],rm[2][2],t[2]))).inverse()
            
            d = self.data_set(r.dataset)
            if d:
                v = d.volume_model(self.session)
                if v:
                    v.name += ' %dD electron microscopy' % (3 if v.data.size[2] > 1 else 2)
                    v.data.set_step((r.pixel_size_width, r.pixel_size_height,
                                     v.data.step[2]))
                    if r._id in rt:
                        v.ihm_model_projections = rt[r._id]
                    v.set_display_style('image')
                    v.show()
                    emmodels.append(v)
        return emmodels

    # -----------------------------------------------------------------------------
    #
    def read_3d_electron_microscopy_maps(self):
        def get_parent_volume(restraint, dfound):
            for p in restraint.dataset.parents:
                d = self.data_set(p)
                if d not in dfound:
                    dfound.add(d)
                    v = d.volume_model(self.session)
                    if v:
                        return v
        emmodels = []
        dfound = set()
        for r in self.system.restraints:
            if not isinstance(r, ihm.restraint.EM3DRestraint):
                continue
            d = self.data_set(r.dataset)
            if d:
                if d in dfound:
                    # Show one copy of map even if it is used to constrain multiple models (e.g. mediator.cif)
                    continue
                dfound.add(d)
                v = d.volume_model(self.session)
                # If we can't visualize the dataset, see if we can visualize
                # one of its parents (e.g. a GMM may be derived from an MRC
                # file from EMDB)
                if v is None:
                    v = get_parent_volume(r, dfound)
                if v:
                    v.name += ' %dD electron microscopy' % (3 if v.data.size[2] > 1 else 2)
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
        pmods = []
        asym_colors = self.asym_colors()
        from chimerax.map.volume import open_map

        for e in self.system.ensembles:
            name = 'Localization map ensemble %s' % e._id
            m = None
            for loc in e.densities:
                finfo = self.file_info(loc.file)
                if finfo is None:
                    continue
                map_path = finfo.path(self.session)
                if map_path is None:
                    self.session.logger.warning('Could not find localization map "%s"'
                                                % finfo.file_path)
                    continue
                maps,msg = open_map(self.session, map_path, show = False, show_dialog=False)
                asym_id = loc.asym_unit._id
                color = asym_colors[asym_id][:3] + (opacity,)
                v = maps[0]
                ms = v.matrix_value_statistics()
                vlev = ms.mass_rank_data_value(level)
                v.set_parameters(surface_levels = [vlev], surface_colors = [color])
                v.show_in_volume_viewer = False
                v.show()
                if m is None:
                    m = Model(name, self.session)
                    m.ihm_group_id = e.model_group._id
                    pmods.append(m)
                m.add([v])
        return pmods

    # -----------------------------------------------------------------------------
    #
    def read_gaussian_localization_maps(self, level = 0.2, opacity = 0.5):
        '''Level sets surface threshold so that fraction of mass is outside the surface.'''

        return [] # not currently handled by python-ihm
        eit = self.tables['ihm_ensemble_info']
        goet = self.tables['ihm_gaussian_obj_ensemble']
        if not eit or not goet:
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
        asym_colors = self.asym_colors()
        from chimerax.map import volume_from_grid_data

        for ensemble_id in sorted(cov.keys()):
            asym_gaussians = cov[ensemble_id]
            gid, n = ens_group[ensemble_id]
            m = Model('Localization map ensemble %s of %d models' % (ensemble_id, n), self.session)
            m.ihm_group_id = gid
            pmods.append(m)
            for asym_id in sorted(asym_gaussians.keys()):
                g = probability_grid(asym_gaussians[asym_id])
                g.name = '%s Gaussians' % asym_id
                g.rgba = asym_colors[asym_id][:3] + (opacity,)
                v = volume_from_grid_data(g, self.session, style = 'surface',
                                          open_model = False, show_dialog = False)
                ms = v.matrix_value_statistics()
                vlev = ms.mass_rank_data_value(level)
                v.set_parameters(surface_levels = [vlev])
                v.show_in_volume_viewer = False
                m.add([v])

        return pmods

    # -----------------------------------------------------------------------------
    #
    @property
    def description(self):
        lines = ['Opened IHM file %s' % self.filename]
        # Report what was read in
        nx = len([m for m in self.starting_models if not m.comparative_model])
        if nx:
            lines.append('%d xray and nmr models' % nx)
        nc = len([m for m in self.starting_models if m.comparative_model])
        if nc:
            lines.append('%d comparative models' % nc)
        nsa = len(self.sequence_alignment_models)
        if nsa:
            lines.append('%d sequence alignments' % nsa)
        nt = sum([len(sqm.template_models) for sqm in self.sequence_alignment_models], 0)
        if nt:
            lines.append('%d templates' % nt)
        xldesc = ', '.join('%d %s crosslinks' % (len(xls),type)
                           for type,xls in self.crosslink_models.items())
        if xldesc:
            lines.append(xldesc)
        nem = len(self.electron_microscopy_models)
        if nem:
            lines.append('%d electron microscopy images' % nem)
        na = len(self.atomic_models)
        if na:
            lines.append('%d atomic models' % na)
        ns = len(self.sphere_models)
        if ns:
            lines.append('%d sphere models' % ns)
        nse = len(self.ensemble_sphere_models)
        if nse:
            esizes = ' and '.join('%d'%em.num_coordsets for em in self.ensemble_sphere_models)
            lines.append('%d ensembles with %s models' % (nse, esizes))
        nl = sum([len(lm.child_models()) for lm in self.localization_models], 0)
        if nl:
            lines.append('%d localization maps' % nl)
        msg = '\n'.join(lines)
        return msg


# -----------------------------------------------------------------------------
#
class FileInfo:
    def __init__(self, file_id, ref, file_path, ihm_dir):
        self.file_id = file_id
        self.ref = ref		# ExternalReference object or None
        self.file_path = file_path
        self.ihm_dir = ihm_dir
        self._warn = True
        # Handle repositories that contain a single file
        if file_path in ('.', None) and ref and ref.url:
            self.file_path = os.path.basename(ref.url)

    def stream(self, session, mode = 'r', uncompress = False):
        r = self.ref
        if r is None or r.ref_type == 'Supplementary Files':
            f = self._open_local_file(session, self.file_path, mode, uncompress)
        elif r.ref_type == 'DOI':
            f = self._open_doi_file(session, self.file_path, mode, uncompress)
        else:
            f = None
            if self._warn:
                session.logger.warning('Unrecognized external file reference_type "%s" for file "%s",'
                                       ' expecting "Supplementary Files" or "DOI"'
                                       % (r.ref_type, self.file_path))
                self._warn = False
        return f

    def _open_local_file(self, session, file_path, mode, uncompress):
        from os.path import join, exists
        path = join(self.ihm_dir, file_path)
        if not exists(path):
            f = None
            if self._warn:
                session.logger.warning('Missing file "%s"' % path)
                self._warn = False
        elif uncompress and path.endswith('.gz'):
            import gzip
            f = gzip.open(path, mode)
        else:
            f = open(path, mode)
        return f

    def _open_doi_file(self, session, file_path, mode, uncompress):
        r = self.ref
        if r._fetch_failed:
            f = None
        elif r.content == 'Archive':
            from .doi_fetch import fetch_doi_archive_file
            from chimerax.core.errors import UserError
            try:
                f = fetch_doi_archive_file(session, r.ref, r.url, file_path)
            except UserError as e:
                session.logger.warning(str(e))
                r._fetch_failed = True
                f = None
            # TODO: Handle gzip decompression of archive files.
        elif r.content == 'File':
            from .doi_fetch import fetch_doi
            from chimerax.core.errors import UserError
            try:
                path = fetch_doi(session, r.ref, r.url)
            except UserError as e:
                session.logger.warning(str(e))
                r._fetch_failed = True
                f = None
            else:
                if uncompress and path.endswith('.gz'):
                    import gzip
                    f = gzip.open(path, mode)
                else:
                    f = open(path, mode)
        else:
            f = None
            if self._warn:
                session.logger.warning('Unrecognized DOI content type "%s" for file "%s",'
                                       ' expecting "Archive" or "File".'
                                       % (r.content, file_path))
                self._warn = False
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
        r = self.ref
        if (r is None or r.ref_type == 'Supplementary Files') and self.file_path:
            from os.path import join, isfile
            path = join(self.ihm_dir, self.file_path)
            if isfile(path):
                return path
            
        if r and r.ref_type == 'DOI':
            if r._fetch_failed:
                path = None
            elif r.content == 'Archive' and self.file_path:
                from .doi_fetch import unzip_archive
                from chimerax.core.errors import UserError
                try:
                    dir = unzip_archive(session, r.ref, r.url)
                except UserError as e:
                    session.logger.warning(str(e))
                    r._fetch_failed = True
                    path = None
                else:
                    from os.path import join, isfile
                    path = join(dir, self.file_path)
                    if not isfile(path):
                        session.logger.warning('Failed to find map file in zip archive'
                                               'DOI "%s", url "%s", path "%s"'
                                               % (r.ref, r.url, path))
                        path = None
            elif r.content == 'File':
                from .doi_fetch import fetch_doi
                from chimerax.core.errors import UserError
                try:
                    path = fetch_doi(session, r.ref, r.url)
                except UserError as e:
                    session.logger.warning(str(e))
                    r._fetch_failed = True
                    path = None
            else:
                path = None
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
        self._fetch_failed = False	# Remember if fetch failed to avoid multiple error messages

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
            if fs:
                models, msg = open_model(session, fs, finfo.file_name, auto_style = False, log_info = False)
                fs.close()
            else:
                models = []
        else:
            models = []	# Don't know how to read atomic model file
        return models
    def volume_model(self, session):
        finfo = self.file_info
        filename = finfo.file_name
        image_path = finfo.path(session)
        if image_path:
            from chimerax.map.volume import open_map
            from chimerax.map_data import UnknownFileType
            try:
                maps,msg = open_map(session, image_path)
            except UnknownFileType:
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
            from chimerax.mmcif import fetch_mmcif
            models, msg = fetch_mmcif(session, self.db_code, auto_style = False, log_info = False)
        else:
            models = []
        return models
    def volume_model(self, session):
        dbc = self.db_code
        if self.db_name == 'EMDB' and dbc != '?':
            dbc = dbc[4:] if dbc.startswith('EMD-') else dbc
            from chimerax.map.emdb_fetch import fetch_emdb
            models, status = fetch_emdb(session, dbc)
            return models[0]
        return None

# -----------------------------------------------------------------------------
#
class Crosslink:
    def __init__(self, asym1, seq1, atom1, asym2, seq2, atom2, dist, dist_low = None):
        # mmCIF asym IDs read by IHM module can be strings or ints, but ChimeraX always
        # uses strings, so coerce to str so that comparisons work
        self.asym1 = str(asym1)	# Chain id
        self.seq1 = seq1	# Residue number, integer
        self.atom1 = atom1 	# Atom name, can be None
        self.asym2 = str(asym2)
        self.seq2 = seq2
        self.atom2 = atom2
        self.distance_upper = dist	# Upper bound, can be None
        self.distance_lower = dist_low	# Lower bound, can be None

    def color(self, length, color, long_color, short_color):
        d, dlow = self.distance_upper, self.distance_lower
        if d is not None and length > d:
            return long_color
        elif dlow is not None and length < dlow:
            return short_color
        return color

# -----------------------------------------------------------------------------
# Crosslink model controls display of pseudobond groups but does not display
# anything itself.  The controlled pseudobond groups are not generally child models.
#
class CrossLinkModel(Model):
    def __init__(self, session, crosslink_type, count):
        name = '%d %s crosslinks' % (count, crosslink_type)
        Model.__init__(self, name, session)
        self.crosslink_type = cc = crosslink_type if crosslink_type else ''
        self.color = crosslink_colors(cc)[0]
        self._pseudobond_groups = []

    def add_pseudobond_models(self, pbgs):
        self._pseudobond_groups.extend(pbgs)

    def _get_display(self):
        for pbg in self._pseudobond_groups:
            if not pbg.deleted and pbg.display:
                return True
        return False
    def _set_display(self, display):
        for pbg in self._pseudobond_groups:
            if not pbg.deleted:
                pbg.display = display
    display = property(_get_display, _set_display)

    def _get_model_color(self):
        return self.color
    def _set_model_color(self, color):
        self.color = color
        for pbg in self._pseudobond_groups:
            if not pbg.deleted:
                pbg.model_color = color
    model_color = property(_get_model_color, _set_model_color)

# -----------------------------------------------------------------------------
#
def make_crosslink_pseudobonds(session, xlinks, atom_lookup,
                               name = None,
                               parent = None,
                               radius = 1.0,
                               color = (0,255,0,255),		# Green
                               long_color = (255,0,0,255),	# Red
                               short_color = (0,0,255,255)):	# Blue
    
    pbgs = []
    new_pbgroup = session.pb_manager.get_group if parent is None else parent.pseudobond_group
    for xltype, xlist in xlinks.items():
        xname = '%d %s crosslinks' % (len(xlist), xltype)
        if name is not None:
            xname += ' ' + name
        g = new_pbgroup(xname)
        g.crosslink_type = xltype
        pbgs.append(g)
        color, long_color, short_color = crosslink_colors(xltype)
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
                b.color = xl.color(b.length, color, long_color, short_color)
                b.radius = radius
                b.halfbond = False
            elif a1 is None:
                missing.append((xl.asym1, xl.seq1))
            elif a2 is None:
                missing.append((xl.asym2, xl.seq2))
        if missing:
            mres = list(set((asym_id, seq_num) for asym_id, seq_num in missing))
            mres.sort()
            smiss = ','.join('/%s:%d' % ai for ai in mres[:5])
            if len(missing) > 5:
                smiss += '...'
            msg = 'Missing residues for %d of %d %s crosslinks: %s' % (len(missing), len(xlist), xltype, smiss)
            if parent is not None and hasattr(parent, 'name'):
                msg = parent.name + ' ' + msg
            session.logger.info(msg)
                
    return pbgs

# -----------------------------------------------------------------------------
#
_crosslink_colors = {}
def crosslink_colors(xltype):
    global _crosslink_colors
    if xltype not in _crosslink_colors:
        colors = [('lightgreen', 'lime', 'lime'),
                  ('lightskyblue', 'deepskyblue', 'deepskyblue'),
                  ('plum', 'magenta', 'magenta'),
                  ('peachpuff', 'peru', 'peru'),
                  ('aquamarine', 'aqua', 'aqua')]
        i = len(_crosslink_colors) % len(colors)
        from chimerax.core.colors import BuiltinColors
        _crosslink_colors[xltype] = tuple(BuiltinColors[c].uint8x4() for c in colors[i])
    return _crosslink_colors[xltype]
    
# -----------------------------------------------------------------------------
#
def probability_grid(wcc, voxel_size = 5, cutoff_sigmas = 3):
    # Find bounding box for probability distribution
    from chimerax.geometry import Bounds, union_bounds
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

    from chimerax.map_data import ArrayGridData
    g = ArrayGridData(a, origin = xyz0, step = vsize)
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

from chimerax.map import covariance_sum
        
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
        self.base_color = (200,200,200,255)
        
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
            show_colored_ribbon(m, self.base_color, color_offset = 80)
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
            from chimerax.data_formats import NoFormatError
            try:
                fmt = session.data_formats.format_from_file_name(fi.file_name)
            except NoFormatError:
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
        from chimerax.atomic import AtomicStructure
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
        from chimerax.mmcif import open_mmcif
        return open_mmcif
    elif filename.endswith('.pdb'):
        from chimerax.pdb import open_pdb
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
def show_colored_ribbon(m, color = None, color_offset = None):
    if color is None:
        from numpy import random, uint8
        color = random.randint(128,255,(4,),uint8)
        color[3] = 255
    else:
        color = list(color)
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
            from chimerax.geometry import align_points
            from numpy import array, float64
            p, rms = align_points(array(mxyz,float64), array(sxyz,float64))
            m.position = p
            print ('aligned %s, %d residues, rms %.4g' % (m.name, len(mxyz), rms))
        else:
            print ('could not align aligned %s to spheres, %d matching residues' % (m.name, len(mxyz)))

# -----------------------------------------------------------------------------
#
def align_starting_models_to_atoms(amodels, refmodel):
    if len(amodels) == 0:
        return

    # TODO: Handle case where model coordinate systems are different
    rloc = {}
    for r in refmodel.residues:
        pa = r.principal_atom
        if pa:
            rloc[(r.chain_id, r.number)] = pa.coord
            
    for m in amodels:
        # Align comparative model atoms to result model atoms
        asym_id = m.asym_id
        mxyz = []
        sxyz = []
        for r in m.residues:
            pa = r.principal_atom
            if pa:
                xyz = rloc.get((asym_id, r.number))
                if xyz is not None:
                    mxyz.append(pa.coord)
                    sxyz.append(xyz)
                else:
                    print ('could not find res for alignment', (r.chain_id, r.number))
        if len(mxyz) >= 3:
            from chimerax.geometry import align_points
            from numpy import array, float64
            p, rms = align_points(array(mxyz,float64), array(sxyz,float64))
            m.position = p
            print ('aligned %s to %s, %d residues, rms %.4g' % (m.name, refmodel.name, len(mxyz), rms))
        else:
            print ('could not align %s to %s, only %d matching residues' % (m.name, refmodel.name, len(mxyz)))
            
# -----------------------------------------------------------------------------
#
def atom_lookup(models):
    amap = {}
    for m in models:
        for a in m.atoms:
            res = a.residue
            amap[(res.chain_id, res.number, a.name)] = a
        for r in m.residues:
            # If multiple residues with same chain ID and number, pick the
            # last one that has a defined principal atom
            if r.principal_atom is not None:
                amap[(r.chain_id, r.number, None)] = r.principal_atom
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
from chimerax.atomic import Structure
class SphereModel(Structure):
    def __init__(self, session, name, ihm_model_id,
                 sphere_list = None, entity_names = {}, asym_detail_text = {}, asym_colors = {},
                 c_pointer = None, auto_style = False, log_info = False):
        Structure.__init__(self, session, name = name,
                           auto_style = auto_style, log_info = log_info,
                           c_pointer = c_pointer)
        self.ihm_model_ids = [ihm_model_id]
        self.ihm_group_id = None
        self._asym_colors = asym_colors
        
        self._asym_models = {}
        self._sphere_atom = {}	# (asym_id, res_num) -> sphere atom

        self._polymers = []	# List of Residues objects for making ribbons

        if sphere_list is not None:
            self._add_spheres(sphere_list, entity_names, asym_detail_text)

    def _add_spheres(self, sphere_list, entity_names, asym_detail_text):

        # Find spheres for each asym_id in residue number order.
        asym_spheres = {}
        for (asym_id, sb,se,xyz,r) in sphere_list:
            asym_spheres.setdefault(asym_id, []).append((sb,se,xyz,r))
        for spheres in asym_spheres.values():
            spheres.sort(key = lambda bexr: bexr[0])

        # Create sphere atoms, residues and connecting pseudobonds
        from chimerax.atomic import colors, Residues
        polymers = []
        pbg = self.pseudobond_group('missing structure')
        sa = self._sphere_atom
        asym_colors = self._asym_colors
        for asym_id, aspheres in asym_spheres.items():
            color = asym_colors.get(asym_id, (200,200,200,255))
            ename = entity_names.get(asym_id, '?')
            adetail = asym_detail_text.get(asym_id, '')
            last_atom = None
            polymer = []
            for (sb,se,xyz,r) in aspheres:
                aname = 'CA'
                a = self.new_atom(aname, 'C')
                a.coord = xyz
                a.radius = r
                a.draw_mode = a.SPHERE_STYLE
                a.color = color
                rname = ''
                # Convention on ensemble PDB files is beads get middle residue number of range
                rnum = sb
                r = self.new_residue(rname, asym_id, rnum)
                r.ribbon_color = color
                r.entity_name = ename
                r.asym_detail = adetail
                r.add_atom(a)
                polymer.append(r)
                for s in range(sb, se+1):
                    sa[(asym_id,s)] = a
                if last_atom:
                    pbg.new_pseudobond(a, last_atom)
                last_atom = a
            polymers.append(Residues(polymer))

        self._polymers.extend(polymers)	# Needed for ribbon rendering

    def copy(self, name = None):
        if name is None:
            name = self.name
        from chimerax.atomic.molobject import StructureData
        m = SphereModel(self.session, name, self.ihm_model_ids[0],
                        c_pointer = StructureData._copy(self))
        m.positions = self.positions
        if self._polymers:
            rmap = dict(zip(self.residues, m.residues))
            from chimerax.atomic import Residues
            m._polymers = [Residues([rmap[r] for r in p]) for p in self._polymers]
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

    def polymers(self, missing_structure_treatment = Structure.PMS_ALWAYS_CONNECTS,
                 consider_chains_ids = True):
        # This allows ribbons rendering for a Structure.
        # Usually only AtomicStructure supports ribbon rendering.
        from chimerax.atomic import Residue
        polymer_type = Residue.PT_NONE
        return [(res, polymer_type) for res in self._polymers]
