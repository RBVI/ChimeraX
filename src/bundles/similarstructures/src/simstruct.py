# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.state import State  # For session saving
class SimilarStructures(State):
    def __init__(self, hits, query_chain, program = '', database = '',
                 trim = True, alignment_cutoff_distance = 2.0, session = None):
        self.hits = hits
        self._query_chain = query_chain
        self._session = query_chain.structure.session if query_chain else session
        self.program = program			# Name of program that did the search, e.g. foldseek, mmseqs2, blast
        self.program_database = database	# Database program searched, e.g. pdb100, afdb50

        self._allowed_residue_names = None
        if program == 'foldseek':
            if len(hits) > 0 and hits[0].get('foldseek release') is None:
                # Older foldseek versions excluded various non-standard residues. ChimeraX ticket #17890
                from .foldseek_search import foldseek_accepted_3_letter_codes
                self._allowed_residue_names = foldseek_accepted_3_letter_codes
        
        # Default values used when opening and aligning structures
        self.trim = trim
        self.alignment_cutoff_distance = alignment_cutoff_distance

        # Cached values
        self._clear_cached_values()

        if query_chain:
            r2i = {r:i for i,r in enumerate(query_chain.residues) if r is not None}
            qc2f = [r2i[r] for r in self.query_residues]  # Query coordinate index to full sequence index
            self._query_coord_to_sequence_index = qc2f
            qf2c = {fi:ci for ci, fi in enumerate(qc2f)}
            self._query_sequence_to_coord_index = qf2c
        else:
            self._query_coord_to_sequence_index = None
            self._query_sequence_to_coord_index = None
        self._alignment_indexing = 'coordinates' if len(hits) > 0 and hits[0].get('coordinate_indexing') else 'sequence'

        self.name = add_similar_structures(self._session, self)

    def _clear_cached_values(self):
        self._query_residues = None
        self._query_alignment_range = None
        self._query_residue_names = None
        self._sequence_alignment_array = None
        self._alignment_coordinates = None
        self._lddt_score_array = None

    def named_hits(self, hit_names, raise_error = True):
        if isinstance(hit_names, str):
            hit_names = hit_names.split(',')
        if hit_names is None:
            hits = self.hits
        else:
            names = set(hit_names)
            hits = [hit for hit in self.hits if hit['database_full_id'] in names]

        if raise_error and len(hits) == 0:
            msg = 'No similar structures specified'
            if hit_names:
                msg += ' by ' + ', '.join(hit_names)
            from chimerax.core.errors import UserError
            raise UserError(msg)

        return hits
        
    def replace_hits(self, hits):
        self.hits = hits
        self._clear_cached_values()

    @property
    def num_hits(self):
        return len(self.hits)

    @property
    def session(self):
        qc = self.query_chain
        return qc.structure.session if qc else None

    @property
    def query_chain(self):
        qc = self._query_chain
        if qc is not None and qc.structure is None:
            self._query_chain = qc = None
        return qc

    @property
    def query_residues(self):
        qres = self._query_residues
        if qres is None:
            if self._allowed_residue_names is not None:
                qres = _alignment_residues(self._query_chain.existing_residues,
                                           self._allowed_residue_names)
            else:
                qres = self._query_chain.existing_residues
            self._query_residues = qres
        return qres

    def query_residue_names(self):
        if self._query_residue_names is None:
            qstart, qend = self.query_alignment_range()
            qres = self.query_residues
            if self._alignment_indexing == 'coordinates':
                self._query_residue_names = [(r.one_letter_code, r.number) for r in qres[qstart-1:qend]]
            else:
                qs2c = self._query_sequence_to_coord_index
                rnames = []
                for qi in range(qstart-1,qend):
                    if qi in qs2c:
                        r = qres[qs2c[qi]]
                        rnames.append((r.one_letter_code, r.number))
                    else:
                        rnames.append(None)
                self._query_residue_names = rnames
        return self._query_residue_names

    @property
    def description(self):
        msg = f'{self.num_hits} similar structures'
        if self.query_chain:
            q = self.query_chain.string(include_structure = True)
            msg += f' to {q}'
        if self.program_database:
            msg += f' in {self.program_database} database'
        if self.program:
            msg += f' using {self.program}'
        if self.name:
            msg += f', name {self.name}'
        return msg

    def query_alignment_range(self):
        '''Return the range of alignment indices (qstart, qend) that includes all the hit alignments.'''
        qar = self._query_alignment_range
        if qar is not None:
            return qar
        qstarts = []
        qends = []
        for hit in self.hits:
            qstarts.append(hit['qstart'])
            qends.append(hit['qend'])
        qar = qstart, qend = min(qstarts), max(qends)
        return qar

    def query_coordinate_to_alignment_index(self, qci):
        if self._alignment_indexing == 'coordinates':
            return qci
        elif self._alignment_indexing == 'sequence':
            return self._query_coord_to_sequence_index[qci]

    def have_c_alpha_coordinates(self):
        for hit in self.hits:
            if 'tca' not in hit:
                return False
        return True

    def hit_residue_pairing(self, hit):
        return _hit_residue_pairing_coordinate_indexing(hit, self._query_sequence_to_coord_index)
 
    def sequence_alignment_array(self):
        saa = self._sequence_alignment_array
        if saa is not None:
            return saa
        qstart, qend = self.query_alignment_range()
        self._sequence_alignment_array = saa = _sequence_alignment(self.hits, qstart, qend)
        return saa

    def alignment_coordinates(self):
        ac = self._alignment_coordinates
        if ac is not None:
            return ac
        qstart, qend = self.query_alignment_range()
        self._alignment_coordinates = hits_xyz, hits_mask = _alignment_coordinates(self.hits, qstart, qend)
        return hits_xyz, hits_mask
    
    def compute_rmsds(self, alignment_cutoff_distance = None):
        if self.query_chain is None:
            return False
        # Compute percent coverage and percent close C-alpha values per hit.
        qres = self.query_residues
        qatoms = qres.find_existing_atoms('CA')
        query_xyz = qatoms.coords
        for hit in self.hits:
            hxyz = hit_coords(hit)
            if hxyz is not None:
                hi, qi = self.hit_residue_pairing(hit)
                if len(hi) >= 3:  # Need at least 3 atom pairs to align
                    try:
                        p, rms, npairs = align_xyz_transform(hxyz[hi], query_xyz[qi],
                                                             cutoff_distance=alignment_cutoff_distance)
                    except:
                        print (hi, len(hxyz), qi, len(query_xyz), hit)
                        raise
                    hit['rmsd'] = rms
                    hit['close'] = 100*npairs/len(hi)
                    hit['cutoff_distance'] = alignment_cutoff_distance
                hit['coverage'] = 100 * len(qi) / len(query_xyz)
        return True

    def set_coverage_attribute(self):
        if self.query_chain is None  or getattr(self, '_coverage_attribute_set', False):
            return
        self._coverage_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'coverage', "Similar Structures", attr_type = int)

        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            ai = self.query_coordinate_to_alignment_index(ri)
            if ai >= qstart-1 and ai <= qend-1:
                ai -= (qstart-1)
                alignment_array = self.sequence_alignment_array()
                count = count_nonzero(alignment_array[1:,ai])
            else:
                count = 0
            r.coverage = count

    def set_conservation_attribute(self):
        if self.query_chain is None  or getattr(self, '_conservation_attribute_set', False):
            return
        self._conservation_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'conservation', "Similar Structures", attr_type = float)

        alignment_array = self.sequence_alignment_array()
        seq, count, total = _consensus_sequence(alignment_array)
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            ai = self.query_coordinate_to_alignment_index(ri)
            if ai >= qstart-1 and ai <= qend-1:
                ai -= (qstart-1)
                conservation = count[ai] / total[ai]
            else:
                conservation = 0
            r.conservation = conservation

    def set_entropy_attribute(self):
        if self.query_chain is None or getattr(self, '_entropy_attribute_set', False):
            return
        self._entropy_attribute_set = True
        
        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'entropy', "Similar Structures", attr_type = float)

        alignment_array = self.sequence_alignment_array()
        entropy = _sequence_entropy(alignment_array)
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            ai = self.query_coordinate_to_alignment_index(ri)
            if ai >= qstart-1 and ai <= qend-1:
                ai -= (qstart-1)
                r.entropy = entropy[ai]

    def set_lddt_attribute(self):
        if self.query_chain is None or getattr(self, '_lddt_attribute_set', False):
            return
        self._lddt_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'lddt', "Similar Structures", attr_type = float)

        lddt_scores = self.lddt_scores()
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        # TODO: This only works with coordinate indexing.
        for ri,r in enumerate(self.query_residues):
            ai = self.query_coordinate_to_alignment_index(ri)
            if ai >= qstart-1 and ai <= qend-1:
                ai -= (qstart-1)
                alignment_array = self.sequence_alignment_array()
                nscores = count_nonzero(alignment_array[1:,ai])
                ave_lddt = lddt_scores[:,ai].sum() / nscores
            else:
                ave_lddt = 0
            r.lddt = ave_lddt

    def lddt_scores(self):
        lddt_scores = self._lddt_score_array
        if lddt_scores is not None:
            return lddt_scores
        
        qstart, qend = self.query_alignment_range()
        qres = self.query_residues
        qatoms = qres.find_existing_atoms('CA')
        query_xyz = qatoms.coords[qstart-1:qend,:]
        hits_xyz, hits_mask = self.alignment_coordinates()
        query_missing_coords = (len(query_xyz) < hits_xyz.shape[1])
        if query_missing_coords:
            qs2ci = self._query_sequence_to_coord_index
            qsi_with_coords = [qi-(qstart-1) for qi in range(qstart-1, qend) if qi in qs2ci]
            if len(qsi_with_coords) != len(query_xyz):
                raise ValueError(f'Query structure {len(query_xyz)} coordinates does not match query sequence index to coordinate index map with {len(qsi_with_coords)} coordinates')
            hits_xyz = hits_xyz[:,qsi_with_coords,:]
            hits_mask = hits_mask[:,qsi_with_coords]

        from . import lddt
        lddt_scores = lddt.local_distance_difference_test(query_xyz, hits_xyz, hits_mask)
        if query_missing_coords:
            from numpy import zeros, float32
            expanded_lddt_scores = zeros((len(lddt_scores), qend-qstart+1), float32) 
            expanded_lddt_scores[:,qsi_with_coords] = lddt_scores
            lddt_scores = expanded_lddt_scores
        self._lddt_score_array = lddt_scores
        return lddt_scores

    def take_snapshot(self, session, flags):
        data = {'hits': self.hits,
                'query_chain': self.query_chain,
                'program': self.program,
                'program_database': self.program_database,
                'trim': self.trim,
                'alignment_cutoff_distance': self.alignment_cutoff_distance,
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        return SimilarStructures(data['hits'], data['query_chain'],
                                 program = data.get('program', ''),
                                 database = data.get('database', ''),
                                 trim = data.get('trim', True),
                                 alignment_cutoff_distance = data.get('alignment_cutoff_distance', 2.0))

    def save_sms_file(self, path):
        j = self.sms_data()
        with open(path, 'w') as file:
            file.write(j)

    def save_to_directory(self, directory, filename = None, add_to_file_history = True):
        if filename is None:
            filename = self._sms_filename(directory)
        from os.path import join, exists
        path = join(directory, filename)
        if not exists(directory):
            import os
            os.makedirs(directory)

        self.save_sms_file(path)

        if add_to_file_history:
            # Record in file history so it is easy to reopen similar structure results.
            models = [self.query_chain.structure] if self.query_chain else []
            from chimerax.core.filehistory import remember_file
            remember_file(self.session, path, 'sms', models, file_saved=True)

        return path

    def _sms_filename(self, directory):
        qc = self.query_chain
        if qc is None:
            filename = 'results'
        else:
            from os.path import splitext
            filename = splitext(qc.structure.name)[0]
            if qc.structure.num_chains > 1:
                filename += f'_{qc.chain_id}'
        if self.program:
            filename += f'_{self.program}'
        if self.program_database:
            filename += f'_{self.program_database}'
        filename = filename.replace(' ', '_')
        prefix = filename
        filename += '.sms'
        count = 1
        from os.path import exists, join
        while exists(join(directory, filename)):
            filename = prefix + f'_{count}.sms'
            count += 1
        return filename

    def sms_data(self):
        attributes = ('hits', 'program', 'program_database', 'trim', 'alignment_cutoff_distance')
        data = {attr:getattr(self, attr)  for attr in attributes}

        # Replace numpy arrays with lists since
        # JSON encoder cannot handle numpy arrays.
        hits = []
        from numpy import ndarray
        for hit in data['hits']:
            if 'tca' in hit and isinstance(hit['tca'], ndarray):
                hit = hit.copy()
                hit['tca'] = hit['tca'].tolist()
            hits.append(hit)
        data['hits'] = hits

        chain = self.query_chain
        if chain:
            cpath = getattr(chain.structure, 'filename')
            if cpath:
                data['query_chain_path'] = cpath
                data['query_chain_id'] = chain.chain_id
        else:
            if hasattr(self, '_query_chain_path'):
                data['query_chain_path'] = self._query_chain_path
            if hasattr(self, '_query_chain_id'):
                data['query_chain_id'] = self._query_chain_id

        data['version'] = 1

        import json
        j = json.dumps(data)

        return j

    @staticmethod
    def read_sms_file(session, path):
        with open(path, 'r') as file:
            j = file.read()
        import json
        data = json.loads(j)

        for hit in data['hits']:
            if 'tca' in hit:
                from numpy import array, float32
                hit['tca'] = array(hit['tca'], float32)

        query_chain = None
        if 'query_chain_path' in data:
            cpath = data['query_chain_path']
            chain_id = data['query_chain_id']
            query_chain = _find_chain(session, cpath, chain_id)

        r = SimilarStructures(data['hits'], query_chain,
                              program = data['program'], database = data['program_database'],
                              trim = data['trim'], alignment_cutoff_distance = data['alignment_cutoff_distance'],
                              session = session)

        if 'query_chain_path' in data:
            r._query_chain_path = data['query_chain_path']
            r._query_chain_id = data['query_chain_id']

        r.sms_path = path
        
        return r
        
    def open_hit(self, session, hit, trim = None, align = True, alignment_cutoff_distance = None,
                 in_file_history = True, log = True):
        if trim is None:
            trim = self.trim
        if alignment_cutoff_distance is None:
            alignment_cutoff_distance = self.alignment_cutoff_distance
        query_chain = self.query_chain
        
        af_frag = hit.get('alphafold_fragment')
        if af_frag is not None and af_frag != 'F1':
            session.logger.warning(f'AlphaFold database entry {hit["alphafold_id"]} was predicted in fragments and ChimeraX is not able to fetch the fragment structures')
            return

        db_id = hit.get('database_id')
        if in_file_history:
            from_db = 'from alphafold' if hit['database'].startswith('afdb') else ''
            open_cmd = f'open {db_id} {from_db}'
            if not log:
                open_cmd += ' logInfo false'
            from chimerax.core.commands import run
            structures = run(session, open_cmd, log = log)
        else:
            kw = {'from_database': 'alphafold'} if hit['database'].startswith('afdb') else {}
            structures, status = session.open_command.open_data(db_id, log_info = log, **kw)
            session.models.add(structures)

        name = hit.get('database_full_id')
        # Can get multiple structures such as NMR ensembles from PDB.
        stats = []
        for structure in structures:
            structure.name = name
            self._remember_sequence_alignment(structure, hit)
            _check_structure_sequence(structure, hit)
            _trim_structure(structure, hit, trim, log = log,
                            allowed_residue_names = self._allowed_residue_names)
            if structure.deleted:
                continue  # No residues aligned and every residue trimmed
            _show_ribbons(structure)

            if query_chain is not None and align:
                # Align the model to the query structure using sequence alignment.
                hit_chain = _hit_chain(structure, hit)
                if hit_chain is None:
                    continue	# Hit hand no aligned residues with coordinates
                res, query_res = _alignment_residue_pairs(hit, hit_chain, query_chain)
                if len(res) >= 3:
                    p, rms, npairs = alignment_transform(res, query_res, alignment_cutoff_distance)
                    stats.append((rms, npairs))
                    structure.position = p

        if log and query_chain is not None and align and stats:
            chain_id = hit.get('chain_id')
            cname = '' if chain_id is None else f' chain {chain_id}'
            if len(structures) == 1:
                msg = f'Alignment of {db_id}{cname} to query has RMSD {"%.3g" % rms} using {npairs} of {len(res)} paired residues'
            else:
                rms = [rms for rms,npair in stats]
                rms_min, rms_max = min(rms), max(rms)
                npair = [npair for rms,npair in stats]
                npair_min, npair_max = min(npair), max(npair)
                msg = f'Alignment of {db_id}{cname} ensemble of {len(stats)} structures to query has RMSD {"%.3g" % rms_min} - {"%.3g" % rms_max} using {npair_min}-{npair_max} of {len(res)} paired residues'      
            if alignment_cutoff_distance is not None and alignment_cutoff_distance > 0:
                msg += f' within cutoff distance {alignment_cutoff_distance}'
            session.logger.info(msg)

        structures = [s for s in structures if not s.deleted]
        return structures

    def _remember_sequence_alignment(self, structure, hit):
        '''
        Remember the positions of the aligned residues as part of the hit dictionary,
        so that after hit residues are trimmed we can still deduce the hit to query
        residue pairing.  Foldseek hit and query alignment only includes a subset
        of residues from the structures and only include residues that have atomic coordinates.
        '''
        if hit.get('coordinate_indexing'):
            # Foldseek indexing only counts residues with atom coordinates
            chain_id = hit.get('chain_id')
            hit_chain = structure_chain_with_id(structure, chain_id)
            rnames = self._allowed_residue_names
            hro = _indices_of_residues_with_coords(hit_chain, hit['tstart'], hit['tend'], rnames)
            if self.query_chain:
                qro = _indices_of_residues_with_coords(self.query_chain, hit['qstart'], hit['qend'], rnames)
            else:
                qro = None
        else:
            hro = list(range(hit['tstart']-1, hit['tend']))
            qro = list(range(hit['qstart']-1, hit['qend']))
        hit['aligned_residue_offsets'] = hro
        hit['query_residue_offsets'] = qro

from chimerax.core.state import StateManager
class SimilarStructuresManager(StateManager):
    def __init__(self):
        self._name_to_simstruct = {}
    def add_similar_structures(self, results, name = None):
        uname = self._unique_name(name, results.program)
        self._name_to_simstruct[uname] = results
        return uname
    def remove_similar_structures(self, results):
        names = [name for name,r in self._name_to_simstruct.items() if r is results]
        for name in names:
            del self._name_to_simstruct[name]
    def find_similar_structures(self, name):
        n2s = self._name_to_simstruct
        if name is None and len(n2s) == 1:
            for name, s in n2s.items():
                return s
        return n2s.get(name)
    def _unique_name(self, name, program):
        if name is None:
            prefix = {'foldseek':'fs', 'mmseqs2':'mm', 'blast':'bl'}.get(program, 'sim')
            name = self._add_numeric_suffix(prefix)
        elif name in self._name_to_simstruct:
            name = self._add_numeric_suffix(name)
        return name
    def _add_numeric_suffix(self, prefix):
            i = 1
            while True:
                name = prefix + str(i)
                if name in self._name_to_simstruct:
                    i += 1
                else:
                    return name
    @property
    def count(self):
        return len(self._name_to_simstruct)
    @property
    def names(self):
        return list(self._name_to_simstruct.keys())
    def take_snapshot(self, session, flags):
        return {'name_to_simstruct': self._name_to_simstruct,
                'version': 1}
    @classmethod
    def restore_snapshot(cls, session, data):
        ssm = cls()
        ssm._name_to_simstruct = data['name_to_simstruct']
        return ssm
    def reset_state(self, session):
        self._name_to_simstruct.clear()

def similar_structures_manager(session, create = True):
    ssm = getattr(session, 'similar_structures', None)
    if ssm is None and create:
        session.similar_structures = ssm = SimilarStructuresManager()
    return ssm

def add_similar_structures(session, results, name = None):
    ssm = similar_structures_manager(session)
    name = ssm.add_similar_structures(results, name = name)
    return name

def remove_similar_structures(session, results):
    ssm = similar_structures_manager(session)
    ssm.remove_similar_structures(results)
            
def similar_structure_results(session, name = None, raise_error = True):
    '''Return currently open similar structure results.'''
    ssm = similar_structures_manager(session, create = False)
    s = ssm.find_similar_structures(name) if ssm else None
    if raise_error and s is None:
        if ssm is None or ssm.count == 0:
            msg = 'There are no open sets of similar structures'
        elif name is None:
            names = ', '.join(ssm.names)
            msg = f'There are multiple open sets of similar structures ({names}). Specify one with the fromSet option'
        else:
            names = ', '.join(ssm.names)
            msg = f'There is no set of similar structures named "{name}". Available names are {names}.'
        from chimerax.core.errors import UserError
        raise UserError(msg)
    return s
    
def _indices_of_residues_with_coords(chain, start, end, allowed_residue_names):
    seqi = []
    si = 0
    for i, r in enumerate(chain.residues):
        if r is not None and (allowed_residue_names is None or r.name in allowed_residue_names) and r.find_atom('CA'):
            si += 1
            if si >= start and si <= end:
                seqi.append(i)
            elif si > end:
                break
    return seqi
    
def _trim_structure(structure, hit, trim, ligand_range = 3.0, log = True, allowed_residue_names = None):
    if not trim:
        return

    chain_res, ri_start, ri_end = _residue_range(structure, hit, structure.session.logger,
                                                 allowed_residue_names = allowed_residue_names)
    if ri_start is None and log:
        name = hit.get('database_full_id')
        msg = f'Hit {name} has no coordinates for aligned residues'
        structure.session.logger.warning(msg)

    msg = []
    logger = structure.session.logger  # Get logger before structure deleted.
    
    trim_lig = (trim is True or 'ligands' in trim) and hit['database'] == 'pdb'
    chain_id = hit.get('chain_id')
    if (trim is True or 'chains' in trim) and chain_id is not None:
        if trim_lig:
            # Delete polymers in other chains, but not ligands from other chains.
            ocres = [c.existing_residues for c in tuple(structure.chains) if c.chain_id != chain_id]
            if ocres:
                from chimerax.atomic import concatenate
                ocres = concatenate(ocres)
        else:
            # Delete everything from other chain ids, including ligands.
            sres = structure.residues
            ocres = sres[sres.chain_ids != chain_id]
        if ocres:
            msg.append(f'{structure.num_chains-1} extra chains')
            ocres.delete()

    trim_seq = (trim is True or 'sequence' in trim)
    if trim_seq:
        if ri_start is None:
            chain_res.delete()
        else:
            # Trim end first because it changes the length of the res array.
            if ri_end < len(chain_res)-1:
                cterm = chain_res[ri_end+1:]
                msg.append(f'{len(cterm)} C-terminal residues')
                cterm.delete()
            if ri_start > 0:
                nterm = chain_res[:ri_start]
                msg.append(f'{len(nterm)} N-terminal residues')
                nterm.delete()

    if trim_lig:
        if not structure.deleted and structure.num_residues > len(chain_res):
            sres = structure.residues
            from chimerax.atomic import Residue
            npres = sres[sres.polymer_types == Residue.PT_NONE]
            if len(chain_res) == 0:
                npres.delete()
            else:
                npnear = _find_close_residues(npres, chain_res, ligand_range)
                npfar = npres.subtract(npnear)
                if npfar:
                    msg.append(f'{len(npfar)} non-polymer residues more than {ligand_range} Angstroms away')
                    npfar.delete()
                
    if log and msg:
        logger.info(f'Deleted {", ".join(msg)}.')


def _find_close_residues(residues1, residues2, distance):
    a1 = residues1.atoms
    axyz1 = a1.scene_coords
    axyz2 = residues2.atoms.scene_coords
    from chimerax.geometry import find_close_points
    ai1, ai2 = find_close_points(axyz1, axyz2, distance)
    r1near = a1[ai1].unique_residues
    return r1near

def _hit_chain(structure, hit):
    chain_id = hit.get('chain_id')
    if chain_id is None:
        chain = structure.chains[0] if structure.num_chains == 1 else None
    else:
        chain = structure_chain_with_id(structure, chain_id)
    return chain

def _residue_range(structure, hit, log, allowed_residue_names):
    '''
    Return all the chain residues with atom coordinates and a start and end index into
    that set of residues for the segment used in the hit alignment.
    '''
    chain_id = hit.get('chain_id')
    chain = structure_chain_with_id(structure, chain_id)
    if hit.get('coordinate_indexing'):
        # Index is for the sequence of residues with atomic coordinates
        ri0, ri1 = hit['tstart']-1, hit['tend']-1
        res = _alignment_residues(chain.existing_residues, allowed_residue_names)
    else:
        # Alignment indexing is for the full sequence including residues without coordinates.
        hro = hit['aligned_residue_offsets']
        ares = chain.residues[hro[0]:hro[-1]+1]
        eares = [r for r in ares if r is not None]
        res = chain.existing_residues
        if len(eares) == 0:
            # There are no hit residues with coordinates that are part of the alignment.
            ri0 = ri1 = None
        else:
            ri0 = res.index(eares[0])
            ri1 = res.index(eares[-1])
    return res, ri0, ri1

def _check_structure_sequence(structure, hit):
    # Check that the database structure sequence matches what is given in the hit sequence alignment.
    hri = hit['aligned_residue_offsets']
    chain = _hit_chain(structure, hit)
    struct_seq = chain.characters
    saligned_seq = ''.join(struct_seq[ri] for ri in hri)
    hseq_ungapped = hit['taln'].replace('-', '')
    # Don't warn when sequence alignment has an X but ChimeraX has non-X for a modified residue.
    mismatches = [i for i, (saa, haa) in enumerate(zip(saligned_seq, hseq_ungapped)) if saa != haa and haa != 'X']
    if mismatches:
        i = mismatches[0]
        msg = f'Hit structure sequence {structure.name} amino acid {saligned_seq[i]} at sequence position {hri[i]} does not match sequence alignment {hseq_ungapped[i]} at position {i}.  Structure sequence {struct_seq}.  Alignment sequence {hseq_ungapped}.'
        structure.session.logger.warning(msg)

def structure_chain_with_id(structure, chain_id):
    if chain_id is None:
        chains = structure.chains
    else:
        chains = [chain for chain in structure.chains if chain.chain_id == chain_id]
    chain = chains[0] if len(chains) == 1 else None
    return chain

def _alignment_residues(residues, allowed_residue_names):
    '''
    Filter residues keeping only ones with C-alpha atom and an allowed residue name.
    '''
    if allowed_residue_names is None:
        rok = residues
    else:
        rok = [r for r in residues if r.name in allowed_residue_names]
        if len(rok) < len(residues):
            from chimerax.atomic import Residues
            residues = Residues(rok)
    if (residues.atoms.names == 'CA').sum() < len(residues):
        from chimerax.atomic import Residues
        residues = Residues([r for r in residues if 'CA' in r.atoms.names])
    return residues
    
def _alignment_residue_pairs(hit, hit_chain, query_chain):
    hres = hit_chain.residues
    qres = query_chain.residues
    hro = hit['aligned_residue_offsets']
    qro = hit['query_residue_offsets']
    
    qaln, taln = hit['qaln'], hit['taln']
    ti = qi = 0
    atr, aqr = [], []
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            qr = qres[qro[qi]]
            tr = hres[hro[ti]]
            if tr is not None and qr is not None:
                atr.append(tr)
                aqr.append(qr)
        if taa != '-':
            ti += 1
        if qaa != '-':
            qi += 1

    from chimerax.atomic import Residues
    return Residues(atr), Residues(aqr)

def _sequence_alignment(hits, qstart, qend):
    '''
    Return the sequence alignment between hits and query as a numpy 2D array of bytes.
    The first row is the query sequence with one subsequent row for each hit.
    There are no gaps in the query, and gaps in the hit sequences have 0 byte values.
    '''
    nhit = len(hits)
    qlen = qend - qstart + 1
    from numpy import zeros, byte
    alignment = zeros((nhit+1, qlen), byte)
    for h, hit in enumerate(hits):
        qaln, taln = hit['qaln'], hit['taln']
        qi = hit['qstart']
        for qaa, taa in zip(qaln, taln):
            if qi >= qstart and qi <= qend:
                if qaa != '-':
                    ai = qi-qstart
                    alignment[0,ai] = ord(qaa)	# First row is query sequence
                    if taa != '-':
                        alignment[h+1,ai] = ord(taa)
            if qaa != '-':
                qi += 1
    return alignment

def _alignment_coordinates(hits, qstart, qend):
    '''
    Return C-alpha atom coordinates for aligned sequences.
    The returned coordinate array is size (nhits, qend-qstart+1, 3).
    Also return a mask indicating positions that have coordinates.
    The returned mask array is size (nhits, qend-qstart+1).
    '''
    nhit = len(hits)
    qlen = qend - qstart + 1

    from numpy import zeros, float32
    xyz = zeros((nhit, qlen, 3), float32)
    mask = zeros((nhit, qlen), bool)

    for h, hit in enumerate(hits):
        qaln, taln = hit['qaln'], hit['taln']
        qi = hit['qstart']
        hi = hit['tstart']
        hxyz = hit_coords(hit)
        coord_indexing = hit.get('coordinate_indexing')
        if not coord_indexing:
            hit_seq_to_coord_index = _hit_sequence_to_coordinate_index(hit)
        for qaa, taa in zip(qaln, taln):
            if qaa != '-' and taa != '-' and qi >= qstart and qi <= qend:
                qai = qi-qstart
                if coord_indexing:
                    hci = hi-1
                else:
                    hci = hit_seq_to_coord_index.get(hi-1)  # Map sequence to coordinate indices.
                if hci is not None:
                    xyz[h,qai,:] = hxyz[hci]
                    mask[h,qai] = True
            if qaa != '-':
                qi += 1
            if taa != '-':
                hi += 1

    return xyz, mask

def alignment_transform(res, query_res, cutoff_distance = None):
    qatoms = query_res.find_existing_atoms('CA')
    qxyz = qatoms.scene_coords
    tatoms = res.find_existing_atoms('CA')
    txyz = tatoms.coords
    p, rms, npairs = align_xyz_transform(txyz, qxyz, cutoff_distance = cutoff_distance)
    return p, rms, npairs

def align_xyz_transform(txyz, qxyz, cutoff_distance = None):
    if cutoff_distance is None or cutoff_distance <= 0:
        from chimerax.geometry import align_points
        p, rms = align_points(txyz, qxyz)
        npairs = len(txyz)
    else:
        from chimerax.std_commands.align import align_and_prune
        p, rms, indices = align_and_prune(txyz, qxyz, cutoff_distance)
#        _look_for_more_alignments(txyz, qxyz, cutoff_distance, indices)
        npairs = len(indices)
    return p, rms, npairs

def _look_for_more_alignments(txyz, qxyz, cutoff_distance, indices):
    sizes = [len(txyz), len(indices)]
    from numpy import ones
    mask = ones(len(txyz), bool)
#    from chimerax.std_commands.align import align_and_prune, IterationError
    while True:
        mask[indices] = 0
        if mask.sum() < 3:
            break
        try:
#            p, rms, indices = align_and_prune(txyz, qxyz, cutoff_distance, mask.nonzero()[0])
            p, rms, indices, close_mask = _align_and_prune(txyz, qxyz, cutoff_distance, mask.nonzero()[0])
            sizes.append(len(indices))
            nclose = close_mask.sum()
            if nclose > len(indices):
                sizes.append(-nclose)
                sizes.append(-(mask*close_mask).sum())
        except IterationError:
            break
    print (' '.join(str(s) for s in sizes))

from chimerax.core.errors import UserError
class IterationError(UserError):
    pass

def _align_and_prune(xyz, ref_xyz, cutoff_distance, indices = None):

    import numpy
    if indices is None:
        indices = numpy.arange(len(xyz))
    axyz, ref_axyz = xyz[indices], ref_xyz[indices]
    from chimerax.geometry import align_points
    p, rms = align_points(axyz, ref_axyz)
    dxyz = p*axyz - ref_axyz
    d2 = (dxyz*dxyz).sum(axis=1)
    cutoff2 = cutoff_distance * cutoff_distance
    i = d2.argsort()
    if d2[i[-1]] <= cutoff2:
        dxyz = p*xyz - ref_xyz
        d2 = (dxyz*dxyz).sum(axis=1)
        close_mask = (d2 <= cutoff2)
        return p, rms, indices, close_mask

    # cull 10% or...
    index1 = int(len(d2) * 0.9)
    # cull half the long pairings
    index2 = int(((d2 <= cutoff2).sum() + len(d2)) / 2)
    # whichever is fewer
    index = max(index1, index2)
    survivors = indices[i[:index]]

    if len(survivors) < 3:
        raise IterationError("Alignment failed;"
            " pruning distances > %g left less than 3 atom pairs" % cutoff_distance)
    return _align_and_prune(xyz, ref_xyz, cutoff_distance, survivors)

def hit_coords(hit):
    '''
    Returns the C-alpha atom positions for each residue that has such coordinates in the hit structure
    including residues outside the aligned interval.
    '''
    hxyz = hit.get('tca')
    return hxyz

def hit_coordinates_sequence(hit):
    '''
    Return a map from hit coordinate index to sequence 1 letter code.
    '''
    hsi = range(hit['tstart']-1, hit['tend'])  # 0-based sequence indexing
    hseq = hit['taln'].replace('-','')
    hi2ci = _hit_sequence_to_coordinate_index(hit)
    if hi2ci is None:
        ci2aa = {hi:aa for hi,aa in zip(hsi, hseq)}
    else:
        ci2aa = {hi2ci[hi]:aa for hi,aa in zip(hsi, hseq) if hi in hi2ci}
    return ci2aa

def _hit_sequence_to_coordinate_index(hit):
    '''Return dictionary mapping hit sequence indices (0-based) to coordinate indices (0-based).'''
    hit_coord_to_seq_index = hit.get('tca_index')  # Array mapping coordinate index to full sequence index
    if hit_coord_to_seq_index is None:
        return None
    hit_seq_to_coord_index = {fi:ci for ci, fi in enumerate(hit_coord_to_seq_index)}
    return hit_seq_to_coord_index

def _hit_residue_pairing_coordinate_indexing(hit, query_seq_to_coord_index):
    '''
    Returns an array of hit indices and query indices that are paired that index into the
    C-alpha coordinates array for the hit and the query.  Those coordinate arrays include
    all residues which have C-alpha coordinates even outside the alignment interval for the hit.
    '''
    ati, aqi = _alignment_index_pairs(hit, offset = True)

    if hit.get('coordinate_indexing'):
        cti, cqi = ati, aqi
    else:
        # Convert sequence indexing to coordinate indexing.
        # Paired residues may not have coordinates and those are eliminated.
        hit_seq_to_coord_index = _hit_sequence_to_coordinate_index(hit)        
        if hit_seq_to_coord_index is None:
            return None, None
        cti, cqi = [], []
        for ti,qi in zip(ati, aqi):
            if ti in hit_seq_to_coord_index and qi in query_seq_to_coord_index:
                cti.append(hit_seq_to_coord_index[ti])
                cqi.append(query_seq_to_coord_index[qi])

    return cti, cqi

def _alignment_index_pairs(hit, offset):
    '''
    Return an array of hit indices and query indices that are paired that index into
    the hit and query ungapped sequences starting at the start of the alignment.
    If offset is true then indexing is shifted by hit['tstart'] and hit['qstart']
    to give full sequence indexing or coordinate indexing.
    '''
    qaln, taln = hit['qaln'], hit['taln']
    ti = qi = 0
    ati, aqi = [], []
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            ati.append(ti)
            aqi.append(qi)
        if taa != '-':
            ti += 1
        if qaa != '-':
            qi += 1
    from numpy import array, int32
    ati, aqi = array(ati, int32), array(aqi, int32)
    if offset:
        ati += hit['tstart']-1
        aqi += hit['qstart']-1
    return ati, aqi

def _show_ribbons(structure):
    for c in structure.chains:
        cres = c.existing_residues
        if not cres.ribbon_displays.any():
            cres.ribbon_displays = True
            cres.atoms.displays = False
            
# -----------------------------------------------------------------------------
#
def _find_chain(session, structure_path, chain_id):
    from chimerax.atomic import AtomicStructure
    for m in session.models.list(type = AtomicStructure):
        if getattr(m, 'filename', '') == structure_path:
            for chain in m.chains:
                if chain.chain_id == chain_id:
                    return chain

    # Open file if possible
    from os.path import exists
    if exists(structure_path):
        from chimerax.core.commands import run, quote_path_if_necessary
        cmd = f'open {quote_path_if_necessary(structure_path)}'
        structure = run(session, cmd)[0]
        query_chain = structure_chain_with_id(structure, chain_id)
    else:
        query_chain = None

    return query_chain

# -----------------------------------------------------------------------------
#
def _consensus_sequence(alignment_array):
    seqlen = alignment_array.shape[1]
    from numpy import count_nonzero, bincount, argmax, empty, byte, int32
    seq = empty((seqlen,), byte)
    count = empty((seqlen,), int32)
    total = empty((seqlen,), int32)
    for i in range(seqlen):
        aa = alignment_array[:,i]
        total[i] = t = count_nonzero(aa)
        if t > 0:
            bc = bincount(aa)
            mi = argmax(bc[1:]) + 1
            seq[i] = mi
            count[i] = bc[mi]
        else:
            # No hit aligns with this column.
            seq[i] = count[i] = 0
    return seq, count, total

# -----------------------------------------------------------------------------
#
def _sequence_entropy(alignment_array):
    seqlen = alignment_array.shape[1]
    from numpy import bincount, empty, float32, array, int32
    entropy = empty((seqlen,), float32)
    aa_1_letter_codes = 'ARNDCEQGHILKMFPSTWYV'
    aa_char = array([ord(c) for c in aa_1_letter_codes], int32)
    bins = aa_char.max() + 1
    for i in range(seqlen):
        aa = alignment_array[:,i]
        bc = bincount(aa, minlength = bins)[aa_char]
        entropy[i] = _entropy(bc)
    return entropy
    
# -----------------------------------------------------------------------------
#
def _entropy(bin_counts):
    total = bin_counts.sum()
    if total == 0:
        return 0.0
    nonzero = (bin_counts > 0)
    p = bin_counts[nonzero] / total
    from numpy import log2
    e = -(p*log2(p)).sum()
    return e
    
# -----------------------------------------------------------------------------
#
def similar_structures_open(session, hit_name, trim = None, align = True, alignment_cutoff_distance = None,
                            in_file_history = True, log = True, from_set = None):
    results = similar_structure_results(session, from_set)

    matches = [hit for hit in results.hits if hit['database_full_id'] == hit_name]
    for hit in matches:
        results.open_hit(session, hit, trim = trim,
                         align = align, alignment_cutoff_distance = alignment_cutoff_distance,
                         in_file_history = in_file_history, log = log)
    if len(matches) == 0:
        session.logger.error(f'No similar structure {hit_name} in table')
    
# -----------------------------------------------------------------------------
#
def similar_structures_close(session, set_name):
    results = similar_structure_results(session, set_name)
    from .gui import similar_structures_panel
    panel = similar_structures_panel(session)
    if panel and panel.results is results:
        panel.delete()
    else:
        remove_similar_structures(session, results)
    
# -----------------------------------------------------------------------------
#
def similar_structures_list(session):
    ssm = similar_structures_manager(session)
    names = ', '.join(ssm.names)
    msg = f'There are {ssm.count} open sets of similar structures: {names}'
    session.logger.status(msg, log=True)

def similar_structure_hit_by_name(session, hit_name, from_set = None):
    results = similar_structure_results(session, from_set)
    for hit in results.hits:
        if hit['database_full_id'] == hit_name:
            return hit, results
    return None, None
    
def similar_structures_pairing(session, hit_structure, color = None, radius = None,
                               halfbond_coloring = None, from_set = None):
    '''Show pseudobonds between aligned residues for a similar structures hit and query.'''
    hit_name = hit_structure.name
    hit, results = similar_structure_hit_by_name(session, hit_name, from_set)
    if hit is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any hit {hit_name} in similar structures table')

    query_chain = results.query_chain
    r_pairs = hit_and_query_residue_pairs(hit_structure, query_chain, hit)
    ca_pairs = []
    for hr, qr in r_pairs:
        hca = hr.find_atom('CA')
        qca = qr.find_atom('CA')
        if hca and qca:
            ca_pairs.append((hca, qca))

    if len(ca_pairs) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any residues to pair for {hit_name}')

    g = session.pb_manager.get_group(f'{hit_name} pairing')
    if g.id is not None:
        g.clear()

    for hca, qca in ca_pairs:
        b = g.new_pseudobond(hca, qca)
        if color is not None:
            b.color = color
        if radius is not None:
            b.radius = radius
        if halfbond_coloring is not None:
            b.halfbond = halfbond_coloring

    if g.id is None:
        session.models.add([g])

    return g

def hit_and_query_residue_pairs(hit_structure, query_chain, hit):
    ho = hit.get('aligned_residue_offsets')
    qo = hit.get('query_residue_offsets')
    if ho is None or qo is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Could not find alignment offsets for {hit["database_id"]}')
    
    qres = query_chain.residues
    hit_chain = structure_chain_with_id(hit_structure, hit.get('chain_id'))
    hres = hit_chain.residues
    ahi, aqi = _alignment_index_pairs(hit, offset = False)
    r_pairs = []
    for hi, qi in zip(ahi, aqi):
        hr = hres[ho[hi]]
        qr = qres[qo[qi]]
        if hr is not None and qr is not None:
            r_pairs.append((hr, qr))

    return r_pairs
    
def similar_structures_sequence_alignment(session, hit_structure, from_set = None):
    '''Show pairwise sequence alignment returned by similar structure search.'''
    hit_name = hit_structure.name
    hit, results = similar_structure_hit_by_name(session, hit_name, from_set)
    if hit is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any hit named {hit_name} in similar structures table')

    # Create query and hit gapped aligned sequences
    query_chain = results.query_chain
    qname = f'{query_chain.structure.name}_{query_chain.chain_id}'
    from chimerax.atomic import Sequence, SeqMatchMap
    qseq = Sequence(name = qname, characters = hit['qaln'])
    hseq = Sequence(name = hit_name, characters = hit['taln'])

    matches = len([1 for qc,tc in zip(hit['qaln'], hit['taln']) if qc == tc])
    paired = len([1 for qc,tc in zip(hit['qaln'], hit['taln']) if qc != '-' and tc != '-'])
    qlen = len([1 for qc in hit['qaln'] if qc != '-'])
    tlen = len([1 for qc in hit['taln'] if qc != '-'])
#    print (f'{matches} identical, {len(hit["qaln"])} gapped alignment length, query length {qlen}, hit length {tlen}, paired {paired}')

    # Create alignment
    seqs = [qseq, hseq]
    am = session.alignments
    a = am.new_alignment(seqs, identify_as = hit_name, name = f'Search query {qname} and hit {hit_name}',
                         auto_associate = False, intrinsic = True)

    # Create query structure association with sequence
    reassoc = None  # Not used
    errors = 0
    query_match_map = SeqMatchMap(qseq, query_chain)
    qres = query_chain.residues
    for pos,qo in enumerate(hit.get('query_residue_offsets')):
        if qres[qo] is not None:
            query_match_map.match(qres[qo], pos)
    a.prematched_assoc_structure(query_match_map, errors, reassoc)  # Associate query

    # Create hit structure association with sequence
    hit_chain = structure_chain_with_id(hit_structure, hit.get('chain_id'))
    hit_match_map = SeqMatchMap(hseq, hit_chain)
    hres = hit_chain.residues
    for pos,ho in enumerate(hit.get('aligned_residue_offsets')):
        if hres[ho] is not None:
            hit_match_map.match(hres[ho], pos)
    a.prematched_assoc_structure(hit_match_map, errors, reassoc)  # Associate hit

    return a
        
def register_similar_structures_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or, ListOf, FloatArg, Color8Arg, StringArg
    from chimerax.atomic import ChainArg, StructureArg
    TrimArg = Or(ListOf(EnumOf(['chains', 'sequence', 'ligands'])), BoolArg)

    desc = CmdDesc(
        required = [('hit_name', StringArg)],
        keyword = [('trim', TrimArg),
                   ('align', BoolArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('in_file_history', BoolArg),
                   ('log', BoolArg),
                   ('from_set', StringArg),
                   ],
        synopsis = 'Open a structure from similar structure search results and align to query'
    )
    register('similarstructures open', desc, similar_structures_open, logger=logger)

    desc = CmdDesc(
        required = [('set_name', StringArg)],
        synopsis = 'Close set of similar structure search results'
    )
    register('similarstructures close', desc, similar_structures_close, logger=logger)

    desc = CmdDesc(
        synopsis = 'List names of open similar structure sets'
    )
    register('similarstructures list', desc, similar_structures_list, logger=logger)

    desc = CmdDesc(
        required = [('hit_structure', StructureArg)],
        keyword = [('color', Color8Arg),
                   ('radius', FloatArg),
                   ('halfbond_coloring', BoolArg),
                   ('from_set', StringArg),
                   ],
        synopsis = 'Show pseudobonds between paired residues between a similar structure search result and the query structure'
    )
    register('similarstructures pairing', desc, similar_structures_pairing, logger=logger)

    desc = CmdDesc(
        required = [('hit_structure', StructureArg)],
        keyword = [('from_set', StringArg)],
        synopsis = 'Show sequence alignment from a similar structures search for one hit'
    )
    register('similarstructures seqalign', desc, similar_structures_sequence_alignment, logger=logger)
