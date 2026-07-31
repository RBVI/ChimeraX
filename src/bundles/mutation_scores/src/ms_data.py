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

from .variants import Variant

from chimerax.core.state import State  # For session saving
class MutationSet(State):
    def __init__(self, name, mutation_scores, chains = None, allow_mismatches = False, path = None):
        self.name = name
        self.path = path
        self.mutation_scores = mutation_scores	# List of MutationScores instances
        self.uniprot_id = None			# Sometimes set for data from MaveDB
        self._associated_chains = []		# Chain instances
        self._associated_residues = []		# List of (res_number, residue)
        self._computed_scores = {}		# Map computed score name to ScoreValues instance

        # Cached values
        self._score_names = None
        self._resnum_to_aa = None

        if chains:
            self.set_associated_chains(chains, allow_mismatches)

    def score_values(self, score_name, include_modifications = False, raise_error = True):
        svalues = []
        for ms in self.mutation_scores:
            if score_name in ms.scores:
                v = ms.variant
                if include_modifications or v.to_aa is not None:
                    svalues.append((v, ms.scores[score_name]))
        if len(svalues) == 0:
            values = self.computed_values(score_name)
        else:
            values = ScoreValues(svalues)
        if raise_error and values is None:
            from chimerax.core.errors import UserError
            raise UserError(f'No score named "{score_name}" in mutation scores {self.name}.')
        return values

    def score_names(self, include_computed = False, include_per_residue = True, exclude_names = []):
        if self._score_names is None:
            names = set()
            for ms in self.mutation_scores:
                names.update(ms.scores.keys())
            self._score_names = tuple(sorted(names))
        snames = self._score_names

        if include_computed:
            cnames = self.computed_values_names()
            if not include_per_residue:
                cnames = [cname for cname in cnames if not self.computed_values(cname).per_residue]
            used_names = set(self._score_names)
            cnames = [name for name in cnames if name not in used_names]
            snames += tuple(sorted(cnames))

        if exclude_names:
            snames = tuple(name for name in snames if name not in exclude_names)

        return snames

    def add_scores(self, mutation_scores):
        self.mutation_scores.extend(mutation_scores)
        self._score_names = None
    def rename_score(self, name, new_name):
        for ms in self.mutation_scores:
            if name in ms.scores:
                value = ms.scores[name]
                del ms.scores[name]
                ms.scores[new_name] = value
        self._score_names = None

    def computed_values(self, score_name):
        return self._computed_scores.get(score_name)
    def set_computed_values(self, score_name, score_values):
        self._computed_scores[score_name] = score_values
    def rename_computed_values(self, score_name, new_score_name):
        score_values = self.computed_values(score_name)
        self.set_computed_values(new_score_name, score_values)
        self.remove_computed_values(score_name)
    def remove_computed_values(self, score_name):
        if score_name in self._computed_scores:
            del self._computed_scores[score_name]
            return True
        return False
    def computed_values_names(self):
        return tuple(self._computed_scores.keys())

    @property
    def number_of_variants(self):
        return len(set(ms.variant for ms in self.mutation_scores))

    def modification_names(self, max_names = 10):
        changes = [ms.variant.change for ms in self.mutation_scores if ms.variant.change]
        change_counts = {name:0 for name in set(changes)}
        for c in changes:
            change_counts[c] += 1
        change_names = list(change_counts.keys())
        change_names.sort(key = lambda n: change_counts[n], reverse=True)
        names = change_names[:max_names]
        names.sort()
        return tuple(names)

    def sequence(self, missing_code = 'X'):
        '''
        Returns a Sequence object starting with residue number 1.
        Missing data will use 1-letter code "X".
        '''
        rnum_to_aa = self.residue_number_to_amino_acid()
        rmax = max(rnum_to_aa.keys())
        seq_list = [missing_code] * rmax
        for rnum, aa in rnum_to_aa.items():
            seq_list[rnum-1] = aa
        seq_chars = ''.join(seq_list)
        from chimerax.atomic import Sequence
        seq = Sequence(characters = seq_chars)
        return seq
        
    def associate_chains(self, chains, add = True):
        new_chains = [c for c in chains if c not in self._associated_chains]
        added_chains = _find_matching_chains(new_chains, self.residue_number_to_amino_acid())
        if not add:
            # Clear current associations
            self._associated_chains.clear()
            self._associated_residues.clear()
            
        self._associated_chains.extend(added_chains)
        ares = [(r.number, r) for chain in added_chains for r in chain.existing_residues]
        self._associated_residues.extend(ares)

        return added_chains

    def associated_chains(self):
        self._remove_deleted_chains()
        return self._associated_chains

    def set_associated_chains(self, chains, allow_mismatches = None, minimum_identity = 0.5, pairing = None,
                              replace = True):
        '''
        Pairing maps chains to match list where a match list is a two tuple of mutation score
        sequence numbers and corresponding chain ungapped positions.
        '''
        if not replace:
            # Unassociate and reassociate chains that are already associated.
            self.remove_associated_chains(chains)

        if allow_mismatches is None:
            allow_mismatches = (pairing is not None)
        achains = []
        ares = []
        accepted_messages = []
        rejected_messages = []
        rnum_to_aa = self.residue_number_to_amino_acid()
        for chain in chains:
            cres = chain.existing_residues
            if pairing is None:
                cres_num = cres.numbers
            else:
                c2m = {crnum: mrnum for mrnum, crnum in pairing[chain]}
                cmnums = [(cr, c2m[crnum]) for crnum,cr in enumerate(chain.residues) if cr and crnum in c2m]
                if len(cmnums) == 0:
                    chain.structure.session.logger.warning(f'No residues of {chain} aligned to mutation score sequence')
                    continue
                from chimerax.atomic import Residues
                cres = Residues([cr for cr,mrnum in cmnums])
                cres_num = [mrnum+1 for cr,mrnum in cmnums]

            matches, mismatches = _residue_type_matches(cres, cres_num, rnum_to_aa)

            if mismatches and not allow_mismatches:
                r, maa = mismatches[0]
                msg = f'Did not associate chain {chain} because sequence does not match at {len(mismatches)} positions, first mismatch is {r.one_letter_code}{r.number}{maa}.  Use the "alignSequences" or "allowMismatches" command options to associate this chain.'
                accept = False
            elif matches < minimum_identity * len(rnum_to_aa):
                msg = (f'Did not associate chain {chain} because only {matches} residues matched, less than {"%.0f"%(100*minimum_identity)} percent of {len(rnum_to_aa)} mutation set residues.')
                accept = False
            elif pairing:
                nalign = matches + len(mismatches)
                if mismatches:
                    mismatch_rnums = ', '.join(f'{r.one_letter_code}{r.number}{maa}' for (r,maa) in mismatches)
                    plural = 'es' if len(mismatches) > 1 else ''
                    mismatch_info = f'{len(mismatches)} amino acid mismatch{plural} ({mismatch_rnums})'
                else:
                    mismatch_info = ' no mismatches'
                msg = f'Associated {nalign} residues of chain {chain} with {mismatch_info}.'
                accept = True
            else:
                msg = f'Associated chain {chain} with {len(mismatches)} mismatches.'
                accept = True
                
            if accept:
                achains.append(chain)
                aligned_res = [(rnum,r) for r,rnum in zip(cres,cres_num) if rnum in rnum_to_aa]
                ares.extend(aligned_res)
                accepted_messages.append(msg)
            else:
                rejected_messages.append(msg)

        if replace:
            self._associated_chains = []
            self._associated_residues = []

        self._associated_chains.extend(achains)
        self._associated_residues.extend(ares)

        if chains:
            log = chains[0].structure.session.logger
            plural = '' if len(accepted_messages) == 1 else 's'
            summary = f'Associated {len(accepted_messages)} chain{plural} to mutation set {self.name}.'
            if accepted_messages:
                log.info(f'{summary}\n' + '\n'.join(accepted_messages))
            elif rejected_messages:
                log.warning(f'{summary}\n\n' + '\n\n'.join(rejected_messages))

    def remove_associated_chains(self, chains):
        cset = set(chains)
        rchains = [c for c in self._associated_chains if c in cset]
        achains = [c for c in self._associated_chains if c not in cset]
        self._associated_chains = achains
        ares = [(rnum,r) for rnum,r in self._associated_residues if r.chain not in cset]
        self._associated_residues = ares
        return rchains

    def _remove_deleted_chains(self):
        deleted = False
        for chain in self._associated_chains:
            if chain.structure is None:
                deleted = True
        if deleted:
            self._associated_chains = [chain for chain in self._associated_chains if chain.structure is not None]
            self._remove_deleted_residues()

    def _remove_deleted_residues(self):
        self._associated_residues = [(rnum, r) for rnum, r in self._associated_residues if not r.deleted]

    def associated_residues(self, res_nums = None):
        rlist = []
        rnums = []
        res_nums_set = set(res_nums) if res_nums is not None else None
        deleted = False
        for rnum, r in self._associated_residues:
            if res_nums is None or rnum in res_nums_set:
                if r.deleted:
                    deleted = True
                else:
                    rlist.append(r)
                    rnums.append(rnum)
        if deleted:
            self._remove_deleted_residues()
            self._remove_deleted_chains()
        from chimerax.atomic import Residues
        res = Residues(rlist)
        return res, rnums

    def residue_number_to_amino_acid(self):
        if self._resnum_to_aa is None:
            self._resnum_to_aa = rnum_to_aa = {}
            for ms in self.mutation_scores:
                v = ms.variant
                if v.residue_number:
                    rnum_to_aa[v.residue_number] = v.from_aa
        return self._resnum_to_aa

    def take_snapshot(self, session, flags):
        self._remove_deleted_chains()
        self._remove_deleted_residues()
        return {'name': self.name,
                'path': self.path,
                'mutation_scores': self.mutation_scores,
                'associated_chains': self._associated_chains,
                'associated_residues': self._associated_residues,
                'computed_scores': self._computed_scores,
                'uniprot_id': self.uniprot_id,
                'version': 1}

    @classmethod
    def restore_snapshot(cls, session, data):
        ms = cls(data['name'], data['mutation_scores'], path = data['path'])
        ms._associated_chains = data.get('associated_chains', [])
        ms._associated_residues = data.get('associated_residues', [])
        if 'chain' in data:
            chain = data['chain']
            ms._associated_chains = [chain]
            ms._associated_residues = [(r.number,r) for r in chain.existing_residues]
        ms._computed_scores = data['computed_scores']
        if 'uniprot_id' in data:
            ms.uniprot_id = data['uniprot_id']
        return ms

def _find_matching_chains(chains, resnum_to_aa):
    mchains = []
    for c in chains:
        cres = c.existing_residues
        matches, mismatches = _residue_type_matches(cres, cres.numbers, resnum_to_aa)
        if len(mismatches) == 0 and matches > 0:
            mchains.append(c)
    return mchains
        
def _residue_type_matches(residues, res_nums, resnum_to_aa):
    matches = 0
    mismatches = []
    for r,rnum in zip(residues,res_nums):
        rtype = resnum_to_aa.get(rnum)
        if rtype is not None:
            if r.one_letter_code == rtype:
                matches += 1
            else:
                mismatches.append((r, rtype))
    return matches, mismatches

def _check_scores_sequence(scores_sequence, rnum_to_aa):
    for rnum, aa in rnum_to_aa.items():
        if rnum > len(scores_sequence) or scores_sequence[rnum-1] not in (aa, 'X'):
            mseq = ['X'] * max(rnum_to_aa.keys())
            for rnum, aa in rnum_to_aa.items():
                mseq[rnum-1] = aa
            mseq = ''.join(mseq)
            from chimerax.core.errors import UserError
            raise UserError(f'Alignment reference sequence "{scores_sequence}" does not match mutation scores sequence "{mseq}" at position {rnum}')
    
class MutationScores(State):
    def __init__(self, variant, scores):
        self.variant = variant
        self.scores = scores	# Map of score name to score value

    def filter(self, score_names):
        scores = {score_name:value for score_name, value in self.scores.items() if score_name in score_names}
        return MutationScores(self.variant, scores)

    def take_snapshot(self, session, flags):
        return {'variant': self.variant,
                'scores': self.scores,
                'version': 2}

    @classmethod
    def restore_snapshot(cls, session, data):
        ver = data['version']
        if ver == 1:
            hgvs_pro = f'p.{data["from_aa"]}{data["residue_number"]}{data["to_aa"]}'
            variant = Variant(hgvs_pro)
        else:
            variant = data['variant']
        ms = cls(variant, data['scores'])
        return ms
    
class ScoreValues(State):
    def __init__(self, mutation_values, per_residue = False):
        self._mutation_values = mutation_values # List of (variant, value)
        self._values_by_residue_number = None	# res_num -> (variant, value)
        self.per_residue = per_residue

    def all_values(self):
        return self._mutation_values

    def count(self):
        return len(self._mutation_values)

    def residue_numbers(self):
        rnums = tuple(set(variant.residue_number for variant, value in self._mutation_values
                          if variant.residue_number is not None))
        return rnums

    def residue_numbers_and_types(self):
        return tuple((rnum, rvals[0][0].from_aa) for rnum, rvals in self.values_by_residue_number.items())

    def residue_value(self, residue_number):
        mvalues = self.mutation_values(residue_number)
        return None if len(mvalues) == 0 else sum(value for variant, value in mvalues)

    def mutation_values(self, residue_number):
        '''Return list of (from_aa, to_aa, value).'''
        res_values = self.values_by_residue_number.get(residue_number, {})
        return res_values

    @property
    def values_by_residue_number(self):
        if self._values_by_residue_number is None:
            self._values_by_residue_number = vbrn = {}
            for variant, value in self._mutation_values:
                rnum = variant.residue_number
                if rnum is not None:
                    if rnum in vbrn:
                        vbrn[rnum].append((variant,value))
                    else:
                        vbrn[rnum] = [(variant,value)]
        return self._values_by_residue_number
        
    def value_range(self):
        values = [value for variant,value in self._mutation_values]
        return min(values), max(values)

    def synonymous_mean_and_sdev(self):
        values = [value for variant, value in self._mutation_values if variant.is_synonymous]
        if len(values) == 0:
            return None, None
        from numpy import mean, std
        return mean(values), std(values)

    def mean_and_sdev(self):
        values = [value for variant, value in self._mutation_values]
        if len(values) == 0:
            return None, None
        from numpy import mean, std
        return mean(values), std(values)

    def subtract_fit(self, score_values):
        values = subtract_fit_values(self.all_values(), score_values.all_values())
        return ScoreValues(values)
    
    def take_snapshot(self, session, flags):
        return {'mutation_values': self._mutation_values,
                'per_residue': self.per_residue,
                'version': 1}

    @classmethod
    def restore_snapshot(cls, session, data):
        sv = cls(data['mutation_values'], per_residue = data['per_residue'])
        return sv

def subtract_fit_values(cvalues, svalues):
    smap = {variant:value for variant,value in svalues}
    x = []
    y = []
    for variant, value in cvalues:
        svalue = smap.get(variant)
        if svalue is not None:
            x.append(svalue)
            y.append(value)
    from numpy import polyfit
    degree = 1
    m,b = polyfit(x, y, degree)
    sfvalues = [(variant, value - (m*smap[variant] + b))
                for variant, value in cvalues if variant in smap]
    return sfvalues

from chimerax.core.state import StateManager  # For session saving
class MutationScoresManager(StateManager):
    def __init__(self, session):
        self._session = session
        self._scores = {}	# Maps name to MutationSet

        triggers = session.triggers
        create_mutation_set_add_remove_triggers(triggers)

        # Update associated structure
        triggers.add_handler('add models', self._structure_opened)
        triggers.add_handler('remove models', self._structure_closed)

    def mutation_set(self, mutation_set_name):
        return self._scores.get(mutation_set_name)
    def scores(self, mutation_set_name, allow_abbreviation = False):
        if mutation_set_name is None:
            s = tuple(self._scores.values())[0] if len(self._scores) == 1 else None
        else:
            s = self._scores.get(mutation_set_name)
            if s is None and allow_abbreviation:
                full_names = [name for name in self._scores.keys() if name.startswith(mutation_set_name)]
                if len(full_names) == 1:
                    s = self._scores[full_names[0]]
        return s
    def add_scores(self, mutation_set):
        if mutation_set.name in self._scores:
            mutation_set.name = self._unique_mutation_set_name(mutation_set.name)
        self._scores[mutation_set.name] = mutation_set
        self._session.triggers.activate_trigger('mutation set added', mutation_set)
        chains = mutation_set.associate_chains(_all_chains(self._session))
        if chains:
            self._session.logger.info(self._associate_message(mutation_set, chains))
    def remove_scores(self, mutation_set_name):
        mutation_set = self._scores.get(mutation_set_name)
        if mutation_set:
            del self._scores[mutation_set_name]
            self._session.triggers.activate_trigger('mutation set removed', mutation_set)
            return True
        return False
    def all_scores(self):
        return tuple(self._scores.values())
    def names(self):
        return tuple(self._scores.keys())
    def _unique_mutation_set_name(self, name):
        new_name = name
        suffix = 1
        while new_name in self._scores:
            suffix += 1
            new_name = f'{name} {suffix}'
        return new_name
    def _structure_opened(self, trigger_name, models):
        self._update_associated_chains(models, 'add')
    def _structure_closed(self, trigger_name, models):
        self._update_associated_chains(models, 'remove')
    def _update_associated_chains(self, models, add_or_remove):
        if self.all_scores():
            chains = _structure_chains(models)
            if chains:
                messages = []
                for mset in self.all_scores():
                    if add_or_remove == 'add':
                        achains = mset.associate_chains(chains)
                        if achains:
                            messages.append(self._associate_message(mset, achains))
                    elif add_or_remove == 'remove':
                        mset.remove_associated_chains(chains)
                if messages:
                    log = models[0].session.logger
                    log.info('\n'.join(messages))
    def _associate_message(self, mset, chains):
        from chimerax.atomic import concise_chain_spec
        cspec = concise_chain_spec(chains)
        return f'Associated {len(chains)} chains {cspec} with mutations {mset.name}'

    def take_snapshot(self, session, flags):
        return {'scores': self._scores,
                'version': 1}
    @classmethod
    def restore_snapshot(cls, session, data):
        msm = mutation_scores_manager(session)
        for mset in data['scores'].values():
            msm.add_scores(mset)
        return msm
    def reset_state(self, session):
        self._scores.clear()

def _structure_chains(models):
    from chimerax.atomic import AtomicStructure
    structures = [m for m in models if isinstance(m, AtomicStructure)]
    chains = []
    for s in structures:
        chains.extend(list(s.chains))
    return chains

def _all_chains(session):
    chains = []
    from chimerax.atomic import AtomicStructure
    for s in session.models.list(type = AtomicStructure):
        chains.extend(s.chains)
    return chains

def create_mutation_set_add_remove_triggers(triggers, added_callback = None, removed_callback = None):
    if not triggers.has_trigger('mutation set added'):
        triggers.add_trigger('mutation set added')
    if not triggers.has_trigger('mutation set removed'):
        triggers.add_trigger('mutation set removed')
    if added_callback:
        triggers.add_handler('mutation set added', added_callback)
    if removed_callback:
        triggers.add_handler('mutation set removed', removed_callback)
        
def mutation_scores_manager(session, create = True):
    msm = getattr(session, 'mutation_scores_manager', None)
    if msm is None and create:
        session.mutation_scores_manager = msm = MutationScoresManager(session)
    return msm

def mutation_scores(session, mutation_set, raise_error = True):
    msm = mutation_scores_manager(session)
    scores = msm.scores(mutation_set, allow_abbreviation = True)
    if raise_error and scores is None:
        msg = 'No mutation scores found' if mutation_set is None else f'No mutation scores named {mutation_set}'
        from chimerax.core.errors import UserError
        raise UserError(msg)
    return scores

def mutation_all_scores(session):
    msm = mutation_scores_manager(session)
    return msm.all_scores()
    
def mutation_scores_list(session):
    msm = mutation_scores_manager(session)
    score_sets = msm.all_scores()
    sets = '\n'.join(f'{scores.name} ({", ".join(scores.score_names())})' for scores in score_sets)
    session.logger.info(f'{len(score_sets)} mutation score sets\n{sets}')
    return msm.names()
    
def mutation_scores_names(session):
    msm = mutation_scores_manager(session)
    return msm.names()

def mutation_scores_structure(session, chains = None, add = None, remove = None,
                              allow_mismatches = None, minimum_percent_identity = 50,
                              align_sequences = None, mutation_set = None):
    mset = mutation_scores(session, mutation_set)

    if (chains is None and add is None and remove is None) or chains == 'list':
        _report_associated_chains(mset, session.logger)
        return

    if chains == 'clear':
        mset.set_associated_chains([])
        return

    if remove:
        rchains = mset.remove_associated_chains(remove)
        if rchains:
            from chimerax.atomic import concise_chain_spec
            cspec = concise_chain_spec(rchains)
            session.logger.info(f'Unassociated chains {cspec} from mutation set {mset.name}')

    if chains is None and add is None:
        return

    chains = list(chains) if chains else []
    if add:
        for chain in add:
            chains.append(chain)

    if isinstance(align_sequences, bool) and align_sequences:
        # Use mutation data sequence.
        align_sequences = mset.sequence()
    from chimerax.atomic import Sequence
    from chimerax.seqalign.alignment import Alignment
    if isinstance(align_sequences, Sequence):
        pairing = _sequence_pairing(align_sequences, chains)
    elif isinstance(align_sequences, Alignment):
        if len(align_sequences.seqs) == 1:
            pairing = _sequence_pairing(align_sequences.seqs[0], chains)	# Single sequence specified
        else:
            pairing = _sequence_pairing_from_alignment(mset, chains, align_sequences)
    else:
        pairing = None
        
    mset.set_associated_chains(chains, allow_mismatches = allow_mismatches,
                               minimum_identity = minimum_percent_identity/100,
                               pairing = pairing, replace = not add)

def _report_associated_chains(mset, logger):
    chains = mset.associated_chains()
    from chimerax.atomic import concise_chain_spec
    cspec = concise_chain_spec(chains)
    logger.status(f'Mutation set {mset.name} has {len(chains)} associated chains {cspec}.', log=True)

def _sequence_pairing(mseq, chains):
    pairing = {}
    from chimerax.alignment_algs.NeedlemanWunsch import nw
    for chain in chains:
        score, match_list = nw(mseq, chain)
        pairing[chain] = match_list
    return pairing

def _sequence_pairing_from_alignment(mset, chains, alignment):
    # Make sure first sequence of alignment matches mutation data sequence.
    rnum_to_aa = mset.residue_number_to_amino_acid()
    mseq = alignment.seqs[0]
    _check_scores_sequence(mseq.ungapped(), rnum_to_aa)

    pairing = {}
    for chain in chains:
        if chain not in alignment.associations:
            from chimerax.core.errors import UserError
            raise UserError(f'Chain {chain} is not present in alignment {alignment.ident}')
        cseq = alignment.associations[chain]
        match_list = [(mseq.gapped_to_ungapped(gi), cseq.gapped_to_ungapped(gi)) for gi in range(len(mseq))]
        pairing[chain] = match_list

    return pairing

def mutation_scores_merge(session, mutation_set, into, scores = None):
    mset = mutation_scores(session, mutation_set)
    into_mset = mutation_scores(session, into)

    # Check if replacing score names.
    score_names = set(mset.score_names())
    if scores is not None:
        only_these_scores = [score_name.strip() for score_name in scores.split(',')]
        score_names = score_names.intersection(only_these_scores)

    common_names = score_names.intersection(into_mset.score_names())
    if common_names:
        from chimerax.core.errors import UserError
        raise UserError(f'Cannot replace existing score names: {", ".join(common_names)}')

    # Check if sequences match.
    _mutation_sequences_match(mset, into_mset, raise_error = True)

    if scores:
        mut_scores = [ms.filter(only_these_scores) for ms in mset.mutation_scores]
    else:
        mut_scores = mset.mutation_scores
    into_mset.add_scores(mut_scores)

def _mutation_sequences_match(mset1, mset2, raise_error = False):
    ra1 = mset1.residue_number_to_amino_acid()
    ra2 = mset2.residue_number_to_amino_acid()
    for rnum, aa in ra1.items():
        if rnum in ra2 and ra2[rnum] != aa:
            if raise_error:
                from chimerax.core.errors import UserError
                raise UserError(f'Mutation set {mset.name} has residue {aa}{rnum} that conflicts with mutation set {mset2.name} which has {aa}{ra2[rnum]}')
            else:
                return False
    return True
    
def mutation_scores_close(session, mutation_set = None):
    msm = mutation_scores_manager(session)
    if mutation_set is None:
        for mutation_set in msm.names():
            msm.remove_scores(mutation_set)
            _close_plots(session, mutation_set)
    elif msm.remove_scores(mutation_set):
        _close_plots(session, mutation_set)
    else:
        from chimerax.core.errors import UserError
        raise UserError(f'No mutation scores named {mutation_set}')

def _close_plots(session, mutation_set_name):
    for tool in session.tools.list():
        if getattr(tool, 'mutation_set_name', None) == mutation_set_name:
            tool.delete()

def register_commands(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, FloatArg, Or, EnumOf
    from chimerax.atomic import UniqueChainsArg, SequenceArg
    from chimerax.seqalign import AlignmentArg
    
    desc = CmdDesc(synopsis = 'List names of sets of mutation scores')
    register('mutationscores list', desc, mutation_scores_list, logger=logger)

    desc = CmdDesc(
        optional = [('chains', Or(EnumOf(('list', 'clear')), UniqueChainsArg))],
        keyword = [('add', UniqueChainsArg),
                   ('remove', UniqueChainsArg),
                   ('allow_mismatches', BoolArg),
                   ('minimum_percent_identity', FloatArg),
                   ('align_sequences', Or(BoolArg, SequenceArg, AlignmentArg)),
                   ('mutation_set', StringArg)],
        synopsis = 'Associate a structure with a set of mutation scores'
    )
    register('mutationscores structure', desc, mutation_scores_structure, logger=logger)

    desc = CmdDesc(
        required = [('mutation_set', StringArg)],
        keyword = [('into', StringArg),
                   ('scores', StringArg)],
        required_arguments = ['into'],
        synopsis = 'Merge one mutation set into another'
    )
    register('mutationscores merge', desc, mutation_scores_merge, logger=logger)

    desc = CmdDesc(
        optional = [('mutation_set', StringArg)],
        synopsis = 'Close sets of mutation scores'
    )
    register('mutationscores close', desc, mutation_scores_close, logger=logger)
