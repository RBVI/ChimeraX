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
class MutationSet(State):
    def __init__(self, name, mutation_scores, chains = None, allow_mismatches = False, path = None):
        self.name = name
        self.path = path
        self.mutation_scores = mutation_scores	# List of MutationScores instances
        self._associated_chains = []		# Chain instances
        self._associated_residues = []		# List of (res_number, residue)
        self._computed_scores = {}	# Map computed score name to ScoreValues instance

        # Cached values
        self._score_names = None
        self._resnum_to_aa = None

        if chains:
            self.set_associated_chains(chains, allow_mismatches)

    def score_values(self, score_name, raise_error = True):
        svalues = [(ms.residue_number, ms.from_aa, ms.to_aa, ms.scores[score_name])
                   for ms in self.mutation_scores if score_name in ms.scores]
        if len(svalues) == 0:
            values = self.computed_values(score_name)
        else:
            values = ScoreValues(svalues)
        if raise_error and values is None:
            from chimerax.core.errors import UserError
            raise UserError(f'No score named "{score_name}" in mutation scores {self.name}.')
        return values

    def score_names(self):
        if self._score_names is None:
            names = set()
            for ms in self.mutation_scores:
                names.update(ms.scores.keys())
            self._score_names = tuple(sorted(names))
        return self._score_names

    def add_scores(self, mutation_scores):
        self.mutation_scores.extend(mutation_scores)
        self._score_names = None
        
    def computed_values(self, score_name):
        return self._computed_scores.get(score_name)
    def set_computed_values(self, score_name, score_values):
        self._computed_scores[score_name] = score_values
    def remove_computed_values(self, score_name):
        if score_name in self._computed_scores:
            del self._computed_scores[score_name]
            return True
        return False
    def computed_values_names(self):
        return tuple(self._computed_scores.keys())

    def associate_chains(self, session, report = True):
        if len(self.associated_chains()) == 0:
            chains = _find_matching_chains(session, self.residue_number_to_amino_acid())
            self._associated_chains = chains
            self._associated_residues = [(r.number, r) for chain in chains for r in chain.existing_residues]
            if report and len(chains) > 0:
                _report_associated_chains(self, session.logger)
        return self._associated_chains

    def associated_chains(self):
        self._remove_deleted_chains()
        return self._associated_chains

    def set_associated_chains(self, chains, allow_mismatches = False, alignment = None):
        achains = []
        ares = []
        rnum_to_aa = self.residue_number_to_amino_acid()
        if alignment:
            _check_scores_sequence(alignment.seqs[0].ungapped(), rnum_to_aa)
        for chain in chains:
            cres = chain.existing_residues
            if alignment:
                renum = _alignment_renumbering(alignment, chain)
                rnums = {r:rnum for r, rnum in zip(chain.residues, renum) if r is not None and rnum is not None}
                from chimerax.atomic import Residues
                cres = Residues([r for r in cres if r in rnums])
                if len(cres) == 0:
                    chain.structure.session.logger.warning(f'No residues of {chain} aligned to mutation score sequence')
                    continue
                cres_num = [rnums[r] for r in cres]
            else:
                cres_num = cres.numbers
            matches, mismatches = _residue_type_matches(cres, cres_num, rnum_to_aa)
            if matches > 0 and (len(mismatches) == 0 or allow_mismatches or alignment):
                achains.append(chain)
                ares.extend([(rnum,r) for r,rnum in zip(cres,cres_num) if rnum in rnum_to_aa])
            if alignment:
                msg = (f'Aligned {len(rnums)} residues of sequence {chain} to mutation scores {self.name} with {len(mismatches)} amino acid mismatches. and {len(cres)} aligned residues have coordinates and mutation scores.')
                chain.structure.session.logger.info(msg)
            elif mismatches and not allow_mismatches:
                r = mismatches[0]
                msg = f'Did not associate chain {chain} because sequence does not match mutation set {self.name} at {len(mismatches)} positions, first mistmatch is {r.name} {r.number}.  Use the "alignSequences" or "allowMismatches" command options to associate this chain.'
                chain.structure.session.logger.warning(msg)
            else:
                msg = f'Associated chain {chain} to mutation scores {self.name} with {len(mismatches)} mismatches.'
                chain.structure.session.logger.info(msg)

        self._associated_chains = achains
        self._associated_residues = ares

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

    def associated_residues(self, res_nums):
        rlist = []
        rnums = []
        res_nums_set = set(res_nums)
        deleted = False
        for rnum, r in self._associated_residues:
            if rnum in res_nums_set:
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
            self._resnum_to_aa = {ms.residue_number:ms.from_aa for ms in self.mutation_scores}
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
        return ms

def _find_matching_chains(session, resnum_to_aa):
    from chimerax.atomic import AtomicStructure
    structs = session.models.list(type = AtomicStructure)
    mchains = []
    for s in structs:
        chains = list(s.chains)
        chains.sort(key = lambda c: c.chain_id)
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
                mismatches.append(r)
    return matches, mismatches

def _check_scores_sequence(scores_sequence, rnum_to_aa):
    for rnum, aa in rnum_to_aa.items():
        if rnum > len(scores_sequence) or scores_sequence[rnum-1] != aa:
            mseq = ['X'] * max(rnum_to_aa.keys())
            for rnum, aa in rnum_to_aa.items():
                mseq[rnum-1] = aa
            mseq = ''.join(mseq)
            from chimerax.core.errors import UserError
            raise UserError(f'Alignment reference sequence "{scores_sequence}" does not match mutation scores sequence "{mseq}" at position {rnum}')

def _alignment_renumbering(alignment, chain):
    if chain not in alignment.associations:
        from chimerax.core.errors import UserError
        raise UserError(f'Chain {chain} is not present in alignment {alignment.ident}')

    cseq = alignment.associations[chain]
    rseq = alignment.seqs[0]
    renum = [None] * len(cseq.ungapped())
    for gi in range(len(rseq)):
        ci = cseq.gapped_to_ungapped(gi)
        ri = rseq.gapped_to_ungapped(gi)
        if ci is not None and ri is not None:
            renum[ci] = ri + 1

    return renum
    
class MutationScores(State):
    def __init__(self, residue_number, from_aa, to_aa, scores):
        self.residue_number = residue_number
        self.from_aa = from_aa
        self.to_aa = to_aa
        self.scores = scores

    def take_snapshot(self, session, flags):
        return {'residue_number': self.residue_number,
                'from_aa': self.from_aa,
                'to_aa': self.to_aa,
                'scores': self.scores,
                'version': 1}

    @classmethod
    def restore_snapshot(cls, session, data):
        ms = cls(data['residue_number'], data['from_aa'], data['to_aa'], data['scores'])
        return ms

class ScoreValues(State):
    def __init__(self, mutation_values, per_residue = False):
        self._mutation_values = mutation_values # List of (res_num, from_aa, to_aa, value)
        self._values_by_residue_number = None	# res_num -> (from_aa, to_aa, value)
        self.per_residue = per_residue

    def all_values(self):
        return self._mutation_values

    def count(self):
        return len(self._mutation_values)

    def residue_numbers(self):
        return tuple(self.values_by_residue_number.keys())

    def residue_numbers_and_types(self):
        return tuple((rnum, rvals[0][0]) for rnum, rvals in self.values_by_residue_number.items())

    def residue_value(self, residue_number):
        mvalues = self.mutation_values(residue_number)
        return None if len(mvalues) == 0 else sum(value for from_aa, to_aa, value in mvalues)

    def mutation_values(self, residue_number):
        '''Return list of (from_aa, to_aa, value).'''
        res_values = self.values_by_residue_number.get(residue_number, {})
        return res_values

    @property
    def values_by_residue_number(self):
        if self._values_by_residue_number is None:
            self._values_by_residue_number = vbrn = {}
            for val in self._mutation_values:
                if val[0] in vbrn:
                    vbrn[val[0]].append(val[1:])
                else:
                    vbrn[val[0]] = [val[1:]]
        return self._values_by_residue_number
        
    def value_range(self):
        values = [val[3] for val in self._mutation_values]
        return min(values), max(values)

    def take_snapshot(self, session, flags):
        return {'mutation_values': self._mutation_values,
                'per_residue': self.per_residue,
                'version': 1}

    @classmethod
    def restore_snapshot(cls, session, data):
        sv = cls(data['mutation_values'], per_residue = data['per_residue'])
        return sv

from chimerax.core.state import StateManager  # For session saving
class MutationScoresManager(StateManager):
    def __init__(self):
        self._scores = {}	# Maps name to MutationSet

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
        self._scores[mutation_set.name] = mutation_set
    def remove_scores(self, mutation_set_name):
        if mutation_set_name in self._scores:
            del self._scores[mutation_set_name]
            return True
        return False
    def all_scores(self):
        return tuple(self._scores.values())
    def names(self):
        return tuple(self._scores.keys())
    def take_snapshot(self, session, flags):
        return {'scores': self._scores,
                'version': 1}
    @classmethod
    def restore_snapshot(cls, session, data):
        msm = cls()
        msm._scores = data['scores']
        return msm
    def reset_state(self, session):
        self._scores.clear()

def mutation_scores_manager(session, create = True):
    msm = getattr(session, 'mutation_scores_manager', None)
    if msm is None and create:
        session.mutation_scores_manager = msm = MutationScoresManager()
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

def mutation_scores_structure(session, chains = None, allow_mismatches = False, align_sequences = None,
                              mutation_set = None):
    mset = mutation_scores(session, mutation_set)
    if chains is None:
        _report_associated_chains(mset, session.logger)
    else:
        from chimerax.atomic import Sequence
        from chimerax.seqalign.alignment import Alignment
        if isinstance(align_sequences, Sequence):
            alignment = _sequence_alignment(align_sequences, chains)
        elif isinstance(align_sequences, Alignment):
            if len(align_sequences.seqs) == 1:
                alignment = _sequence_alignment(align_sequences.seqs[0], chains)	# Single sequence specified
            else:
                alignment = align_sequences
        else:
            alignment = None
        mset.set_associated_chains(chains, allow_mismatches = allow_mismatches, alignment = alignment)

def _report_associated_chains(mset, logger):
    chains = [chain for chain in mset.associated_chains()]
    from chimerax.atomic import concise_chain_spec
    cspec = concise_chain_spec(chains)
    logger.status(f'Mutation set {mset.name} has {len(chains)} associated chains {cspec}.', log=True)

def _sequence_alignment(scores_sequence, chains):
    seqs = set()
    seqs.add(scores_sequence.characters)
    align_chains = []
    for chain in chains:
        if chain.characters not in seqs:
            seqs.add(chain.characters)
            align_chains.append(chain)

    if len(align_chains) == 0:
        if len(chains) > 0:
            logger = chains[0].structure.session.logger
            from chimerax.atomic import concise_chain_spec
            cspec = concise_chain_spec(chains)
            logger.info(f'All chains {cspec} have the same sequence as the mutation scores so no alignment is needed')
        return None

    session = chains[0].structure.session
    aseqs = ','.join(ac.characters for ac in align_chains)
    from chimerax.core.commands import run
    alignment = run(session, f'sequence align {scores_sequence.characters},{aseqs}')
    alignment.seqs[0].name = 'scores'
    for i,c in enumerate(align_chains):
        alignment.seqs[i+1].name = f'{c.structure.name} {c.chain_id}'
    return alignment

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
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, Or
    from chimerax.atomic import UniqueChainsArg, SequenceArg
    from chimerax.seqalign import AlignmentArg
    
    desc = CmdDesc(synopsis = 'List names of sets of mutation scores')
    register('mutationscores list', desc, mutation_scores_list, logger=logger)

    desc = CmdDesc(
        optional = [('chains', UniqueChainsArg)],
        keyword = [('allow_mismatches', BoolArg),
                   ('align_sequences', Or(SequenceArg, AlignmentArg)),
                   ('mutation_set', StringArg)],
        synopsis = 'Associate a structure with a set of mutation scores'
    )
    register('mutationscores structure', desc, mutation_scores_structure, logger=logger)

    desc = CmdDesc(
        optional = [('mutation_set', StringArg)],
        synopsis = 'Close sets of mutation scores'
    )
    register('mutationscores close', desc, mutation_scores_close, logger=logger)
