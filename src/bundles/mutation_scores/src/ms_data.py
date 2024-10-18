def open_mutation_scores_csv(session, path, chain = None, name = None):
    mset = _read_mutation_scores_csv(path)

    msm = _mutation_scores_manager(session)
    if name is None:
        from os.path import basename, splitext
        name = splitext(basename(path))[0]
    mset.name = name
    msm.add_scores(name, mset)

    if chain:
        mset.chain = chain

    nmut = len(mset.mutation_scores)
    dresnums = set(mset.residue_number_to_amino_acid().keys())
    score_names = ', '.join(mset.score_names())
    message = f'Opened deep mutational scan data for {nmut} mutations of {len(dresnums)} residues with score names {score_names}.'
    
    if chain:
        cres = chain.existing_residues
        sresnums = set(r.number for r in cres)
        message += f' Assigned scores to {len(sresnums & dresnums)} of {len(cres)} residues of chain {chain}.'
        mres = len(dresnums - sresnums)
        if mres > 0:
            message += f' Found scores for {mres} residues not present in atomic model.'

    return mset, message

def _read_mutation_scores_csv(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    headings = [h.strip() for h in lines[0].split(',')]
    mscores = []
    mut = set()
    for i, line in enumerate(lines[1:]):
        if line.strip() == '':
            continue	# Ignore blank lines
        fields = line.split(',')
        if len(fields) != len(headings):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has wrong number of comma-separated fields, got {len(fields)}, but there are {len(headings)} headings')
        hgvs = fields[0]
        if not hgvs.startswith('p.(') or not hgvs.endswith(')'):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has hgvs field "{hgvs}" not starting with "p.(" and ending with ")"')
        if 'del' in hgvs or 'ins' in hgvs or '_' in hgvs:
            continue
        res_type = hgvs[3]
        res_num = int(hgvs[4:-2])
        res_type2 = hgvs[-2]
        if (res_num, res_type, res_type2) in mut:
            from chimerax.core.errors import UserError
            raise UserError(f'Duplicated mutation "{hgvs}" at line {i+2}')
        mut.add((res_num, res_type, res_type2))
        scores = _parse_scores(headings, fields)
        mscores.append(MutationScores(res_num, res_type, res_type2, scores))

    from os.path import basename, splitext
    name = splitext(basename(path))[0]
    mset = MutationSet(name, mscores, path = path)

    return mset

def _parse_scores(headings, fields):
    scores = {}
    for h,f in zip(headings[1:], fields[1:]):
        try:
            scores[h] = float(f)
        except ValueError:
            continue
    return scores
        
class MutationSet:
    def __init__(self, name, mutation_scores, chain = None, path = None):
        self.name = name
        self.path = path
        self.mutation_scores = mutation_scores
        self._chain = chain		# Associated Chain instance
        self._computed_scores = {}	# Map computed score name to ScoreValues instance

        # Cached values
        self._score_names = None
        self._resnum_to_aa = None

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

    def _get_chain(self):
        if self._chain is not None and self._chain.structure is None:
            self._chain = None
        return self._chain
    def _set_chain(self, chain):
        self._chain = chain
    chain = property(_get_chain, _set_chain)

    def find_matching_chain(self, session):
        self._chain = _find_matching_chain(session, self.residue_number_to_amino_acid())
        return self._chain

    def residue_number_to_amino_acid(self):
        if self._resnum_to_aa is None:
            self._resnum_to_aa = {ms.residue_number:ms.from_aa for ms in self.mutation_scores}
        return self._resnum_to_aa

def _find_matching_chain(session, resnum_to_aa):
    from chimerax.atomic import AtomicStructure
    structs = session.models.list(type = AtomicStructure)
    for s in structs:
        chains = list(s.chains)
        chains.sort(key = lambda c: c.chain_id)
        for c in chains:
            matches, mismatches = _residue_type_matches(c.existing_residues, resnum_to_aa)
            if len(mismatches) == 0 and matches > 0:
                return c
    return None
        
def _residue_type_matches(residues, resnum_to_aa):
    matches = 0
    mismatches = []
    for r in residues:
        rtype = resnum_to_aa.get(r.number)
        if rtype is not None:
            if r.one_letter_code == rtype:
                matches += 1
            else:
                mismatches.append(r)
    return matches, mismatches

class MutationScores:
    def __init__(self, residue_number, from_aa, to_aa, scores):
        self.residue_number = residue_number
        self.from_aa = from_aa
        self.to_aa = to_aa
        self.scores = scores

class ScoreValues:
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

class MutationScoresManager:
    def __init__(self):
        self._scores = {}
    def scores(self, mutation_set, allow_abbreviation = False):
        if mutation_set is None:
            s = tuple(self._scores.values())[0] if len(self._scores) == 1 else None
        else:
            s = self._scores.get(mutation_set)
            if s is None and allow_abbreviation:
                full_names = [name for name in self._scores.keys() if name.startswith(mutation_set)]
                if len(full_names) == 1:
                    s = self._scores[full_names[0]]
        return s
    def add_scores(self, mutation_set, scores):
        self._scores[mutation_set] = scores
    def remove_scores(self, mutation_set):
        if mutation_set in self._scores:
            del self._scores[mutation_set]
            return True
        return False
    def all_scores(self):
        return tuple(self._scores.values())
    def names(self):
        return tuple(self._scores.keys())

def _mutation_scores_manager(session, create = True):
    msm = getattr(session, '_mutation_scores_manager', None)
    if msm is None and create:
        session._mutation_scores_manager = msm = MutationScoresManager()
    return msm

def mutation_scores(session, mutation_set):
    msm = _mutation_scores_manager(session)
    scores = msm.scores(mutation_set, allow_abbreviation = True)
    if scores is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No mutation scores named {mutation_set}')
    return scores

def mutation_all_scores(session):
    msm = _mutation_scores_manager(session)
    return msm.all_scores()
    
def mutation_scores_list(session):
    msm = _mutation_scores_manager(session)
    score_sets = msm.all_scores()
    sets = '\n'.join(f'{scores.name} ({", ".join(scores.score_names())})' for scores in score_sets)
    session.logger.info(f'{len(score_sets)} mutation score sets\n{sets}')
    return msm.names()

def mutation_scores_structure(session, chain, mutation_set = None):
    scores = mutation_scores(session, mutation_set)
    scores.chain = chain
    matches, mismatches = _residue_type_matches(chain.existing_residues, scores.residue_number_to_amino_acid())
    if mismatches:
        r = mismatches[0]
        session.logger.warning(f'Sequence of chain {chain} does not match mutation scores {scores.name} at {len(mistmatches)} positions, first mistmatch is {r.name}{r.number}')

def mutation_scores_close(session, mutation_set = None):
    msm = _mutation_scores_manager(session)
    if mutation_set is None:
        for mutation_set in msm.names():
            msm.remove_scores(mutation_set)
    elif not msm.remove_scores(mutation_set):
        from chimerax.core.errors import UserError
        raise UserError(f'No mutation scores named {mutation_set}')
    
def register_commands(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg
    from chimerax.atomic import ChainArg
    
    desc = CmdDesc(synopsis = 'List names of sets of mutation scores')
    register('mutationscores list', desc, mutation_scores_list, logger=logger)

    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('mutation_set', StringArg)],
        synopsis = 'Associate a structure with a set of mutation scores'
    )
    register('mutationscores structure', desc, mutation_scores_structure, logger=logger)

    desc = CmdDesc(
        optional = [('mutation_set', StringArg)],
        synopsis = 'Close sets of mutation scores'
    )
    register('mutationscores close', desc, mutation_scores_close, logger=logger)
