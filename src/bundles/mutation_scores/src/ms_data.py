def open_deep_mutational_scan_csv(session, path, chain = None, name = None):
    dms_data = DeepMutationScores(session, path)

    msm = _mutation_scores_manager(session)
    if name is None:
        from os.path import basename, splitext
        name = splitext(basename(path))[0]
    msm.add_scores(name, dms_data)

    if chain:
        dms_data.chain = chain

    nmut = sum(len(mscores) for mscores in dms_data.scores.values())
    dresnums = set(dms_data.scores.keys())
    score_names = ', '.join(dms_data.score_names())
    message = f'Opened deep mutational scan data for {nmut} mutations of {len(dresnums)} residues with score names {score_names}.'
    
    if chain:
        cres = chain.existing_residues
        sresnums = set(r.number for r in cres)
        message += f' Assigned scores to {len(sresnums & dresnums)} of {len(cres)} residues of chain {chain}.'
        mres = len(dresnums - sresnums)
        if mres > 0:
            message += f' Found scores for {mres} residues not present in atomic model.'

    return dms_data, message

class DeepMutationScores:
    def __init__(self, session, csv_path):
        self.session = session
        self.path = csv_path
        from os.path import basename, splitext
        self.name = splitext(basename(csv_path))[0]
        self.headings, self.scores, self.res_types = self.read_deep_mutation_csv(csv_path)
        self._chain = None	# Associated Chain instance
        self._computed_scores = {}	# Map computed score name to ScoreValues instance

    def read_deep_mutation_csv(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        headings = [h.strip() for h in lines[0].split(',')]
        scores = {}
        res_types = {}
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
            if res_num not in scores:
                scores[res_num] = {}
            if res_type2 in scores[res_num]:
                from chimerax.core.errors import UserError
                raise UserError(f'Duplicated hgvs "{hgvs}" at line {i+2}')
            scores[res_num][res_type2] = fields
            res_types[res_num] = res_type
        return headings, scores, res_types

    def score_values(self, score_name, raise_error = True):
        cvalues = self._column_value_list(score_name)
        if cvalues is None:
            values = self.computed_values(score_name)
        else:
            values = ScoreValues(cvalues)
        if raise_error and values is None:
            from chimerax.core.errors import UserError
            raise UserError(f'No score named "{score_name}" in mutation scores {self.name}.')
        return values

    def score_names(self):
        return tuple(self.headings[i] for i in self._numeric_columns())

    def _numeric_columns(self):
        for res_num, rscores in self.scores.items():
            for res_type, fields in rscores.items():
                return tuple(i for i,f in enumerate(fields) if _is_string_float(f) or f == 'NA')
    
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

    def _column_value_list(self, column_name):
        c = self.column_index(column_name)
        if c is None:
            return None
        cvalues = []
        for res_num, rscores in self.scores.items():
            for res_type, fields in rscores.items():
                if fields[c] != 'NA':
                    cvalues.append((res_num, self.res_types[res_num], res_type, float(fields[c])))
        return cvalues
    
    def column_index(self, column_name):
        for i,h in enumerate(self.headings):
            if column_name == h:
                return i
        return None

    def _residue_type_matches(self, residues):
        matches = 0
        mismatches = []
        rtypes = self.res_types
        for r in residues:
            rtype = rtypes.get(r.number)
            if rtype is not None:
                if r.one_letter_code == rtype:
                    matches += 1
                else:
                    mismatches.append(r)
        return matches, mismatches

    def _get_chain(self):
        if self._chain is None or self._chain.structure is None:
            self._chain = self._find_matching_chain()
        return self._chain
    def _set_chain(self, chain):
        self._chain = chain
    chain = property(_get_chain, _set_chain)

    def _find_matching_chain(self):
        from chimerax.atomic import AtomicStructure
        structs = self.session.models.list(type = AtomicStructure)
        for s in structs:
            chains = list(s.chains)
            chains.sort(key = lambda c: c.chain_id)
            for c in chains:
                matches, mismatches = self._residue_type_matches(c.existing_residues)
                if len(mismatches) == 0 and matches > 0:
                    return c
        return None

    
def _is_string_float(f):
    try:
        float(f)
        return True
    except:
        return False

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
    def scores(self, scores_name, allow_abbreviation = False):
        if scores_name is None:
            s = tuple(self._scores.values())[0] if len(self._scores) == 1 else None
        else:
            s = self._scores.get(scores_name)
            if s is None and allow_abbreviation:
                full_names = [name for name in self._scores.keys() if name.startswith(scores_name)]
                if len(full_names) == 1:
                    s = self._scores[full_names[0]]
        return s
    def add_scores(self, scores_name, scores):
        self._scores[scores_name] = scores
    def remove_scores(self, scores_name):
        if scores_name in self._scores:
            del self._scores[scores_name]
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

def mutation_scores(session, scores_name):
    msm = _mutation_scores_manager(session)
    scores = msm.scores(scores_name, allow_abbreviation = True)
    if scores is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No mutation scores named {scores_name}')
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

def mutation_scores_structure(session, chain, scores_name = None):
    scores = mutation_scores(session, scores_name)
    scores.chain = chain
    matches, mismatches = scores._residue_type_matches(chain.existing_residues)
    if mismatches:
        r = mismatches[0]
        session.logger.warning(f'Sequence of chain {chain} does not match mutation scores {scores.name} at {len(mistmatches)} positions, first mistmatch is {r.name}{r.number}')

def mutation_scores_close(session, scores_name = None):
    msm = _mutation_scores_manager(session)
    if scores_name is None:
        for scores_name in msm.names():
            msm.remove_scores(scores_name)
    elif not msm.remove_scores(scores_name):
        from chimerax.core.errors import UserError
        raise UserError(f'No mutation scores named {scores_name}')
    
def register_commands(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg
    from chimerax.atomic import ChainArg
    
    desc = CmdDesc(synopsis = 'List names of sets of mutation scores')
    register('mutationscores list', desc, mutation_scores_list, logger=logger)

    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('scores_name', StringArg)],
        synopsis = 'Associate a structure with a set of mutation scores'
    )
    register('mutationscores structure', desc, mutation_scores_structure, logger=logger)

    desc = CmdDesc(
        optional = [('scores_name', StringArg)],
        synopsis = 'Close sets of mutation scores'
    )
    register('mutationscores close', desc, mutation_scores_close, logger=logger)
