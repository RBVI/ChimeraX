def open_deep_mutational_scan_csv(session, path, chain = None):
    dms_data = DeepMutationScores(path)

    if chain is None:
        chain = _matching_chain(session, dms_data)
        if chain is None:
            from chimerax.core.errors import UserError
            raise UserError(f'Opening a deep mutational scan data requires specifying the structure to associate. If there is one open structure it will be associated, otherwise use the open command structure option, for example\n\n\topen {path} format dms chain #2/B')

    chain._deep_mutation_data = dms_data

    cres = chain.existing_residues
    sresnums = set(r.number for r in cres)
    dresnums = set(dms_data.scores.keys())
    score_names = ', '.join(dms_data.score_column_names)
    message = f'Opened deep mutational scan data for {len(dresnums)} residues, assigned to {len(sresnums & dresnums)} of {len(cres)} chain {chain} residues, found data for {len(dresnums - sresnums)} residues not present in chain, score column names {score_names}.'

    return dms_data, message

def _matching_chain(session, dms_data):
    from chimerax.atomic import AtomicStructure
    structs = session.models.list(type = AtomicStructure)
    for s in structs:
        for c in s.chains:
            matches, mismatches = dms_data.residue_type_matches(c.existing_residues)
            if mismatches == 0 and matches > 0:
                return c
    return None

class DeepMutationScores:
    def __init__(self, csv_path):
        self.path = csv_path
        from os.path import basename
        self.name = basename(csv_path)
        self.headings, self.scores, self.res_types = self.read_deep_mutation_csv(csv_path)

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

    def column_values(self, column_name, subtract_fit = None):
        cvalues = self._column_value_list(column_name)

        if subtract_fit is not None:
            svalues = self._column_value_list(subtract_fit)
            cvalues = _subtract_fit_values(cvalues, svalues)
            
        return ColumnValues(cvalues)

    def _column_value_list(self, column_name):
        cvalues = []
        c = self.column_index(column_name)
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

    def residue_type_matches(self, residues):
        matches = mismatches = 0
        rtypes = self.res_types
        for r in residues:
            rtype = rtypes.get(r.number)
            if rtype is not None:
                if r.one_letter_code == rtype:
                    matches += 1
                else:
                    mismatches += 1
        return matches, mismatches

    @property
    def score_column_names(self):
        return [h for h in self.headings if 'score' in h]

class ColumnValues:
    def __init__(self, mutation_values):
        self._mutation_values = mutation_values # List of (res_num, from_aa, to_aa, value)
        self._values_by_residue_number = None	# res_num -> (from_aa, to_aa, value)

    def all_values(self):
        return self._mutation_values

    def residue_numbers(self):
        return tuple(self.values_by_residue_number.keys())
    
    # Allowed value_type in residue_value() function.
    residue_value_types = ('sum', 'sum_absolute', 'synonymous')
        
    def residue_value(self, residue_number, value_type='sum_absolute', above=None, below=None):
        value = None
        dms_values = self.mutation_values(residue_number)
        if dms_values:
            if value_type == 'sum_absolute' or value_type == 'sum':
                values = [value for from_aa, to_aa, value in dms_values
                          if ((above is None and below is None)
                              or (above is not None and value >= above)
                              or (below is not None and value <= below))]
                if value_type == 'sum_absolute':
                    values = [abs(v) for v in values]
                value = sum(values) if values else None
            elif value_type == 'synonymous':
                values = [value for from_aa, to_aa, value in dms_values if to_aa == from_aa]
                if values:
                    value = values[0]
        return value

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

def _subtract_fit_values(cvalues, svalues):
    smap = {(res_num,from_aa,to_aa):value for res_num, from_aa, to_aa, value in svalues}
    x = []
    y = []
    for res_num, from_aa, to_aa, value in cvalues:
        svalue = smap.get((res_num,from_aa,to_aa))
        if svalue is not None:
            x.append(svalue)
            y.append(value)
    from numpy import polyfit
    degree = 1
    m,b = polyfit(x, y, degree)
    sfvalues = [(res_num, from_aa, to_aa, value - (m*smap[(res_num,from_aa,to_aa)] + b))
                for res_num, from_aa, to_aa, value in cvalues
                if (res_num,from_aa,to_aa) in smap]
    return sfvalues

def dms_data(chain):
    return getattr(chain, '_deep_mutation_data', None)
