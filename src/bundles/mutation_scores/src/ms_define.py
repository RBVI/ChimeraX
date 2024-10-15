# Define a new mutation score or residue score computed from existing mutation scores.
def mutation_scores_define(session, score_name, from_score_name, scores_name = None,
                           subtract_fit = None, operation = None, above = None, below = None,
                           set_attribute = True):
    from .ms_data import mutation_scores, ScoreValues
    scores = mutation_scores(session, scores_name)
    from_score_values = scores.score_values(from_score_name)

    if subtract_fit:
        sub_score_values = scores.score_values(subtract_fit)
        values = _subtract_fit_values(from_score_values.all_values(), sub_score_values.all_values())
        svalues = ScoreValues(values)
    else:
        svalues = from_score_values

    if operation is None:
        scores.set_computed_values(score_name, svalues)
        return svalues

    # Compute per-residue values from per-mutation values.
    res_values = []
    to_aa = amino_acid_types.get(operation)
    for res_num, aa_type in svalues.residue_numbers_and_types():
        value = _residue_value(svalues, res_num, value_type=operation, above=above, below=below)
        if value is not None:
            res_values.append((res_num, aa_type, to_aa, value))
    if len(res_values) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'No residues have score {score_name}')
    rvalues = ScoreValues(res_values, per_residue = True)
    scores.set_computed_values(score_name, rvalues)

    # Set residue attribute
    if set_attribute:
        chain = scores.chain
        if chain:
            from chimerax.atomic import Residue
            Residue.register_attr(session, score_name, "Deep Mutational Scan", attr_type=float)

            count = 0
            for res in chain.existing_residues:
                value = rvalues.residue_value(res.number)
                if value is not None:
                    setattr(res, score_name, value)
                    count += 1

            message = f'Set attribute {score_name} for {count} residues of chain {chain}'
            session.logger.info(message)

    return rvalues

def _attribute_name(column_name, type, above, below):
    attr_name = f'{column_name}_{type}'
    if above is not None:
        attr_name += f'_ge_{"%.3g"%above}'
    if below is not None:
        attr_name += f'_le_{"%.3g"%below}'
    return attr_name
    
# Allowed value_type in _residue_value() function.
amino_acid_types = {'ala':'A','arg':'R','asn':'N','asp':'D','cys':'C','glu':'E','gln':'Q','gly':'G',
                    'his':'H','ile':'I','leu':'L','lys':'K','met':'M','phe':'F','pro':'P','ser':'S',
                    'thr':'T','trp':'W','tyr':'Y','val':'V'}
combine_operations = ('sum', 'sum_absolute', 'mean', 'stddev', 'count')
residue_value_types = combine_operations + ('synonymous',) + tuple(amino_acid_types.keys())
    
def _residue_value(score_values, residue_number, value_type='sum_absolute', above=None, below=None):
    dms_values = score_values.mutation_values(residue_number)
    if len(dms_values) == 0:
        return None

    value = None
    if value_type in combine_operations:
        values = [value for from_aa, to_aa, value in dms_values
                  if ((above is None and below is None)
                      or (above is not None and value >= above)
                      or (below is not None and value <= below))]
        if len(values) == 0:
            value = None
        elif value_type == 'sum':
            value = sum(values)
        elif value_type == 'sum_absolute':
            value = sum([abs(v) for v in values])
        elif value_type == 'mean':
            from numpy import mean
            value = mean(values)
        elif value_type == 'stddev':
            from numpy import std
            value = std(values)
        elif value_type == 'count':
            value = len(values)
        else:
            value = None
    elif value_type == 'synonymous':
        values = [value for from_aa, to_aa, value in dms_values if to_aa == from_aa]
        if values:
            value = values[0]
    elif value_type in amino_acid_types:
        one_letter_code = amino_acid_types[value_type]
        values = [value for from_aa, to_aa, value in dms_values if to_aa == one_letter_code]
        if len(values) == 1:
            value = values[0]
    return value

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

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, FloatArg, BoolArg
    desc = CmdDesc(
        required = [('score_name', StringArg)],
        keyword = [('from_score_name', StringArg),
                   ('scores_name', StringArg),
                   ('subtract_fit', StringArg),
                   ('operation', EnumOf(residue_value_types)),
                   ('above', FloatArg),
                   ('below', FloatArg),
                   ('set_attribute', BoolArg),
                   ],
        required_arguments = ['from_score_name'],
        synopsis = 'Compute and name a new score from existing mutation scores'
    )
    register('mutationscores define', desc, mutation_scores_define, logger=logger)
