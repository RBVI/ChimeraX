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

# Define a new mutation score or residue score computed from existing mutation scores.
def mutation_scores_define(session, score_name = None, from_score_name = None, mutation_set = None,
                           subtract_fit = None, aa = None, to_aa = None, synonymous = False,
                           above = None, below = None, ranges = None, combine = None,
                           set_attribute = True):

    from .ms_data import mutation_scores, ScoreValues
    scores = mutation_scores(session, mutation_set)

    if score_name is None:
        # List existing computed scores
        names = scores.computed_values_names()
        session.logger.info(f'Computed scores for {scores.name}: {", ".join(names)}')
        return names

    if from_score_name is None:
        from chimerax.core.errors import UserError
        raise UserError('Missing fromScoreName argument')

    from_score_values = scores.score_values(from_score_name)

    from_aa = aa
    if subtract_fit or from_aa or to_aa or synonymous or above is not None or below is not None or ranges:
        values = from_score_values.all_values()
        if subtract_fit:
            sub_score_values = scores.score_values(subtract_fit)
            values = _subtract_fit_values(values, sub_score_values.all_values())
        if from_aa is not None:
            values = [(rnum, faa, taa, value) for rnum, faa, taa, value in values if faa in from_aa]
        if to_aa is not None:
            values = [(rnum, faa, taa, value) for rnum, faa, taa, value in values if taa in to_aa]
        if synonymous:
            values = [(rnum, faa, taa, value) for rnum, faa, taa, value in values if taa == faa]
        if above is not None:
            values = [(rnum, faa, taa, value) for rnum, faa, taa, value in values if value >= above]
        if below is not None:
            values = [(rnum, faa, taa, value) for rnum, faa, taa, value in values if value <= below]
        if ranges is not None:
            values = _range_filter(values, ranges, scores)
        if len(values) == 0:
            from chimerax.core.errors import UserError
            raise UserError(f'No residues have score {score_name}')
        per_residue = (synonymous or (to_aa is not None and len(to_aa) == 1) or from_score_values.per_residue)
        svalues = ScoreValues(values, per_residue = per_residue)
    else:
        svalues = from_score_values

    if combine is None:
        scores.set_computed_values(score_name, svalues)
        values = [value for res_num, from_aa, to_aa, value in svalues.all_values()]
        range = f'having range {"%.3g"%min(values)} to {"%.3g"%max(values)}' if len(values) > 0 else ''
        session.logger.info(f'Defined score {score_name} {range} for {svalues.count()} mutations')
        return svalues

    # Compute per-residue values from per-mutation values.
    res_values = []
    taa = to_aa if to_aa and len(to_aa) == 1 else None
    for res_num, aa_type in svalues.residue_numbers_and_types():
        value = _combine_scores(svalues, res_num, operation=combine)
        if value is not None:
            res_values.append((res_num, aa_type, taa, value))
    rvalues = ScoreValues(res_values, per_residue = True)
    scores.set_computed_values(score_name, rvalues)
    values = [value for res_num, from_aa, to_aa, value in rvalues.all_values()]
    range = f'having range {"%.3g"%min(values)} to {"%.3g"%max(values)}' if len(values) > 0 else ''
    session.logger.info(f'Defined score {score_name} {range} for {rvalues.count()} residues using {svalues.count()} mutations')

    # Set residue attribute
    if set_attribute:
        scores.associate_chains(session)
        res, rnums = scores.associated_residues(rvalues.residue_numbers())
        if len(res) > 0:
            from chimerax.atomic import Residue
            Residue.register_attr(session, score_name, "Deep Mutational Scan", attr_type=float)
            for r,rnum in zip(res, rnums):
                setattr(r, score_name, rvalues.residue_value(rnum))
            from chimerax.atomic import concise_chain_spec
            cspec = concise_chain_spec(res.unique_chains)
            message = f'Set attribute {score_name} for {len(res)} residues of chain {cspec}'
            session.logger.info(message)

    return rvalues

def _range_filter(values, ranges, scores):
    '''
    Filter values list (rnum, from_aa, to_aa, value) based on a boolean expression involving score ranges
    such as "(dox <= -1.5 or dox >= 1.5) and mtx <= 1.0 and mtx >= -1.2".  The expression is treated
    as false for any mutation which is missing a score value for the scores named in the expression.
    '''
    try:
        co = compile(ranges, filename='expression', mode='eval')
    except SyntaxError as e:
        from chimerax.core.errors import UserError
        raise UserError(f'Ranges has invalid syntax: "{ranges}" at character {e.offset}')

    svalues = []
    for score_name in co.co_names:
        sv = scores.score_values(score_name, raise_error = False)
        if sv is None:
            from chimerax.core.errors import UserError
            raise UserError(f'Ranges variable "{score_name}" is not a mutation score')
        if sv.per_residue:
            vtable = {(rnum, from_aa):value for rnum, from_aa, to_aa, value in sv.all_values()}
        else:
            vtable = {(rnum, from_aa, to_aa):value for rnum, from_aa, to_aa, value in sv.all_values()}
        svalues.append((score_name, sv.per_residue, vtable))

    rvalues = []
    for rnum, from_aa, to_aa, value in values:
        var_values = {}
        missing = False
        for score_name, per_residue, mvalues in svalues:
            v = mvalues.get((rnum, from_aa)) if per_residue else mvalues.get((rnum, from_aa, to_aa))
            if v is None:
                missing = True
                break
            else:
                var_values[score_name] = v
        if not missing:
            vars = var_values.copy()
            if eval(co, {}, var_values):
                rvalues.append((rnum, from_aa, to_aa, value))

    return rvalues

# Allowed value_type in _combine_scores() function.
_combine_operations = ('sum', 'sum_absolute', 'mean', 'stddev', 'count', 'max', 'min', 'median')
    
def _combine_scores(score_values, residue_number, operation):
    values = [value for from_aa, to_aa, value in score_values.mutation_values(residue_number)]
    if len(values) == 0:
        return None
    elif operation == 'sum':
        value = sum(values)
    elif operation == 'sum_absolute':
        value = sum([abs(v) for v in values])
    elif operation == 'mean':
        from numpy import mean
        value = mean(values)
    elif operation == 'median':
        from numpy import median
        value = median(values)
    elif operation == 'stddev':
        from numpy import std
        value = std(values)
    elif operation == 'count':
        value = len(values)
    elif operation == 'max':
        value = max(values)
    elif operation == 'min':
        value = min(values)
    else:
        value = None

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

def mutation_scores_undefine(session, score_name, mutation_set = None):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, mutation_set)
    if not scores.remove_computed_values(score_name):
        from chimerax.core.errors import UserError
        raise UserError(f'No computed score named {score_name}')

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, FloatArg, BoolArg
    desc = CmdDesc(
        optional = [('score_name', StringArg)],
        keyword = [('from_score_name', StringArg),
                   ('mutation_set', StringArg),
                   ('subtract_fit', StringArg),
                   ('aa', StringArg),
                   ('to_aa', StringArg),
                   ('synonymous', BoolArg),
                   ('above', FloatArg),
                   ('below', FloatArg),
                   ('ranges', StringArg),
                   ('combine', EnumOf(_combine_operations)),
                   ('set_attribute', BoolArg),
                   ],
        synopsis = 'Compute and name a new score from existing mutation scores'
    )
    register('mutationscores define', desc, mutation_scores_define, logger=logger)

    desc = CmdDesc(
        required = [('score_name', StringArg)],
        keyword = [('mutation_set', StringArg)],
        synopsis = 'Remove a computed score'
    )
    register('mutationscores undefine', desc, mutation_scores_undefine, logger=logger)
