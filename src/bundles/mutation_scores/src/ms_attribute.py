# Assign a residue attribute computed from mutation scores.
def mutation_scores_attribute(session, score_name = None, scores_name = None, subtract_fit = None,
                              name = None, type = 'sum_absolute', above = None, below = None):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, scores_name)

    score_values = scores.score_values(score_name, subtract_fit = subtract_fit)
    
    attr_name = _attribute_name(score_name, type, above, below) if name is None else name
    from chimerax.atomic import Residue
    Residue.register_attr(session, attr_name, "Deep Mutational Scan", attr_type=float)

    chain = scores.chain
    residues = chain.existing_residues
    count = 0
    for res in residues:
        value = score_values.residue_value(res.number, value_type=type, above=above, below=below)
        if value is not None:
            setattr(res, attr_name, value)
            count += 1

    message = f'Set attribute {attr_name} for {count} residues of chain {chain}'
    session.logger.info(message)

def _attribute_name(column_name, type, above, below):
    attr_name = f'{column_name}_{type}'
    if above is not None:
        attr_name += f'_ge_{"%.3g"%above}'
    if below is not None:
        attr_name += f'_le_{"%.3g"%below}'
    return attr_name

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, FloatArg
    from .ms_data import ColumnValues
    desc = CmdDesc(
        optional = [('score_name', StringArg)],
        keyword = [('scores_name', StringArg),
                   ('subtract_fit', StringArg),
                   ('name', StringArg),
                   ('type', EnumOf(ColumnValues.residue_value_types)),
                   ('above', FloatArg),
                   ('below', FloatArg),
                   ],
        synopsis = 'Assign a residue attribute computed from mutation scores'
    )
    register('mutationscores attribute', desc, mutation_scores_attribute, logger=logger)
