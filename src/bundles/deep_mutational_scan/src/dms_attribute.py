# Assign a residue attribute from deep mutational scan scores.
def dms_attribute(session, chain, column_name, subtract_fit = None,
                  name = None, type = 'sum_absolute', above = None, below = None):
    from .dms_data import dms_data
    data = dms_data(chain)
    if data is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No deep mutation scan data associated with chain {chain}')
    scores = data.column_values(column_name, subtract_fit = subtract_fit)
    
    attr_name = _attribute_name(column_name, type, above, below) if name is None else name
    from chimerax.atomic import Residue
    Residue.register_attr(session, attr_name, "Deep Mutational Scan", attr_type=float)

    residues = chain.existing_residues
    count = 0
    for res in residues:
        value = scores.residue_value(res.number, value_type=type, above=above, below=below)
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
        attr_name += f'_lee_{"%.3g"%below}'
    return attr_name

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, FloatArg
    from chimerax.atomic import ChainArg
    from .dms_data import ColumnValues
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('column_name', StringArg),
                   ('subtract_fit', StringArg),
                   ('name', StringArg),
                   ('type', EnumOf(ColumnValues.residue_value_types)),
                   ('above', FloatArg),
                   ('below', FloatArg),
                   ],
        required_arguments = ['column_name'],
        synopsis = 'Assign a residue attribute using deep mutation scan scores'
    )
    register('dms attribute', desc, dms_attribute, logger=logger)
