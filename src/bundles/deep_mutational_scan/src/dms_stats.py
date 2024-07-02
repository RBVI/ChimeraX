# Assign a residue attribute from deep mutational scan scores.
def dms_statistics(session, chain, column_name, subtract_fit = None, type = 'synonymous'):
    from .dms_data import dms_data
    data = dms_data(chain)
    if data is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No deep mutation scan data associated with chain {chain}')
    scores = data.column_values(column_name, subtract_fit = subtract_fit)
    
    values = []
    wild_type_res = data.res_types
    for res_num, from_aa, to_aa, value in scores.all_values():
        if type == 'synonymous':
            if to_aa == from_aa:
                values.append(value)
        else:
            values.append(value)

    import numpy
    mean = numpy.mean(values)
    std = numpy.std(values)
    
    message = f'Column {column_name}, {len(values)} {type} mutations, mean = {"%.3g"%mean}, standard deviation = {"%.3g"%std}, mean -/+ 2*SD = {"%.3g"%(mean-2*std)} to {"%.3g"%(mean+2*std)}'
    session.logger.info(message)

    return mean, std

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('column_name', StringArg),
                   ('subtract_fit', StringArg),
                   ('type', EnumOf(['synonymous', 'all'])),
                   ],
        required_arguments = ['column_name'],
        synopsis = 'Compute mean and standard deviation of deep mutation scan scores'
    )
    register('dms statistics', desc, dms_statistics, logger=logger)
