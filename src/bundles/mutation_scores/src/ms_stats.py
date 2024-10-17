# Assign a residue attribute computed from mutation scores.
def mutation_scores_statistics(session, score_name = None, scores_name = None, type = 'synonymous'):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, scores_name)
    score_values = scores.score_values(score_name)
    
    values = []
    for res_num, from_aa, to_aa, value in score_values.all_values():
        if type == 'synonymous':
            if to_aa == from_aa or to_aa is None:
                values.append(value)
        else:
            values.append(value)

    import numpy
    mean = numpy.mean(values)
    std = numpy.std(values)
    
    message = f'Score {score_name}, {len(values)} {type} mutations, mean = {"%.3g"%mean}, standard deviation = {"%.3g"%std}, mean -/+ 2*SD = {"%.3g"%(mean-2*std)} to {"%.3g"%(mean+2*std)}'
    session.logger.info(message)

    return mean, std

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf
    desc = CmdDesc(
        required = [('score_name', StringArg)],
        keyword = [('scores_name', StringArg),
                   ('type', EnumOf(['synonymous', 'all'])),
                   ],
        synopsis = 'Compute mean and standard deviation of mutation scores'
    )
    register('mutationscores statistics', desc, mutation_scores_statistics, logger=logger)
