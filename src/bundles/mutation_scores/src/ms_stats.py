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

# Assign a residue attribute computed from mutation scores.
def mutation_scores_statistics(session, score_name = None, mutation_set = None, type = 'synonymous'):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, mutation_set)
    score_values = scores.score_values(score_name)
    if score_values.count() == 0:
        score_values = scores.computed_values(score_name)
        if score_values is None:
            from chimerax.core.errors import UserError
            raise UserError(f'No mutation score named {score_name}')
    
    values = []
    for res_num, from_aa, to_aa, value in score_values.all_values():
        if type == 'synonymous' and not score_values.per_residue:
            if to_aa == from_aa or to_aa is None:
                values.append(value)
        else:
            values.append(value)

    if len(values) == 0:
        message = f'No {score_name} scores for mutations of type {type} found'
        mean = std = None
    else:
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
        keyword = [('mutation_set', StringArg),
                   ('type', EnumOf(['synonymous', 'all'])),
                   ],
        synopsis = 'Compute mean and standard deviation of mutation scores'
    )
    register('mutationscores statistics', desc, mutation_scores_statistics, logger=logger)
