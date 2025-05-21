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

# Project a vector of mutation scores for each residue to 2D using UMAP.
def mutation_scores_umap(session, score_name = None, mutation_set = None):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, mutation_set)
    score_values = scores.score_values(score_name)

    amino_acids = 'PRKHDEFWYNQCSTILVMGA'
    aa_index = {c:i for i,c in enumerate(amino_acids)}
    from chimerax.core.colors import random_colors
    aa_colors = random_colors(20)
    count = 0
    rscores = []
    res_names = []
    colors = []
    from numpy import zeros, float32, array
    for res_num in score_values.residue_numbers():
        values = score_values.mutation_values(res_num)
        if len(values) == 20:
            count += 1
            va = zeros((20,), float32)
            for from_aa, to_aa, value in values:
                va[aa_index[to_aa]] = value
            rscores.append(va)
            res_names.append(f'{from_aa}{res_num}')
            colors.append(aa_colors[aa_index[from_aa]])

    from chimerax.umap import install_umap, umap_embed, UmapPlot
    install_umap(session)
    umap_xy = umap_embed(array(rscores, float32))
    plot = UmapPlot(session, title = 'Mutation Residue Plot', tool_name = 'MutationScores')
    plot.set_nodes(res_names, umap_xy, colors)
    
    message = f'{count} of {len(score_values.residue_numbers())} have 20 mutations'
    session.logger.info(message)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, FloatArg
    desc = CmdDesc(
        required = [('score_name', StringArg)],
        keyword = [('mutation_set', StringArg),
                   ],
        synopsis = 'Project residues in umap plot according to mutation scores'
    )
    register('mutationscores umap', desc, mutation_scores_umap, logger=logger)
