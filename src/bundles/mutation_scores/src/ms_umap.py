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

    from chimerax.diffplot.diffplot import _install_umap, _umap_embed, StructurePlot
    _install_umap(session)
    umap_xy = _umap_embed(array(rscores, float32))
    StructurePlot(session, res_names, umap_xy, colors)
    
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
