# Assign a residue attribute from deep mutational scan scores.
def dms_umap(session, chain, column_name, subtract_fit = None):
    from .dms_data import dms_data
    data = dms_data(chain)
    if data is None:
        from chimerax.core.errors import UserError
        raise UserError(f'No deep mutation scan data associated with chain {chain}')
    scores = data.column_values(column_name, subtract_fit = subtract_fit)

    amino_acids = 'PRKHDEFWYNQCSTILVMGA'
    aa_index = {c:i for i,c in enumerate(amino_acids)}
    from chimerax.core.colors import random_colors
    aa_colors = random_colors(20)
    count = 0
    rscores = []
    res_names = []
    colors = []
    from numpy import zeros, float32, array
    for res_num in scores.residue_numbers():
        values = scores.mutation_values(res_num)
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
    
    message = f'{count} of {len(scores.residue_numbers())} have 20 mutations'
    session.logger.info(message)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, FloatArg
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('column_name', StringArg),
                   ('subtract_fit', StringArg),
                   ],
        required_arguments = ['column_name'],
        synopsis = 'Project residues in umap plot according to mutation scores'
    )
    register('dms umap', desc, dms_umap, logger=logger)
