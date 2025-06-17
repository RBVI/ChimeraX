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

def open_mutation_scores_csv(session, path, name = None, show_plot = True, chains = None, allow_mismatches = False):
    mset = _read_mutation_scores_csv(path, name = name)

    if chains:
        mset.set_associated_chains(chains, allow_mismatches = allow_mismatches)

    from .ms_data import mutation_scores_manager
    msm = mutation_scores_manager(session)
    msm.add_scores(mset)

    nmut = len(mset.mutation_scores)
    dresnums = set(mset.residue_number_to_amino_acid().keys())
    score_names = ', '.join(mset.score_names())
    message = f'Opened deep mutational scan data for {nmut} mutations of {len(dresnums)} residues with score names {score_names}.'
    
    if chains:
        res, rnums = mset.associated_residues(dresnums)
        from chimerax.atomic import concatenate, concise_chain_spec
        cres = concatenate([chain.existing_residues for chain in chains])
        cspec = concise_chain_spec(chains)
        message += f' Assigned scores to {len(res)} of {len(cres)} residues of chain {cspec}.'
        sresnums = set(rnums)
        mres = len(dresnums - sresnums)
        if mres > 0:
            message += f' Found scores for {mres} residues not present in structures {cspec}.'

    if show_plot and session.ui.is_gui and len(mset.score_names()) >= 2:
        x_score_name, y_score_name = mset.score_names()[:2]
        from .ms_scatter_plot import mutation_scores_scatter_plot
        mutation_scores_scatter_plot(session, x_score_name, y_score_name, mset.name, replace = False)
        
    return mset, message

def _read_mutation_scores_csv(path, name = None):
    with open(path, 'r') as f:
        lines = f.readlines()
    headings = [h.strip() for h in lines[0].split(',')]
    mscores = []
    mut = set()
    from .ms_data import MutationScores    
    for i, line in enumerate(lines[1:]):
        if line.strip() == '':
            continue	# Ignore blank lines
        fields = line.split(',')
        if len(fields) != len(headings):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has wrong number of comma-separated fields, got {len(fields)}, but there are {len(headings)} headings')
        hgvs = fields[0]
        if not hgvs.startswith('p.(') or not hgvs.endswith(')'):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has hgvs field "{hgvs}" not starting with "p.(" and ending with ")"')
        if 'del' in hgvs or 'ins' in hgvs or '_' in hgvs:
            continue
        res_type = hgvs[3]
        res_num = int(hgvs[4:-2])
        res_type2 = hgvs[-2]
        if (res_num, res_type, res_type2) in mut:
            from chimerax.core.errors import UserError
            raise UserError(f'Duplicated mutation "{hgvs}" at line {i+2}')
        mut.add((res_num, res_type, res_type2))
        scores = _parse_scores(headings, fields)
        mscores.append(MutationScores(res_num, res_type, res_type2, scores))

    from os.path import basename, splitext
    name = splitext(basename(path))[0] if name is None else name
    from .ms_data import MutationSet
    mset = MutationSet(name, mscores, path = path)

    return mset

def _parse_scores(headings, fields):
    scores = {}
    for h,f in zip(headings[1:], fields[1:]):
        try:
            scores[h] = float(f)
        except ValueError:
            continue
    return scores
