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

def open_mutation_scores_csv(session, path, name = None, show_plot = False, chains = None, allow_mismatches = False):
    mset = _read_mutation_scores_csv(path, name = name, logger = session.logger)

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

    if not show_plot and session.ui.is_gui:
        from .ms_list import show_mutation_scores_list
        show_mutation_scores_list(session)

    return mset, message

def _read_mutation_scores_csv(path, name = None, logger = None):
    with open(path, 'r') as f:
        lines = f.readlines()
    headings = [h.strip() for h in lines[0].split(',')]
    hgvs_column = _hgvs_column(headings)
    mscores = []
    mut = set()
    hgvs_ignored = []
    from .ms_data import MutationScores    
    for i, line in enumerate(lines[1:]):
        if line.strip() == '':
            continue	# Ignore blank lines
        fields = line.split(',')
        if len(fields) != len(headings):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has wrong number of comma-separated fields, got {len(fields)}, but there are {len(headings)} headings')
        hgvs = fields[hgvs_column]
        res_num, res_type, res_type2 = _parse_hgvs(hgvs, line_num = i+2)
        if res_type2 is None:
            hgvs_ignored.append(hgvs)
            continue
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

    if hgvs_ignored and logger:
        ignored = ", ".join(hgvs_ignored[:10])
        if len(hgvs_ignored) > 10:
            ignored += ' ...'
        from chimerax.core.errors import UserError
        logger.info(f'Ignored {len(hgvs_ignored)} variants not of form p.<from_aa><num><to_aa>: {ignored}')

    return mset

def _hgvs_column(headings):
    hgvs_column = None
    hgvs_names = ('hgvs', 'hgvs_pro', 'variants')
    for hgvs_column_name in hgvs_names:
        if hgvs_column_name in headings:
            hgvs_column = headings.index(hgvs_column_name)
            break
    if hgvs_column is None:
        from chimerax.core.errors import UserError
        raise UserError('Did not find variant column ({", ".join(hgvs_names)}) in first line headings ({", ".join(headings)})')
    return hgvs_column


aa_3_to_1 = {'Cys':'C', 'Asp':'D', 'Ser':'S', 'Gln':'Q', 'Lys':'K',
             'Ile':'I', 'Pro':'P', 'Thr':'T', 'Phe':'F', 'Asn':'N', 
             'Gly':'G', 'His':'H', 'Leu':'L', 'Arg':'R', 'Trp':'W', 
             'Ala':'A', 'Val':'V', 'Glu':'E', 'Tyr':'Y', 'Met':'M'}

def _parse_hgvs(hgvs, line_num):
    if not hgvs.startswith('p.'):
        from chimerax.core.errors import UserError
        raise UserError(f'Line {line_num} has hgvs field "{hgvs}" not starting with "p."')
    var = hgvs[2:]
    if var.startswith('(') and var.endswith(')'):
        var = var[1:-1]
    try:
        if var[1].isdigit():
            # One-letter codes
            res_type = var[0]
            res_num = int(var[1:-1])
            res_type2 = var[-1]
            if res_type2 == '=':
                res_type2 = res_type
        else:
            # 3-letter coes
            res_type = aa_3_to_1[var[:3]]
            if var.endswith('='):
                res_num = int(var[3:-1])
                res_type2 = res_type
            else:
                res_num = int(var[3:-3])
                res_type2 = aa_3_to_1[var[-3:]]
    except (IndexError, ValueError, KeyError):
        return None, None, None
    return res_num, res_type, res_type2

def _parse_scores(headings, fields):
    scores = {}
    for h,f in zip(headings[1:], fields[1:]):
        try:
            scores[h] = float(f)
        except ValueError:
            continue
    return scores
