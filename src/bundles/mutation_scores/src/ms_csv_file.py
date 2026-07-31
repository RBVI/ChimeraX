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

def open_mutation_scores_csv(session, path, name = None, score_names = None,
                             show_plot = False, chains = None, allow_mismatches = False, manage = True):
    mset = read_mutation_scores_csv(path, name = name, score_names = score_names)
    from .variants import variant_parsing_problems
    warnings = variant_parsing_problems(mset, path)
    if warnings:
        disclaimer = '<p>ChimeraX only handles single amino acid non-synonymous and synonymous mutations and ignores multi-residue variants, deletions, insertions, stop codons, and alternate codons.</p>'
        session.logger.info(disclaimer + warnings, is_html = True)

    if chains:
        mset.set_associated_chains(chains, allow_mismatches = allow_mismatches)

    if manage:
        from .ms_data import mutation_scores_manager
        msm = mutation_scores_manager(session)
        msm.add_scores(mset)

    nmut = len(mset.mutation_scores)
    dresnums = set(mset.residue_number_to_amino_acid().keys())
    found_score_names = ', '.join(mset.score_names())
    message = f'Opened deep mutational scan data for {nmut} mutations of {len(dresnums)} residues with score names {found_score_names}.'
    
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

def read_mutation_scores_csv(path, name = None, score_names = None):
    with open(path, 'r') as f:
        lines = f.readlines()

    headings = _comma_separated_fields(lines[0])
    hgvs_column = _hgvs_column(headings, path)
    hgvs_nt_column = headings.index('hgvs_nt') if 'hgvs_nt' in headings else None
    if score_names is None:
        score_columns = [col for col,h in enumerate(headings) if h != hgvs_column]
    else:
        score_columns = [col for col,h in enumerate(headings) if h in score_names]

    mscores = []
    from .ms_data import MutationScores, Variant
    for line_num, line in enumerate(lines[1:]):
        if line.strip() == '':
            continue	# Ignore blank lines
        fields = _comma_separated_fields(line)
        if len(fields) != len(headings):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} of file {path} has wrong number of comma-separated fields, got {len(fields)}, but there are {len(headings)} headings')
        hgvs = fields[hgvs_column]
        hgvs_nt = None if hgvs_nt_column is None else fields[hgvs_nt_column]
        variant = Variant(hgvs, hgvs_nt)
        variant.line_number = line_num + 1
        scores = _parse_scores(headings, fields, score_columns)
        mscores.append(MutationScores(variant, scores))

    from os.path import basename, splitext
    filename = basename(path)
    name = splitext(filename)[0] if name is None else name
    from .ms_data import MutationSet
    mset = MutationSet(name, mscores, path = path)

    return mset

def _comma_separated_fields(line):
    fields = [_remove_quotes(field.strip()) for field in line.split(',')]
    return fields

def _remove_quotes(string):
    unquoted = string[1:-1] if string.startswith('"') and string.endswith('"') else string
    return unquoted

def _hgvs_column(headings, path):
    hgvs_column = None
    hgvs_names = ('hgvs', 'hgvs_pro', 'variants')
    for hgvs_column_name in hgvs_names:
        if hgvs_column_name in headings:
            hgvs_column = headings.index(hgvs_column_name)
            break
    if hgvs_column is None:
        from chimerax.core.errors import UserError
        raise UserError(f'ChimeraX only handles protein variant specifiers in HGVS format (e.g. p.Ser49Thr) not nucleotide variant specifiers.  Did not find protein variant column ({", ".join(hgvs_names)}) in {path} first line headings ({", ".join(headings)})')
    return hgvs_column
    
def _parse_scores(headings, fields, score_columns):
    scores = {}
    for column in score_columns:
        h = headings[column]
        f = fields[column]
        try:
            scores[h] = float(f)
        except ValueError:
            continue
    return scores
