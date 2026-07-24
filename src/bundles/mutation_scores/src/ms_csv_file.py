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
    mset, warnings = read_mutation_scores_csv(path, name = name, score_names = score_names)
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
    hgvs_column = _hgvs_column(headings)
    hgvs_nt_column = headings.index('hgvs_nt') if 'hgvs_nt' in headings else None
    if score_names is None:
        score_columns = [col for col,h in enumerate(headings) if h != hgvs_column]
    else:
        score_columns = [col for col,h in enumerate(headings) if h in score_names]
    mscores = []
    mut = set()
    hgvs_ignored = []
    duplicates = []
    from .ms_data import MutationScores    
    for i, line in enumerate(lines[1:]):
        if line.strip() == '':
            continue	# Ignore blank lines
        fields = _comma_separated_fields(line)
        if len(fields) != len(headings):
            from chimerax.core.errors import UserError
            raise UserError(f'Line {i+2} has wrong number of comma-separated fields, got {len(fields)}, but there are {len(headings)} headings')
        hgvs = fields[hgvs_column]
        res_num, res_type, res_type2 = _parse_hgvs(hgvs, line_num = i+2)
        if res_type2 is None:
            hgvs_ignored.append(hgvs)
            continue
        if hgvs_nt_column is not None:
            # MaveDB entry 1 has synonymous and non-synonymous mutations shown in the
            # hgvs_nt column, but only has the non-synonymous in the hgvs_pro column.
            # That is a multiple mutation that we drop.
            hgvs_nt = fields[hgvs_nt_column]
            if _is_multiresidue_mutation(hgvs_nt):
                hgvs_ignored.append(hgvs_nt)
                continue
        if (res_num, res_type, res_type2) in mut:
            duplicates.append((res_num, res_type, res_type2, hgvs, i+2))
            continue
        mut.add((res_num, res_type, res_type2))
        scores = _parse_scores(headings, fields, score_columns)
        mscores.append(MutationScores(res_num, res_type, res_type2, scores))

    from os.path import basename, splitext
    filename = basename(path)
    name = splitext(filename)[0] if name is None else name
    from .ms_data import MutationSet
    mset = MutationSet(name, mscores, path = path)

    warnings = _classify_ignored(hgvs_ignored, duplicates, len(lines), path)

    return mset, warnings

def _comma_separated_fields(line):
    fields = [_remove_quotes(field.strip()) for field in line.split(',')]
    return fields

def _remove_quotes(string):
    unquoted = string[1:-1] if string.startswith('"') and string.endswith('"') else string
    return unquoted

def _hgvs_column(headings):
    hgvs_column = None
    hgvs_names = ('hgvs', 'hgvs_pro', 'variants')
    for hgvs_column_name in hgvs_names:
        if hgvs_column_name in headings:
            hgvs_column = headings.index(hgvs_column_name)
            break
    if hgvs_column is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find variant column ({", ".join(hgvs_names)}) in first line headings ({", ".join(headings)})')
    return hgvs_column


aa_3_to_1 = {'Cys':'C', 'Asp':'D', 'Ser':'S', 'Gln':'Q', 'Lys':'K',
             'Ile':'I', 'Pro':'P', 'Thr':'T', 'Phe':'F', 'Asn':'N', 
             'Gly':'G', 'His':'H', 'Leu':'L', 'Arg':'R', 'Trp':'W', 
             'Ala':'A', 'Val':'V', 'Glu':'E', 'Tyr':'Y', 'Met':'M'}

def _parse_hgvs(hgvs, line_num):
    if hgvs in ('_wt', '_sy'):
        return None, None, None  # Deprecated MaveDB indicators of wild-type or synonymous, occurs in MaveDB entry 3
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
            # 3-letter codes
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

def _is_multiresidue_mutation(hgvs_nt):
    if hgvs_nt.startswith('c.[') and hgvs_nt.endswith(']'):
        mutations = hgvs_nt[3:-1].split(';')
        resnums = set()
        for mut in mutations:
            if 'delins' in mut:
                # Example mut = "241_243delinsCTG"
                rnum0, rnum1 = [1+(int(i)-1)//3 for i in mut[:mut.index('delins')].split('_')]
                for rnum in range(rnum0, rnum1+1):
                    resnums.add(rnum)
            else:
                # Example mut = "371A>G"
                rnum = 1+(int(mut[:-3])-1)//3
                resnums.add(rnum)
        return len(resnums) > 1
    return False
    
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

def _classify_ignored(hgvs_ignored, duplicates, num_lines, path):
    if len(hgvs_ignored) == 0 and len(duplicates) == 0:
        return ''

    multi = [hgvs for hgvs in hgvs_ignored if hgvs.count(';') >= 1 or 'delins' in hgvs]
    single = [hgvs for hgvs in hgvs_ignored if ';' not in hgvs and 'delins' not in hgvs]
    deletions = [hgvs for hgvs in single if 'del' in hgvs]
    insertions = [hgvs for hgvs in single if 'ins' in hgvs]
    stop = [hgvs for hgvs in single if 'Ter' in hgvs or '*' in hgvs]
    types = []
    categorized = set(multi + deletions + insertions + stop)
    other = [hgvs for hgvs in hgvs_ignored if hgvs not in categorized]
    alt_codon = [hgvs for res_num, res_type, res_type2, hgvs, line in duplicates]
    for type, name in [(multi, 'multi-residue'), (deletions, 'deletions'), (insertions, 'insertions'),
                       (stop, 'stop codons'), (alt_codon, 'alternate codons'), (other, 'unrecognized')]:
        if len(type) > 0:
            cat = f'<li>{len(type)} {name} {", ".join(type[:1])}'
            if len(type) > 1:
                cat += ', ...'
            types.append(cat)

    types_info = '<ul style="margin-top: 0; margin-bottom: 0;">' + '\n'.join(types) + '</ul>'
    from os.path import basename
    filename = basename(path)
    ni = len(hgvs_ignored) + len(duplicates)
    message = f'Discarded {ni} of {num_lines-1} variants in {filename}:\n{types_info}'
    return message
