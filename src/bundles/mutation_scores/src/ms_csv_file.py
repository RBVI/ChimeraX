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

def save_mutation_scores_csv(session, path, mutation_sets = None, score_names = None,
                             sort = True, merge_duplicates = True, value_format = '%.5g'):
    mset_to_score_names = _scores_to_save(session, mutation_sets, score_names)
    _check_sequences_match(list(mset_to_score_names.keys()))
    variant_scores = _variant_scores(mset_to_score_names)	# Map variant to list of score->value dictionaries
    if merge_duplicates:
        variant_scores = _merge_score_values(variant_scores)
    if sort:
        def order_by_residue_number(vs):
            variant = vs[0]
            resnum = variant.residue_number
            if resnum is None:
                resnum = 10000000000
            return (resnum, variant.canonical_hgvs_protein)
        variant_scores = dict(sorted(variant_scores.items(), key = order_by_residue_number))
        
    score_names = []
    for mset_score_names in mset_to_score_names.values():
        score_names.extend(mset_score_names)

    csv = _variant_scores_to_csv(variant_scores, score_names, value_format)

    if path:
        with open(path, 'w') as f:
            f.write(csv)

    return csv

def _variant_scores(mset_to_score_names):
    variant_scores = {}
    for mset, mset_score_names in mset_to_score_names.items():
        for mut_scores in mset.mutation_scores:
            var = mut_scores.variant
            scores = mut_scores.scores
            if _score_name_in_scores(mset_score_names, scores):
                if var in variant_scores:
                    variant_scores[var].append(scores)
                else:
                    variant_scores[var] = [scores]
    return variant_scores

def _score_name_in_scores(score_names, scores):
    for score_name in score_names:
        if score_name in scores:
            return True
    return False

def _merge_score_values(variant_scores):
    merged_score_values = {}
    for variant, score_values_list in variant_scores.items():
        if len(score_values_list) == 1:
            merged_score_values[variant] = score_values_list
        else:
            mscores = {}
            for score_values in score_values_list:
                for score_name, value in score_values.items():
                    if score_name in mscores:
                        mscores[score_name].append(value)
                    else:
                        mscores[score_name] = [value]
            ncopies = max(len(values) for values in mscores.values())
            merged_score_values[variant] = \
                [{score_name:values[i] for score_name, values in mscores.items() if len(values) > i}
                 for i in range(ncopies)]
    return merged_score_values

def _variant_scores_to_csv(variant_scores, score_names, value_format):
    include_hgvs_nt = False
    for variant in variant_scores.keys():
        if variant.hgvs_nucleotide:
            include_hgvs_nt = True
            break

    column_names = ['hgvs_pro', 'hgvs_nt'] if include_hgvs_nt else ['hgvs_pro']
    column_names += score_names
    header = csv_join(column_names)

    lines = [header]
    for variant, score_values_list in variant_scores.items():
        values = [variant.canonical_hgvs_protein]
        if include_hgvs_nt:
            hgvs_nt = variant.hgvs_nucleotide
            if hgvs_nt is None:
                hgvs_nt = ''
            values.append(hgvs_nt)
        for score_values in score_values_list:
            for score_name in score_names:
                value = (value_format % score_values[score_name]) if score_name in score_values else ''
                values.append(value)
        lines.append(csv_join(values))

    csv = '\n'.join(lines)
    return csv

def _scores_to_save(session, mutation_sets, score_names):

    # Find score names in mutatation sets.
    mset_to_score_names = _mutation_sets_to_score_names(session, score_names)

    # Add scores from mutation_sets argument for mutation sets not in score_names
    if mutation_sets or score_names is None:
        msets = _named_mutation_sets(session, mutation_sets)
        for mset in msets:
            if mset not in mset_to_score_names:
                mset_to_score_names[mset] = mset.score_names()

    # Check for duplcate score names
    _check_for_duplicate_score_names(mset_to_score_names)

    return mset_to_score_names

def _named_mutation_sets(session, mutation_sets):
    if mutation_sets is None:
        from .ms_data import mutation_all_scores
        msets = mutation_all_scores(session)
    else:
        mset_names = csv_split(mutation_sets)
        from .ms_data import mutation_scores
        msets = [mutation_scores(session, mset_name) for mset_name in mset_names]
    return msets

def _mutation_sets_to_score_names(session, score_names):
    mset_to_score_names = {}
    if score_names is None:
        return mset_to_score_names
    score_name_to_msets = _score_name_to_msets(session)
    score_name_list = csv_split(score_names)
    for score_name in score_name_list:
        mset, score_name = _score_name_mset(score_name, score_name_to_msets)
        if mset in mset_to_score_names:
            if score_name not in mset_to_score_names[mset]:
                mset_to_score_names[mset].append(score_name)
        else:
            mset_to_score_names[mset] = [score_name]
    return mset_to_score_names

def _score_name_to_msets(session):        
    from .ms_data import mutation_all_scores
    all_msets = mutation_all_scores(session)
    score_name_to_msets = {}
    for mset in all_msets:
        for score_name in mset.score_names():
            if score_name in score_name_to_msets:
                score_name_to_msets[score_name].append(mset)
            else:
                score_name_to_msets[score_name] = [mset]
    return score_name_to_msets

def _score_name_mset(score_name, score_name_to_msets):
    '''score_name can be of the form mset_name:score_name'''
    msets = score_name_to_msets.get(score_name, [])
    if len(msets) > 1:
        mset_names = ", ".join(mset.name for mset in msets)
        from chimerax.core.errors import UserError
        raise UserError(f'Score name "{score_name}" appears in more than one mutation set ({mset_names}). Use mutation set name followed by ":" then score name to specify which mutation set')
    elif len(msets) == 0:
        if ':' in score_name:
            mset_name, mset_score_name = score_name.split(':', maxsplit = 1)
            msets = score_name_to_msets.get(mset_score_name, [])
            msets = [mset for mset in msets if mset.name == mset_name]
            if len(msets) == 0:
                from chimerax.core.errors import UserError
                raise UserError(f'Did not find score name "{mset_score_name}" in mutation set {mset_name}')
            score_name = mset_score_name
        else:
            from chimerax.core.errors import UserError
            raise UserError(f'Did not find score name "{score_name}" in any mutation set')
    mset = msets[0]

    return mset, score_name

def _check_sequences_match(msets):
    if len(msets) > 1:
        ra = {}
        for mset in msets:
            for rnum, aa in mset.residue_number_to_amino_acid().items():
                if rnum in ra:
                    aa2, mset2 = ra[rnum]
                    if aa2 != aa:
                        from chimerax.core.errors import UserError
                        raise UserError(f'Mutation set {mset.name} has residue {aa}{rnum} that conflicts with mutation set {mset2.name} which has {aa2}{rnum}')
                else:
                    ra[rnum] = (aa, mset)

def _check_for_duplicate_score_names(mset_to_score_names):
    score_mset = {}
    for mset, mset_score_names in mset_to_score_names.items():
        for score_name in mset_score_names:
            if score_name in score_mset:
                from chimerax.core.errors import UserError
                raise UserError(f'Cannot save duplicate score name "{score_name}" occuring in mutation sets {mset.name} and {score_mset[score_name].name}')
            score_mset[score_name] = mset

def csv_split(string):
    '''Handle commas in quoted fields.'''
    import csv
    reader = csv.reader([string], skipinitialspace=True)
    fields = [field.strip() for field in next(reader)]
    return fields

def csv_join(strings):
    '''Handle commas in quoted fields.'''
    import io
    output = io.StringIO()
    import csv
    writer = csv.writer(output, lineterminator = '')
    writer.writerow(strings)
    csv_string = output.getvalue()
    return csv_string
