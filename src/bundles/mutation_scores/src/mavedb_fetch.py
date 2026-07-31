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

mavedb_cache_directory = '~/Downloads/ChimeraX/MaveDB'
mavedb_api_url = 'https://api.mavedb.org/api/v1'

# -----------------------------------------------------------------------------
# Fetch MaveDB mutation data
#
#	https://mavedb.org
#	https://api.mavedb.org/docs#/
#
def fetch_mavedb(session, experiment_set_id, ignore_cache=False, **kw):
    from chimerax.core.errors import UserError
    nchar = len(experiment_set_id)
    if nchar > 8:
        raise UserError(f'MaveDB experiment set id is longer than 8 characters, "{experiment_set_id}" has {nchar} characters')
    from string import digits
    if [c for c in experiment_set_id if c not in digits]:
        raise UserError(f'MaveDB experiment set id must contain only numeric digits, got "{experiment_set_id}"')
    short_id = experiment_set_id.lstrip('0')
    if not short_id:
        raise UserError(f'MaveDB experiment set id must have non-zero character, got "{experiment_set_id}"')
    full_id = '0' * (8-nchar) + experiment_set_id

    from os.path import expanduser, join, exists
    cache_dir = join(expanduser(mavedb_cache_directory), short_id)
    cache_subdir = f'MaveDB/{short_id}'

    exp_set_path = join(cache_dir, f'{short_id}.json')
    if exists(exp_set_path):
        exp_ids, score_set_ids = _experiment_and_score_set_ids(exp_set_path)
    else:
        exp_ids, score_set_ids = _download_experiment_set_files(session, full_id, short_id,
                                                                cache_dir, cache_subdir, ignore_cache)

    msets, gene_names = _open_mutation_scores(session, cache_dir, score_set_ids)
    if msets:
        grouped_msets = _group_mutation_sets(session, msets)
        from .ms_list import show_mutation_scores_list
        show_mutation_scores_list(session)

        nvar = sum([mset.number_of_variants for mset in grouped_msets]) if msets else 0
        gene_names = tuple(set(mset.gene_name for mset in msets if hasattr(mset, 'gene_name')))
        gene_lengths = {mset.gene_name: mset.gene_length for mset in msets if hasattr(mset, 'gene_length')}
        names = [(f'{name} ({gene_lengths[name]} residues)' if name in gene_lengths else name) for name in gene_names]
        if len(names) == 1:
            targets = f'for protein {names[0]}'
        elif len(names) > 1:
            targets = f'for {len(gene_names)} proteins {", ".join(names)}'
        else:
            targets = f'in {len(exp_ids)} experiments'
        msg = f'Opened MaveDB entry {short_id} containing {nvar} variants {targets} with {len(score_set_ids)} scores from directory {cache_dir}.'
    else:
        targets = ', '.join(set(gene_names))
        msg = f'MaveDB entry {short_id} containing variants for {targets} cannot be opened because ChimeraX only handles variants specified as protein amino acid mutations (e.g. p.Ser49Thr) not as DNA nucleotide mutations (e.g. c.903A>T).'
        from chimerax.core.errors import UserError
        raise UserError(msg)
    models = []

    return models, msg

def _open_mutation_scores(session, cache_dir, score_set_ids):
    msets = []
    messages = []
    gene_names = []
    for score_set_id in score_set_ids:
        filename = _score_set_csv_filename(score_set_id)
        from os.path import join
        csv_path = join(cache_dir, filename)
        json_path = csv_path.replace('.csv', '.json')
        with open(json_path, 'r') as f:
            import json
            score_set_data = json.load(f)
            experiment_name = score_set_data['experiment']['title']
            score_name = score_set_data['title']
            genes = score_set_data.get('targetGenes', [])
            gene_names.extend(gene['name'] for gene in genes)
            if len(experiment_name) > 40 and len(gene_names) == 1:
                # Name mutation set using a shorter name
                experiment_name = gene_names[0]
        from .ms_csv_file import read_mutation_scores_csv
        from chimerax.core.errors import UserError
        try:
            mset = read_mutation_scores_csv(csv_path, name = experiment_name, score_names = ['score'])
        except UserError as e:
            messages.append(str(e))
            continue
        mset.rename_score('score', score_name)
        if len(genes) == 1:
            gene = genes[0]
            gene_name = gene['name']
            if 'targetSequence' in gene:
                ts = gene['targetSequence']
                if 'sequence' in ts and 'sequenceType' in ts and ts['sequenceType'] == 'dna':
                    mset.gene_length = len(ts['sequence']) // 3
            uniprot_id = gene.get('uniprotIdFromMappedMetadata')
        else:
            gene_name = None
            uniprot_id = None
        mset.gene_name = gene_name
        mset.uniprot_id = uniprot_id
        msets.append(mset)
        from .variants import variant_parsing_problems
        warnings = variant_parsing_problems(mset, csv_path)
        if warnings:
            messages.append(warnings)

    if messages:
        msg = '\n'.join(messages)
        session.logger.info(msg, is_html = True)

    return msets, gene_names

def _group_mutation_sets(session, msets):
    merged_msets = []
    remaining_msets = msets
    while remaining_msets:
        mset = remaining_msets[0]
        mset_group = [mset]
        keep_msets = []
        for mset2 in remaining_msets[1:]:
            from .ms_data import _mutation_sequences_match
            if _mutation_sequences_match(mset, mset2):
                mset_group.append(mset2)
            else:
                keep_msets.append(mset2)
        merged_mset = _merge_mutation_sets(session, mset_group)
        merged_msets.append(merged_mset)
        remaining_msets = keep_msets
    return merged_msets

def _merge_mutation_sets(session, msets):
    # Make sure score names are unique
    all_score_names = set()
    for m in msets:
        for score_name in m.score_names():
            if score_name in all_score_names:
                unique_score_name = _unique_name(score_name, all_score_names)
                m.rename_score(score_name, unique_score_name)
                all_score_names.add(unique_score_name)
            else:
                all_score_names.add(score_name)

    if len(set(m.name for m in msets)) == 1:
        name = msets[0].name
    elif msets[0].gene_name:
        name = msets[0].gene_name
    elif msets[0].uniprot_id:
        name = msets[0].uniprot_id
    else:
        name = msets[0].name
        
    from .ms_data import MutationSet, mutation_scores_manager
    mset = MutationSet(name, [])

    for m in msets:
        mset.add_scores(m.mutation_scores)

    uniprot_ids = set(m.uniprot_id for m in msets if m.uniprot_id)
    if len(uniprot_ids) == 1:
        mset.uniprot_id = uniprot_ids.pop()

    msm = mutation_scores_manager(session)
    msm.add_scores(mset)
    return mset

def _unique_name(name, used_names):
    if name in used_names:
        i = 1
        while True:
            new_name = f'{name} {i}'
            if new_name not in used_names:
                return new_name
            i += 1
    return name

def _mavedb_fix_missing_synonymous(hgvs, hgvs_nt, nt_sequence):
    '''
    MaveDB (e.g. entry 1) has synonymous and non-synonymous mutations shown in the
    nucleotide hgvs_nt column, but only has the non-synonymous in the protein hgvs_pro column.
    For example, c.[326C>A;678C>A],p.Ala109Asp or c.[687T>C;705G>T],p.= in entry 80 score set a_1.
    Compute a new protein hgvs using the nucleotide to include the synonymous changes.
    '''
    if hgvs.count(';') == hgvs_nt.count(';'):
        return hgvs
    changes = {}
    nt_changes = hgvs_nt.lstrip('c.[').rstrip(']').split(';')
    for nt_change in nt_changes:
        nnum = int(nt_change[:-3])
        from_nt = nt_change[-3]
        to_nt = nt_change[-1]
        if nt_sequence[nnum-1] != from_nt:
            raise ValueError(f'{hgvs_nt} has component {nt_change} that does not match sequence {nt_sequence}')
        changes[nnum-1] = to_nt
    pchanges = set(nnum//3 for nnum in changes.keys())
    for rnum in pchanges:
        nt_sequence[3*rnum:3*rnum+3]

    # TODO: This is going to be tricky to include insertions and deletions.
    #       Maybe should just try ot add in synonymous?

def _download_experiment_set_files(session, full_id, short_id, cache_dir, cache_subdir, ignore_cache):
    from os import makedirs
    makedirs(cache_dir)

    from urllib.parse import quote as quote_http
    exp_set_id = f'urn:mavedb:{full_id}'
    exp_set_url = mavedb_api_url + f'/experiment-sets/{quote_http(exp_set_id)}'

    from chimerax.core.fetch import fetch_file
    path = fetch_file(session, exp_set_url, f'MaveDB {short_id}', f'{short_id}.json',
                      cache_subdir, ignore_cache=ignore_cache)

    exp_ids, score_set_ids = _experiment_and_score_set_ids(path)

    for exp_id in exp_ids:
        short_exp_id = exp_id.split('-')[-1]
        exp_url = mavedb_api_url + f'/experiments/{quote_http(exp_id)}'
        path = fetch_file(session, exp_url, f'MaveDB experiment {short_exp_id}',
                          f'experiment_{short_exp_id}.json', cache_subdir,
                          ignore_cache=ignore_cache)

    for score_set_id in score_set_ids:
        short_score_set_id = '_'.join(score_set_id.split('-')[-2:])
        http_score_set_id = quote_http(score_set_id)
        score_set_url = mavedb_api_url + f'/score-sets/{http_score_set_id}'
        path = fetch_file(session, score_set_url, f'MaveDB score set {short_score_set_id}',
                          f'score_set_{short_score_set_id}.json', cache_subdir,
                          ignore_cache=ignore_cache)
        scores_url = mavedb_api_url + f'/score-sets/{http_score_set_id}/scores?drop_na_columns=true'
        path = fetch_file(session, scores_url, f'MaveDB scores csv {short_score_set_id}',
                          _score_set_csv_filename(score_set_id), cache_subdir,
                          ignore_cache=ignore_cache)

    return exp_ids, score_set_ids

def _score_set_csv_filename(score_set_id):
    short_score_set_id = '_'.join(score_set_id.split('-')[-2:])
    filename = f'score_set_{short_score_set_id}.csv'
    return filename
    
def _experiment_and_score_set_ids(path):
    import json
    with open(path, 'r') as f:
        exp_set_data = json.load(f)

    exp_ids = []
    score_set_ids = []
    for exp in exp_set_data['experiments']:
        exp_ids.append(exp['urn'])
        score_set_ids.extend(exp['scoreSetUrns'])

    return exp_ids, score_set_ids

