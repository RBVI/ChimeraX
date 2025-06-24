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

def fetch_alpha_missense_scores(session, uniprot_id, chains = None, allow_mismatches = False,
                                identifier = None, ignore_cache = False):
    '''
    Fetch AlphaMissense scores for a UniProt entry specified by its UniProt name or accession code.
    Data is in tab separated value format.
    Create a mutation scores instance. Example URL

       https://alphafold.ebi.ac.uk/files/AF-Q9UNQ0-F1-aa-substitutions.csv

    It seems DeepMind didn't make the data fetchable on a per-protein basis. We can get it from the EBI alphafold
    database only it is per-fragment.  Zenodo has the original data but in a gbyte size file.
    '''
    if '_' in uniprot_id:
        # Convert uniprot name to accession code.
        from chimerax.uniprot import map_uniprot_ident
        uid = map_uniprot_ident(uniprot_id, return_value = 'entry')
    else:
        uid = uniprot_id

    url_pattern = 'https://alphafold.ebi.ac.uk/files/AF-%s-F1-aa-substitutions.csv'
    url = url_pattern % uid
    file_name = url.split('/')[-1]
    save_dir = 'AlphaMissense'
    from chimerax.core.fetch import fetch_file
    path = fetch_file(session, url, f'AlphaMissense {uniprot_id}',
                          file_name, save_dir, ignore_cache = ignore_cache)

    if identifier is None:
        identifier = uniprot_id
    mset, msg = open_alpha_missense_scores(session, path, identifier = identifier,
                                           chains = chains, allow_mismatches = allow_mismatches)
    return mset, msg

def open_alpha_missense_scores(session, path, identifier = None, chains = None, allow_mismatches = False):
    with open(path, 'r') as f:
        lines = f.readlines()

    mutation_scores = parse_alpha_missense_scores(session, lines)
    if len(mutation_scores) == 0:
        msg = f'No mutation scores in {path_id}'
        mset = None
        return mset, msg

    if identifier is None:
        from os.path import basename, splitext
        file_name = splitext(basename(path))[0]
        fields = file_name.split('-', maxsplit=2)
        if len(fields) == 3 and fields[0] == 'AF' and fields[2] == 'F1-aa-substitutions':
            identifier = fields[1]
        else:
            identifier = file_name
    mset_name = identifier

    from .ms_data import mutation_scores_manager
    msm = mutation_scores_manager(session)
    mset = msm.mutation_set(mset_name)
    if mset is None:
        from .ms_data import MutationSet
        mset = MutationSet(mset_name, mutation_scores,
                           chains = chains, allow_mismatches = allow_mismatches,
                           path = path)
        msm.add_scores(mset)
    else:
        mset.add_scores(mutation_scores)
        if chains:
            mset.set_associated_chains(chains, allow_mismatches)

    nres = len(set(ms.residue_number for ms in mutation_scores))
    msg = f'Fetched AlphaMissense scores {identifier} for {nres} residues'

    return mset, msg

def parse_alpha_missense_scores(session, lines, score_name = 'amiss'):
    '''Return a list of MutationScores instances.'''
    mscores = []
    from .ms_data import MutationScores
    for line in lines[1:]:
        m, score, descrip = line.split(',')
        res_num = int(m[1:-1])
        from_aa = m[0]
        to_aa = m[-1]
        scores = {'amiss': float(score)}
        mscores.append(MutationScores(res_num, from_aa, to_aa, scores))
    return mscores

'''
Exapmle UniProt Variants JSON output

protein_variant,am_pathogenicity,am_class
M1A,0.503,Amb
M1C,0.396,Amb
M1D,0.8516,LPath
...
'''
