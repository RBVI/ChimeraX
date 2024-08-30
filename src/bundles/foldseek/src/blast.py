# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

# Create foldseek results from blast results so foldseek visualization can be used.

# TODO: The part of a BLAST hit sequence may have no atomic coordinates in the hit structure
#       (for example 8u1l_C with query 8jnp) or may align to a part of the query sequence
#       with no atomic coordinates. Do we want to filter those out?  We don't know which
#       residues of the hits have coordinates so we can't filter them out unless the
#       hit structures are loaded.

# TODO: Since foldseek uses qstart,qend and tstart,tend indices into the sequence of
#       residues with atomic coordinates, and blast uses start/end values relative to
#       the full sequence including residues without atomic coordinates, any code
#       that uses these values is probably wrong unless it handles both cases.
#	Also qaln and taln sequence strings include only residues with coordinates
#       in foldseek results, but include residues without coordinates in blast.

# TODO: Both query and hit residues in the alignment may not have atomic coordinates.
#       With foldseek results they always have atomic coordinates.  The traces and clustering
#       and LDDT code has to be modified to handle the aligned residues with missing coordinates.

def blast_results(session):
    bpr = _blast_results(session)
    if bpr is None:
        return None

    if hasattr(bpr, '_foldseek_results'):
        return bpr._foldseek_results

    database = bpr.params.database
    if database == 'pdb':
        database = 'pdb100'	# Use Foldseek name
    elif database == 'alphafold':
        database = 'afdb50'	# Use Foldseek name
    else:
        return None

    chain = _blast_query_chain(bpr)
    
    # Extract hit info from raw blast results in blast json output format 15
    j = bpr.job
    r = j.get_results()
    blast_hits = r['BlastOutput2'][0]['report']['results']['search']['hits']
    print (blast_hits[0])
    hits = []
    for hit in blast_hits:
        hits.extend(_blast_hit_to_foldseek(hit, database))

    from .foldseek import FoldseekResults
    results = FoldseekResults(hits, database, chain, program = 'blast')
    bpr._foldseek_results = results
    
    return results

def _blast_results(session):
    from chimerax.blastprotein.ui.results import BlastProteinResults
    bpr = [tool for tool in session.tools.list() if isinstance(tool, BlastProteinResults)]
    return bpr[0] if len(bpr) == 1 else None

def _blast_query_chain(blast_results):
    cspec = blast_results.params.chain	# '#1/B'
    from chimerax.atomic import ChainArg
    from chimerax.core.commands import AnnotationError
    try:
        chain = ChainArg.parse(cspec, blast_results.session)[0]
    except AnnotationError:
        chain = None
    return chain

def _blast_hit_to_foldseek(hit, database):
    hits = []
    # Description list gives multiple PDB entries with identical sequences.
    for desc in hit['description']:
        if database.startswith('pdb'):
            values = parse_pdb_id_title_species(desc)
        elif database.startswith('afdb'):
            values = parse_alphafold_id_title_species(desc)
        # HSPs allow multiple subsegment matches.
        for hsp in hit['hsps']:
            hdict = {
                'evalue': hsp['evalue'],
                'database': database,
                'qstart': hsp['query_from'],	# Base 1 index
                'qend': hsp['query_to'],
                'qaln': hsp['qseq'],
                'tstart': hsp['hit_from'],
                'tend': hsp['hit_to'],
                'taln': hsp['hseq'],
            }
            hdict.update(values)
            hits.append(hdict)

    return hits

def parse_pdb_id_title_species(desc):
    full_pdb_id = desc['accession']	# '8JNB_B'
    pdb_id, chain_id = full_pdb_id.split('_')
    title = desc['title']  # 'Chain B, RAF proto-oncogene serine/threonine-protein kinase, CRaf [Homo sapiens]'
    cprefix = f'Chain {chain_id}, '
    if title.startswith(cprefix):
        title = title[len(cprefix):]
    species = ''
    if title.endswith(']'):
        si = title.rfind('[')
        if si >= 0:
            species = title[si+1:-1]
            title = title[:si]

    values = {
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'description': title,
        'database_id': pdb_id,
        'database_full_id': full_pdb_id,
        'taxname': species,
    }
    return values

def parse_alphafold_id_title_species(desc):
    t = desc['title']  # 'AFDB:AF-A0A2J8NNQ8-F1 Uncharacterized protein UA=A0A2J8NNQ8 UI=A0A2J8NNQ8_PANTR OS=Pan troglodytes OX=9598 GN=CK820_G0008765'
    fields = t.split(maxsplit = 1)
    f0 = fields[0]
    f0s = f0.split('-')
    if len(f0s) < 3:
        from chimerax.core.errors import UserError
        raise UserError(f'BLAST title line does not start with two dashes in id (e.g. expect AFDB:AF-A0A2J8NNQ8-F1), got {f0}')
    uniprot_id = f0s[1]
    fragment = f0s[2]
    kv, prefix = _parse_key_values(fields[1] if len(fields) >= 2 else '')
    description = prefix
    species = kv.get('OS','')

    values = {
        'alphafold_id': uniprot_id,
        'alphafold_fragment': fragment,
        'description': description,
        'database_id': uniprot_id,
        'database_full_id': uniprot_id,
        'taxname': species,
    }
    return values

def _parse_key_values(line):
    # Example line 'AFDB:AF-A0A2J8NNQ8-F1 Uncharacterized protein UA=A0A2J8NNQ8 UI=A0A2J8NNQ8_PANTR OS=Pan troglodytes OX=9598 GN=CK820_G0008765'
    parts = line.split('=')
    np = len(parts)
    prefix = line if np == 1 else parts[0][:-3]
    kv = {}
    for i in range(1,np):
        key = parts[i-1][-2:]
        value = parts[i][:-3] if i < np-1 else parts[i]
        kv[key] = value
    return kv, prefix

    
'''
Alphafold blast hit
{'num': 1, 'description': [{'id': 'gnl|BL_ORD_ID|46458633', 'accession': '46458633', 'title': 'AFDB:AF-A0A2J8NNQ8-F1 Uncharacterized protein UA=A0A2J8NNQ8 UI=A0A2J8NNQ8_PANTR OS=Pan troglodytes OX=9598 GN=CK820_G0008765'}], 'len': 226, 'hsps': [{'num': 1, 'bit_score': 160.999, 'score': 406, 'evalue': 7.98027e-48, 'identity': 78, 'positive': 83, 'query_from': 5, 'query_to': 94, 'hit_from': 49, 'hit_to': 141, 'align_len': 93, 'gaps': 3, 'qseq': 'SDPSKTSNTIRVFLPNSQRTVVNVRNGMSLHDCLMKALKVRGLQPECCAVYRL---QDGEKKPIGWNTDAAWLIGEELQVEFLDHVPLTTHNF', 'hseq': 'TDPSKTSNTIRVFLPNKQRTVVNVRNGMSLHDCLMKALKVRGLQPECCAVFRLLHEHKGKKARLDWNTDAASLIGEELQVDFLDHVPLTTHNF', 'midline': '+DPSKTSNTIRVFLPN QRTVVNVRNGMSLHDCLMKALKVRGLQPECCAV+RL G+K + WNTDAA LIGEELQV+FLDHVPLTTHNF'}]}
'''
