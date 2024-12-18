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

def similar_structures_blast(session, chain, database = 'pdb',
                             evalue_cutoff = 1e-3, max_hits = 1000,
                             trim = None, alignment_cutoff_distance = None,
                             save_directory = None, show_table = True):
    '''Search PDB for similar sequences and display results in a table.'''

    blast_db = {'pdb':'pdb', 'afdb':'alphafold'}[database]
    from chimerax.blastprotein import blastprotein
    job = blastprotein(session, chain, database = blast_db, cutoff = evalue_cutoff,
                       maxSeqs = max_hits, showResultsTable = False)

    def show_results(job, session=session, chain=chain, database=database, save_directory=save_directory, show_table=show_table):
        hits = _blast_job_hits(job, database)
        if len(hits) == 0:
            session.logger.warning(f'No BLAST matches found in {database} database.')
            return

        from .simstruct import SimilarStructures
        results = SimilarStructures(hits, chain, program = 'blast', database = database)

        if save_directory is None:
            from os.path import expanduser
            save_directory = expanduser('~/Downloads/ChimeraX/BLAST')
        results.sms_path = results.save_to_directory(save_directory)

        if show_table:
            from .gui import show_similar_structures_table
            show_similar_structures_table(session, results)
        
    from types import MethodType
    job.on_finish = MethodType(show_results, job)


def similar_structures_from_blast(session, instance_name = None, save = True, save_directory = None):
    br = blast_results(session, instance_name)
    
    from .gui import show_similar_structures_table
    show_similar_structures_table(session, br)

    if save:
        if save_directory is None:
            from os.path import expanduser
            save_directory = expanduser('~/Downloads/ChimeraX/BLAST')
        br.save_to_directory(save_directory)

    return br

def blast_results(session, instance_name):
    bpr = _blast_results(session, instance_name)
    database = bpr.params.database
    if database == 'alphafold':
        database = 'afdb'	# Name used by SimilarStructures class
    elif database != 'pdb':
        from chimerax.core.errors import UserError
        raise UserError('Only BLAST searches of PDB or Alphafold databases supported')

    chain = _blast_query_chain(bpr)
    hits = _blast_job_hits(bpr.job, database)
    
    from .simstruct import SimilarStructures
    results = SimilarStructures(hits, chain, program = 'blast', database = database)
    
    return results

def _blast_job_hits(job, database):
    # Extract hit info from raw blast results in blast json output format 15
    r = job.get_results()
    blast_hits = r['BlastOutput2'][0]['report']['results']['search']['hits']
    hits = []
    for hit in blast_hits:
        hits.extend(_blast_hit_to_simstruct(hit, database))
    return hits

def _blast_results(session, instance_name = None):
    from chimerax.blastprotein.ui.results import BlastProteinResults
    bpr = [tool for tool in session.tools.list() if isinstance(tool, BlastProteinResults)
           if instance_name is None or tool._instance_name == instance_name]
    if len(bpr) == 0:
        msg = 'No BLAST search has been done' if instance_name is None else f'No BLAST search named {instance_name} found'
        from chimerax.core.errors import UserError
        raise UserError(msg)
    if len(bpr) > 1:
        from chimerax.core.errors import UserError
        raise UserError(f'{len(bpr)} sets of BLAST results found.  Must specify a results name.')
    return bpr[0]

def _blast_query_chain(blast_results):
    cspec = blast_results.params.chain	# '#1/B'
    from chimerax.atomic import ChainArg
    from chimerax.core.commands import AnnotationError
    try:
        chain = ChainArg.parse(cspec, blast_results.session)[0]
    except AnnotationError:
        chain = None
    return chain

def _blast_hit_to_simstruct(hit, database):
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
                'pident': 100*hsp['identity']/hsp['align_len'],
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
        
def register_similar_structures_from_blast_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, SaveFolderNameArg, StringArg
    desc = CmdDesc(
        optional = [('instance_name', StringArg)],
        keyword = [('save', BoolArg),
                   ('save_directory', SaveFolderNameArg)],
        synopsis = 'Make a similar structures table from currently shown blast results'
    )
    register('similarstructures fromblast', desc, similar_structures_from_blast, logger=logger)
        
def register_similar_structures_blast_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or, ListOf, FloatArg, SaveFolderNameArg, IntArg
    from chimerax.atomic import ChainArg
    TrimArg = Or(ListOf(EnumOf(['chains', 'sequence', 'ligands'])), BoolArg)
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('database', EnumOf(['pdb', 'afdb'])),
                   ('evalue_cutoff', FloatArg),
                   ('max_hits', IntArg),
                   ('trim', TrimArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('save_directory', SaveFolderNameArg),
                   ('show_table', BoolArg),
                   ],
        synopsis = 'Search for proteins with similar sequences using RBVI BLAST web service'
    )
    register('similarstructures blast', desc, similar_structures_blast, logger=logger)
