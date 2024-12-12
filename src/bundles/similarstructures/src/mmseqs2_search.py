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

def mmseqs2_search(session, chain, database = 'pdb',
                   evalue_cutoff = 1e-3, identity_cutoff = 0, max_hits = 1000,
                   trim = None, alignment_cutoff_distance = None,
                   save_directory = None, show_table = True):
    '''Search PDB for similar sequences and display results in a table.'''
    global _query_in_progress
    if _query_in_progress:
        from chimerax.core.errors import UserError
        raise UserError('mmseqs2 search in progress.  Cannot run another search until current one completes.')

    if save_directory is None:
        from os.path import expanduser
        save_directory = expanduser('~/Downloads/ChimeraX/MMseqs2')

    Mmseqs2WebQuery(session, chain, database=database,
                    evalue_cutoff = evalue_cutoff, identity_cutoff = identity_cutoff, max_hits = max_hits,
                    trim=trim, alignment_cutoff_distance=alignment_cutoff_distance,
                    save_directory = save_directory, show_table = show_table)

mmseqs2_databases = ['pdb', 'afdb']

_query_in_progress = False

class Mmseqs2WebQuery:

    def __init__(self, session, chain, database = 'pdb',
                 evalue_cutoff = 1e-3, identity_cutoff = 0, max_hits = 1000,
                 trim = True, alignment_cutoff_distance = 2.0,
                 save_directory = None, show_table = True,
                 rcsb_search_url = 'https://search.rcsb.org/rcsbsearch/v2/query',
                 rcsb_graphql_url = 'https://data.rcsb.org/graphql'):
        self.session = session
        self.chain = chain
        self.database = database
        self.evalue_cutoff = evalue_cutoff
        self.identity_cutoff = identity_cutoff
        self.max_hits = max_hits	# Maximum PDB entities.  Can get many more chains than entities.
        self.trim = trim
        self.alignment_cutoff_distance = alignment_cutoff_distance
        self.save_directory = save_directory
        self.show_table = show_table
        self.rcsb_search_url = rcsb_search_url
        self.rcsb_graphql_url = rcsb_graphql_url

#        self._save_chain_path(chain)
#        self._save_query_sequence(chain)        
        results = self.submit_query()
        entity_hits = parse_pdb_search_results(results)
        chain_hits = add_chains_descrip_species(entity_hits)
        self.report_results(chain_hits)

    def submit_query(self):
        '''
        Use an https post to run RCSB mmseqs2 sequence search.
        '''
        sequence = self.chain.characters
        data = rcsb_search_template.format(sequence = sequence,
                                           identity_cutoff = self.identity_cutoff,
                                           evalue_cutoff = self.evalue_cutoff,
                                           max_hits = self.max_hits,
                                           structure_type = structure_types[self.database])

        from chimerax.core import version as cx_version
        headers = {
            "Content-Type": "application/json",
            'User-Agent': f'ChimeraX {cx_version}',	# Identify ChimeraX to server
        }

        import requests
        r = requests.post(self.rcsb_search_url, data = data, headers = headers)
        if r.status_code != 200:
            error_msg = r.text
            if not error_msg and r.status_code == 204:
                msg = 'RCSB sequence search no matches found'
            else:
                msg = f'RCSB sequence search failed (status {r.status_code}): {error_msg}'
            from chimerax.core.errors import UserError
            raise UserError(msg)

        results = r.json()
        return results

    def report_results(self, hits):
        from .simstruct import SimilarStructures
        results = SimilarStructures(hits, self.chain, program = 'mmseqs2', database = self.database,
                                    trim = self.trim, alignment_cutoff_distance = self.alignment_cutoff_distance)

        results.sms_path = results.save_to_directory(self.save_directory)
        
        if self.show_table:
            from .gui import show_similar_structures_table
            show_similar_structures_table(self.session, results)
    
structure_types = {'pdb':'experimental', 'afdb':'computational'}

# Use double curly brackets to allow format string substitutions with single curly brackets.
rcsb_search_template = '''{{
  "query": {{
    "type": "terminal",
    "service": "sequence",
    "parameters": {{
      "value": "{sequence}",
      "identity_cutoff": {identity_cutoff},
      "sequence_type": "protein",
      "evalue_cutoff": {evalue_cutoff}
    }}
  }},
  "return_type": "polymer_entity",
  "request_options": {{
    "paginate": {{
      "start": 0,
      "rows": {max_hits}
    }},
    "results_verbosity": "verbose",
    "results_content_type": [
      "{structure_type}"
    ],
    "sort": [
      {{
        "sort_by": "score",
        "direction": "desc"
      }}
    ],
    "scoring_strategy": "sequence"
  }}
}}'''

def parse_pdb_search_results(results):
    rcsb_to_simstruct_names = [
        ('sequence_identity', 'pident'),
        ('evalue', 'evalue'),
        ('bitscore', 'bits'),
        ('alignment_length', 'alnlen'),
        ('mismatches', 'mismatch'),
        ('gaps_opened', 'gapopen'),
        ('query_beg', 'qstart'),
        ('query_end', 'qend'),
        ('query_length', 'qlen'),
        ('subject_beg', 'tstart'),
        ('subject_end', 'tend'),
        ('subject_length', 'tlen'),
        ('query_aligned_seq', 'qaln'),
        ('subject_aligned_seq', 'taln'),
    ]
    hits = []
    for r in results['result_set']:
        entity_id = r['identifier']	#  "1XOM_1", "AF_AFP00442F1_1"
        if entity_id.startswith('MA_'): # Skip computational models that are not from AlphaFold
            continue
        alphafold = entity_id.startswith("AF_")
        if alphafold:
            fields = entity_id.split('_')
            alphafold_id = fields[1][2:-2]
            alphafold_fragment = fields[1][-2:]
            alphafold_version = fields[2] if len(fields) >= 3 else ''
        else:
            pdb_id = entity_id.split('_')[0]
        nodes = r['services'][0]['nodes']
        for node in nodes:
            for mc in node['match_context']:
                hit = {simstruct_attr:mc[rcsb_attr] for rcsb_attr, simstruct_attr in rcsb_to_simstruct_names}
                hit['pident'] *= 100	# Convert from fraction to percent
                if alphafold:
                    hit['alphafold_id'] = alphafold_id
                    hit['alphafold_fragment'] = alphafold_fragment
                    hit['alphafold_version'] = alphafold_version
                    hit['database_id'] = alphafold_id
                    hit['database_full_id'] = alphafold_id
                    hit['database'] = 'afdb'
                else:
                    hit['pdb_entity_id'] = entity_id
                    hit['pdb_id'] = pdb_id
                    hit['database_id'] = pdb_id
                    hit['database'] = 'pdb'
                hits.append(hit)
    return hits

def add_chains_descrip_species(hits):
    entity_ids = pdb_entity_ids(hits)
    if len(entity_ids) == 0:
        return
    pdb_info = fetch_pdb_entity_info(set(entity_ids))
    einfo = parse_pdb_entity_info(pdb_info)
    chits = []	# Chain hits
    for hit, entity_id in zip(hits, entity_ids):
        ei = einfo[entity_id]
        hit['description'] = ei['description']
        hit['taxname'] = ei['species']
        for chain_id in ei['chain_ids']:
            hc = hit.copy()
            hc['chain_id'] = chain_id
            if 'pdb_id' in hit:
                hc['database_full_id'] = f'{hit["pdb_id"]}_{chain_id}'
            chits.append(hc)
    return chits

def pdb_entity_ids(hits):
    ids = []
    for hit in hits:
        if hit['database'] == 'pdb':
            ids.append(hit['pdb_entity_id'])
        elif hit['database'] == 'afdb':
            ids.append(f"AF_AF{hit['alphafold_id']}{hit['alphafold_fragment']}_{hit['alphafold_version']}")
    return ids
    
# -----------------------------------------------------------------------------
# Fetch additional information about PDB hits using the RCSB GraphQL web service.
#
entity_info_query = """{
  polymer_entities(entity_ids: [%s]) {
    rcsb_id
    rcsb_polymer_entity {
        pdbx_description
    }
    rcsb_entity_source_organism {
        scientific_name
    }
    rcsb_polymer_entity_container_identifiers {
        asym_ids
        auth_asym_ids
    }
  }
}"""

def fetch_pdb_entity_info(entity_list, rcsb_graphql_url = 'https://data.rcsb.org/graphql'):
    entity_ids = ['"%s"' % entry_entity for entry_entity in entity_list]
    query = entity_info_query % ",".join(entity_ids)

    from chimerax.core import version as cx_version
    from urllib.request import urlopen, Request
    req = Request(
        rcsb_graphql_url,
        data=query.encode("utf-8"),
        headers={"Content-Type": "application/graphql", 'User-Agent': f'ChimeraX {cx_version}'},
    )
    f = urlopen(req)
    data = f.read()
    f.close()
    data = data.decode("utf-8")
    import json
    info = json.loads(data)
    if "errors" in info:
        raise ValueError("Fetching BLAST PDB info had errors: %s" % info["errors"])
    return info

def parse_pdb_entity_info(info):
    d = info['data']
    pe_list = d['polymer_entities']
    einfos = {}
    for pe in pe_list:
        entity_id = pe['rcsb_id']
        pdb_id = entity_id.split('_')[0]
        reso = pe['rcsb_entity_source_organism']
        species = reso[0]['scientific_name'] if reso else ''
        description = pe['rcsb_polymer_entity']['pdbx_description']
        asym_ids = pe['rcsb_polymer_entity_container_identifiers']['asym_ids']
        auth_asym_ids = pe['rcsb_polymer_entity_container_identifiers'].get('auth_asym_ids')
        chain_ids = auth_asym_ids if auth_asym_ids else asym_ids
        einfo = {
            'pdb_id': pdb_id,
            'chain_ids': chain_ids,
            'species': species,
            'description': description,
        }
        einfos[entity_id] = einfo
    return einfos
        
def register_mmseqs2_search_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or, ListOf, FloatArg, SaveFolderNameArg, IntArg
    from chimerax.atomic import ChainArg
    TrimArg = Or(ListOf(EnumOf(['chains', 'sequence', 'ligands'])), BoolArg)
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('database', EnumOf(['pdb', 'afdb'])),
                   ('trim', TrimArg),
                   ('evalue_cutoff', FloatArg),
                   ('identity_cutoff', FloatArg),
                   ('max_hits', IntArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('save_directory', SaveFolderNameArg),
                   ('show_table', BoolArg),
                   ],
        synopsis = 'Search for proteins with similar sequences using RCSB mmseqs2 web service'
    )
    register('sequence search', desc, mmseqs2_search, logger=logger)
