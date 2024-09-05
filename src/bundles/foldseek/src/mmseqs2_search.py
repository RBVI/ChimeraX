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

def mmseqs2_search(session, chain, database = 'pdb',
                   evalue_cutoff = 1e-3, identity_cutoff = 0, max_hits = 1000,
                   trim = None, alignment_cutoff_distance = None,
                   save_directory = None, wait = False):
    '''Search PDB for similar sequences and display results in a table.'''
    global _query_in_progress
    if _query_in_progress:
        from chimerax.core.errors import UserError
        raise UserError('mmseqs2 search in progress.  Cannot run another search until current one completes.')

    if save_directory is None:
        from os.path import expanduser
        save_directory = expanduser(f'~/Downloads/ChimeraX/mmseqs2/{chain.structure.name}_{chain.chain_id}')

    Mmseqs2WebQuery(session, chain, database=database,
                    evalue_cutoff = evalue_cutoff, identity_cutoff = identity_cutoff, max_hits = max_hits,
                    trim=trim, alignment_cutoff_distance=alignment_cutoff_distance,
                    save_directory = save_directory)

mmseqs2_databases = ['pdb', 'afdb']

_query_in_progress = False

class Mmseqs2WebQuery:

    def __init__(self, session, chain, database = 'pdb',
                 evalue_cutoff = 1e-3, identity_cutoff = 0, max_hits = 1000,
                 trim = True, alignment_cutoff_distance = 2.0,
                 save_directory = None,
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
        self.rcsb_search_url = rcsb_search_url
        self.rcsb_graphql_url = rcsb_graphql_url

        self._save_chain_path(chain)
        self._save_query_sequence(chain)        
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
            from chimerax.core.errors import UserError
            raise UserError(f'RCSB sequence search failed: {error_msg}')

        results = r.json()
        self._save(f'results_{self.database}.json', r.text)
        return results

    def report_results(self, hits):
        database = self.database
        from .foldseek import FoldseekResults
        results = FoldseekResults(hits, database, self.chain, trim = self.trim,
                                  alignment_cutoff_distance = self.alignment_cutoff_distance,
                                  program = 'mmseqs2')
        from .gui import show_foldseek_results
        show_foldseek_results(self.session, results)
#        self._log_open_results_command()

    def _save(self, filename, data):
        save_directory = self.save_directory
        if save_directory is None:
            return
        import os, os.path
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        file_mode = 'w' if isinstance(data, str) else 'wb'
        path = os.path.join(save_directory, filename)
        with open(path, file_mode) as file:
            file.write(data)

    def _save_chain_path(self, chain):
        if chain:
            path = getattr(chain.structure, 'filename', None)
            if path:
                self._save('chain', f'{path}\t{chain.chain_id}')

    def _save_query_sequence(self, chain):
        if chain:
            self._save('sequence', chain.characters)

    '''
    def _log_open_results_command(self):
        if not self.save_directory:
            return
        from os.path import join
        m8_path = join(self.save_directory, self._results_file_name())
        cspec = self.chain.string(style='command')
        from chimerax.core.commands import log_equivalent_command, quote_path_if_necessary
        cmd = f'open {quote_path_if_necessary(m8_path)} database {self.database} chain {cspec}'
        log_equivalent_command(self.session, cmd)

        # Record in file history so it is easy to reopen Foldseek results.
        from chimerax.core.filehistory import remember_file
        remember_file(self.session, m8_path, 'foldseek', [self.chain.structure], file_saved=True)
    '''
    
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
    rcsb_to_foldseek_names = [
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
        entity_id = r['identifier']	#  "1XOM_1"
        pdb_id = entity_id.split('_')[0]
        nodes = r['services'][0]['nodes']
        for node in nodes:
            for mc in node['match_context']:
                hit = {foldseek_attr:mc[rcsb_attr] for rcsb_attr, foldseek_attr in rcsb_to_foldseek_names}
                hit['pdb_entity_id'] = entity_id
                hit['pdb_id'] = pdb_id
                hit['database_id'] = pdb_id
                hit['database'] = 'pdb'
                hits.append(hit)
    return hits

def add_chains_descrip_species(hits):
    entity_ids = set(hit['pdb_entity_id'] for hit in hits)
    if len(entity_ids) == 0:
        return
    pdb_info = fetch_pdb_entity_info(entity_ids)
    einfo = parse_pdb_entity_info(pdb_info)
    chits = []	# Chain hits
    for hit in hits:
        entity_id = hit['pdb_entity_id']
        ei = einfo[entity_id]
        hit['description'] = ei['description']
        hit['taxname'] = ei['species']
        for chain_id in ei['chain_ids']:
            hc = hit.copy()
            hc['chain_id'] = chain_id
            hc['database_full_id'] = f'{hit["pdb_id"]}_{chain_id}'
            chits.append(hc)
    return chits
    
# -----------------------------------------------------------------------------
# Fetch additional information about PDB hits using the RCSB GraphQL web service.
#
entity_info_query = """{
  polymer_entities(entity_ids: [%s]) {
    rcsb_id
    entity_src_gen {
        pdbx_gene_src_scientific_name
    }
    rcsb_polymer_entity {
        pdbx_description
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
        species = pe['entity_src_gen'][0]['pdbx_gene_src_scientific_name']
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
                   ],
        synopsis = 'Search for proteins with similar sequences using RCSB mmseqs2 web service'
    )
    register('mmseqs2 search', desc, mmseqs2_search, logger=logger)
