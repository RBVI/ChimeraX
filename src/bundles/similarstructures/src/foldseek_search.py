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

def foldseek_search(session, chain, database = 'pdb100', trim = None, alignment_cutoff_distance = None,
                    save_directory = None, wait = False, show_table = True):
    '''Submit a Foldseek search for similar structures and display results in a table.'''
    global _query_in_progress
    if _query_in_progress:
        from chimerax.core.errors import UserError
        raise UserError('Foldseek search in progress.  Cannot run another search until current one completes.')

    if save_directory is None:
        from os.path import expanduser
        save_directory = expanduser(f'~/Downloads/ChimeraX/Foldseek')

    FoldseekWebQuery(session, chain, database=database,
                     trim=trim, alignment_cutoff_distance=alignment_cutoff_distance,
                     save_directory = save_directory, wait=wait, show_table=show_table)

foldseek_databases = ['pdb100', 'afdb50', 'afdb-swissprot', 'afdb-proteome']

#foldseek_databases = ['pdb100', 'afdb50', 'afdb-swissprot', 'afdb-proteome',
#                      'bfmd', 'cath50', 'mgnify_esm30', 'gmgcl_id']

_query_in_progress = False

# Foldseek omits some non-standard residues.  It appears the ones it accepts are about 150
# hardcoded in a C++ file
#
#   https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/GemmiWrapper.cpp
#
foldseek_accepted_3_letter_codes = set('ALA,ARG,ASN,ASP,CYS,GLN,GLU,GLY,HIS,ILE,LEU,LYS,MET,PHE,PRO,SER,THR,TRP,TYR,VAL,MSE,MLY,FME,HYP,TPO,CSO,SEP,M3L,HSK,SAC,PCA,DAL,CME,CSD,OCS,DPR,B3K,ALY,YCM,MLZ,4BF,KCX,B3E,B3D,HZP,CSX,BAL,HIC,DBZ,DCY,DVA,NLE,SMC,AGM,B3A,DAS,DLY,DSN,DTH,GL3,HY3,LLP,MGN,MHS,TRQ,B3Y,PHI,PTR,TYS,IAS,GPL,KYN,CSD,SEC,UNK'.split(','))

class FoldseekWebQuery:

    def __init__(self, session, chain, database = 'pdb100', trim = True,
                 alignment_cutoff_distance = 2.0, save_directory = None, wait = False,
                 show_table = True, foldseek_url = 'https://search.foldseek.com/api'):
        self.session = session
        self.chain = chain
        self.database = database
        self.trim = trim
        self.alignment_cutoff_distance = alignment_cutoff_distance
        self.save_directory = save_directory
        self.show_table = show_table
        self.foldseek_url = foldseek_url
        self._last_check_time = None
        self._status_interval = 1.0	# Seconds.  Frequency to check for search completion
        from chimerax.core import version as cx_version
        self._user_agent = {'User-Agent': f'ChimeraX {cx_version}'}	# Identify ChimeraX to Foldseek server
        self._download_chunk_size = 64 * 1024	# bytes

        mmcif_string = _mmcif_as_string(chain)
        # self._save('query.cif', mmcif_string)
        # self._save_query_path(chain)
        if wait:
            ticket_id = self.submit_query(mmcif_string, databases = [database])
            self.wait_for_results(ticket_id)
            results = self.download_results(ticket_id, report_progress = self._report_progress)
            self.report_results(results)
        else:
            self.query_in_thread(mmcif_string, database)

    def report_results(self, results):
        hit_lines = results[self.database]
        hits = [parse_search_result(hit, self.database) for hit in hit_lines]
        db = 'pdb' if self.database == 'pdb100' else 'afdb'
        from .simstruct import SimilarStructures
        results = SimilarStructures(hits, self.chain, program = 'foldseek', database = db,
                                    trim = self.trim, alignment_cutoff_distance = self.alignment_cutoff_distance)
        if self.show_table:
            from .gui import show_similar_structures_table
            show_similar_structures_table(self.session, results)
        results.save_to_directory(self.save_directory)
        # self._log_open_results_command()

    def submit_query(self, mmcif_string, databases = ['pdb100']):
        '''
        Use an https post to start at search using the Foldseek REST API.
        '''

        query_url = self.foldseek_url + '/ticket'
        files = {'q': mmcif_string}
        data = {
            'mode': '3diaa',  # "3diaa" for monomer searhces, "complex-3diaa" for multimer searches
            'database[]': databases
        }
        import requests
        r = requests.post(query_url, files=files, data=data, headers = self._user_agent)
        if r.status_code != 200:
            error_msg = r.text
            from chimerax.core.errors import UserError
            raise UserError(f'Foldseek search failed: {error_msg}')

        ticket = r.json()
        ticket_id = ticket['id']
        return ticket_id

    def wait_for_results(self, ticket_id, report_progress = None):
        # poll until the job is successful or fails
        import time
        elapsed = 0
        while not self.check_for_results(ticket_id):
            time.sleep(self._status_interval)
            elapsed += self._status_interval
            if report_progress:
                report_progress(f'Waiting for Foldseek results, {"%.0f"%elapsed} seconds elapsed')

    def check_for_results(self, ticket_id):
        # poll until the job was successful or failed
        status_url = self.foldseek_url + f'/ticket/{ticket_id}'
        import requests
        r = requests.get(status_url, headers = self._user_agent)
        status = r.json()

        if status['status'] == "ERROR":
            from chimerax.core.errors import UserError
            raise UserError(f'FoldSeek jobs failed {status}')

        if status['status'] == "COMPLETE":
            return True

        return False
    
    def download_results(self, ticket_id, report_progress = None):
        '''
        Return a dictionary mapping database name to results as m8 format tab-separated values.
        This may be done in a separate thread, so don't do any user interface calls that would use Qt.
        '''
        result_url = self.foldseek_url + f'/result/download/{ticket_id}'
        import requests
        results = requests.get(result_url, stream = True, headers = self._user_agent)
        if report_progress:
            total_bytes = results.headers.get('Content-length')
            of_total = '' if total_bytes is None else f'of {"%.1f" % (int(total_bytes) / (1024 * 1024))}'
            from time import time
            start_time = time()
        # Result is a tar gzip file containing a .m8 tab-separated value file.
        import tempfile
        rfile = tempfile.NamedTemporaryFile(prefix = 'foldseek_results_', suffix = '.tar.gz', delete = False)
        size = 0
        with rfile as f:
            for chunk in results.iter_content(chunk_size=self._download_chunk_size):
                f.write(chunk)
                size += len(chunk)
                if report_progress:
                    size_mb = size / (1024 * 1024)
                    elapsed = time() - start_time
                    report_progress(f'Reading Foldseek results {"%.1f" % size_mb}{of_total} Mbytes downloaded in {"%.0f" % elapsed} seconds')
            f.close()  # Cannot open twice to read tar file below on Windows

            # Extract tar file making a dictionary of results for each database searched
            m8_results = {}
            import tarfile
            tfile = tarfile.open(f.name)
            for filename in tfile.getnames():
                mfile = tfile.extractfile(filename)
                dbname = filename.replace('alis_', '').replace('.m8', '')
                m8_results[dbname] = [line.decode('utf-8') for line in mfile.readlines()]
            tfile.close()
            import os
            os.remove(f.name)
            
        '''
        # Save results to a file
        for dbname, hit_lines in m8_results.items():
            filename = self._results_file_name(dbname)
            self._save(filename, ''.join(hit_lines))
        '''
        
        return m8_results

    def _results_file_name(self, database = None):
        dbname = self.database if database is None else database
        c = self.chain
        structure_name = c.structure.name.replace(' ', '_')
        file_name = f'{structure_name}_{c.chain_id}_{dbname}.m8'
        return file_name

    def _report_progress(self, message):
        self.session.logger.status(message)

    def query_in_thread(self, mmcif_string, database):
        global _query_in_progress
        _query_in_progress = True
        from queue import Queue
        result_queue = Queue()
        import threading
        t = threading.Thread(target=self._submit_and_download_in_thread,
                             args=(mmcif_string, database, result_queue))
        t.start()
        # Check for results from query each new frame callback.
        self.session.triggers.add_handler('new frame',
                                          lambda *args, q=result_queue: self._check_for_results_from_thread(q))

    def _submit_and_download_in_thread(self, mmcif_string, database, result_queue):
        def _report_progress(message, result_queue=result_queue):
            result_queue.put(('status',message))
        try:
            _report_progress('Submitted Foldseek query')
            ticket_id = self.submit_query(mmcif_string, databases = [database])
            self.wait_for_results(ticket_id, report_progress = _report_progress)
            results = self.download_results(ticket_id, report_progress = _report_progress)
        except Exception as e:
            result_queue.put(('error', str(e)))
        else:
            result_queue.put(('success', results))
                             
    def _check_for_results_from_thread(self, result_queue):
        if result_queue.empty():
            return
        status, r = result_queue.get()
        if status == 'status':
            self.session.logger.status(r)
            return
        elif status == 'success':
            self.report_results(r)
        elif status == 'error':
            self.session.logger.warning(f'Foldseek query failed: {r}')
        global _query_in_progress
        _query_in_progress = False
        return 'delete handler'

    '''
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

    def _save_query_path(self, chain):
        if chain:
            path = getattr(chain.structure, 'filename', None)
            if path:
                self._save('query', f'{path}\t{chain.chain_id}')
    
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

def _mmcif_as_string(chain):
    structure = chain.structure.copy()
    cchain = [c for c in structure.chains if c.chain_id == chain.chain_id][0]
    extra_residues = structure.residues - cchain.existing_residues
    extra_residues.delete()
    import tempfile
    with tempfile.NamedTemporaryFile(prefix = 'foldseek_mmcif_', suffix = '.cif', mode = 'w+', delete = False) as f:
        f.close()  # On Windows can't open file twice and write, ChimeraX bug #16592
        from chimerax.mmcif.mmcif_write import write_mmcif
        write_mmcif(chain.structure.session, f.name, models = [structure])
        f = open(f.name, 'r')
        mmcif_string = f.read()
        f.close()
        import os
        os.remove(f.name)
    return mmcif_string

'''
query: Identifier for the query sequence/structure.
theader: Header line from the query sequence/structure file.
pident: Percentage identity between the query and target.
alnlen: Length of the alignment between query and target.
mismatch: Number of mismatches in the alignment.
gapopen: Penalty for opening a gap in the alignment.
qstart, qend: Starting and ending positions of the alignment in the query sequence.
tstart, tend: Starting and ending positions of the alignment in the target sequence/structure.
prob (Foldseek only): Probability of the match being homologous (Foldseek specific).
evalue: Expectation value (measure of the statistical significance of the alignment). Lower values indicate better matches.
bits: Bit score of the alignment (higher score indicates better match).
qlen: Length of the query sequence.
tlen: Length of the target sequence/structure.
qaln, taln: Alignment of the query and target sequences, respectively 
tca: Transformation (translation and rotation) applied to the target structure Cα to align with the query
tseq: Target sequence used for the alignment
taxid: taxonomy id
taxname: species name

For pdb100 database theader looks like "4mjs-assembly7.cif.gz_N crystal structure of a PB1 complex"
including the PDB id then a bioassembly number then .cif.gz then underscore and a chain ID, then a space
and then a text description.

For alphafold-swissprot theader looks like "AF-A7E3S4-F1-model_v4 RAF proto-oncogene serine/threonine-protein kinase".

It appears that tstart, tend are not the residue numbers used in the PDB file, they are the numbers
starting from 1 and including only polymer residues with atomic coordinates.  Also the tseq target
sequence only includes residues with atomic coordinates, and tlen is the number of residues with
atomic coordinates.

The tca field in a comma-separted list of C-alpha coordinates for the target structure after moving
to align with the query structure.  Would be more useful to have a transformation matrix.  Foldseek
github says the command-line local foldseek can optionally return transformation.  Figure out how to
get this for REST API results.

The tca field actually has coordinates for all target residues not just the ones that align to the query.
The description above from Martin Steinegger for tca appears to be wrong, it is simply the target CA
coordinates as described on the foldseek github site.

So it looks like the default output does not give a transform, but it gives the alignment of residues.
I can use the sequence alignment and align all C-alpha using the alignment.  Not sure that is what
foldseek does -- iterative pruning may give a better alignment.
'''

def parse_search_result(line, database):
    fields = line.strip().split('\t')
    field_names = ['query', 'theader', 'pident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'prob', 'evalue', 'bits', 'qlen', 'tlen', 'qaln', 'taln', 'tca', 'tseq', 'taxid', 'taxname']
    values = dict(zip(field_names, fields))
    for int_field in ['alnlen', 'mismatch', 'qstart', 'qend', 'tstart', 'tend', 'qlen', 'tlen']:
        values[int_field] = int(values[int_field])
    for float_field in ['pident', 'gapopen', 'prob', 'evalue', 'bits']:
        values[float_field] = float(values[float_field])
    from numpy import array, float32
    xyz = array([float(x) for x in values['tca'].split(',')], float32)
    n = len(xyz)//3
    values['tca'] = xyz.reshape((n,3))
    if database == 'pdb100':
        values.update(parse_pdb100_theader(values['theader']))
        db = 'pdb'
    if database.startswith('afdb'):
        values.update(parse_alphafold_theader(values['theader']))
        db = 'afdb'
    values['database'] = db
    values['coordinate_indexing'] = True	# Alignment indexing includes only residues with coordinates
    values['foldseek release'] = 10		# Handle changed output for different foldseek version
    return values

def parse_pdb100_theader(theader):
    '''Example: "4mjs-assembly7.cif.gz_N crystal structure of a PB1 complex"'''
    ia = theader.find('-assembly')
    if ia == -1:
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not contain expected string "-assembly"')

    iz = theader.find('.cif.gz_')
    if iz == -1:
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not contain expected string ".cif.gz_"')

    id = iz + theader[iz:].find(' ')
    if id < iz:
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not contain expected space character after chain id')

    pdb_id = theader[:4]
    assembly_id = theader[ia+9:iz]
    chain_id = theader[iz+8:id]
    if '-' in chain_id:
        # RCSB bioassembly chains can have a dash in them e.g. H-2.
        # Since we open non-assembly PDB entries remove the -# part to get correct chain id.
        chain_id = chain_id[:chain_id.find('-')]
    description = theader[id+1:]
    
    values = {
        'pdb_id': pdb_id,
        'pdb_assembly_id': assembly_id,
        'chain_id': chain_id,
        'description': description,
    }
    values['database_id'] = values['pdb_id']
    values['database_full_id'] = values['pdb_id'] + '_' + values['chain_id']

    return values

def parse_alphafold_theader(theader):
    '''Example: "AF-A7E3S4-F1-model_v4 RAF proto-oncogene serine/threonine-protein kinase"'''
    fields = theader.split('-')
    if fields[0] != 'AF':
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not start with "AF-"')

    if len(fields) < 3 or not fields[2].startswith('F'):
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not have 3rd "-" separated field starting with "F"')

    if len(fields) < 4 or  not fields[3].startswith('model_v'):
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not have 4th "-" separated field starting with "model_v"')

    version, descrip = fields[3][7:].split(' ', maxsplit=1)
    values = {
        'alphafold_id': fields[1],
        'alphafold_fragment': fields[2],
        'alphafold_version': version,
        'description': descrip,
    }
    values['database_id'] = values['alphafold_id']
    values['database_full_id'] = values['alphafold_id']
    
    return values

def open_foldseek_m8(session, path, query_chain = None, database = None):
    with open(path, 'r') as file:
        hit_lines = file.readlines()

    if query_chain is None:
        query_chain, models = _guess_query_chain(session, path)

    if database is None:
        database = _guess_database(path, hit_lines)
        if database is None:
            from os.path import basename
            filename = basename(path)
            from chimerax.core.errors import UserError
            raise UserError('Cannot determine database for foldseek file {filename}.  Specify which database ({", ".join(foldseek_databases)}) using the open command "database" option, for example "open {filename} database pdb100".')

    # TODO: The query_chain model has not been added to the session yet.  So it is has no model id which
    # messes up the hits table header.  Also it causes an error trying to set the Chain menu if that menu
    # has already been used.  But I don't want to add the structure to the session because then its log output
    # appears in the open Notes table.
    if models:
        session.models.add(models)
        models = []

    hits = [parse_search_result(hit, database) for hit in hit_lines]
    from .simstruct import SimilarStructures
    results = SimilarStructures(hits, query_chain, program = 'foldseek', database = database)
    from .gui import show_similar_structures_table
    show_similar_structures_table(session, results)

    return models, ''

def _guess_query_chain(session, results_path):
    '''Look for files in same directory as foldseek results to figure out query chain.'''

    from os.path import dirname, join, exists
    results_dir = dirname(results_path)
    qpath = join(results_dir, 'query')
    if exists(qpath):
        # See if the original query structure file is already opened.
        path, chain_id = open(qpath,'r').read().split('\t')
        from chimerax.atomic import AtomicStructure
        for s in session.models.list(type = AtomicStructure):
            if hasattr(s, 'filename') and s.filename == path:
                for c in s.chains:
                    if c.chain_id == chain_id:
                        return c, []
        # Try opening the original query structure file
        if exists(path):
            models, status = session.open_command.open_data(path)
            chains = [c for c in models[0].chains if c.chain_id == chain_id]
            return chains[0], models

    qpath = join(results_dir, 'query.cif')
    if exists(qpath):
        # Use the single-chain mmCIF file used in the Foldseek submission
        models, status = session.open_command.open_data(qpath)
        return models[0].chains[0], models
    
    return None, []

def _guess_database(path, hit_lines):
    for database in foldseek_databases:
        if path.endswith(database + '.m8'):
            return database

    from os.path import dirname, join, exists
    dpath = join(dirname(path), 'database')
    if exists(dpath):
        database = open(dpath,'r').read()
        return database

    hit_file = hit_lines[0].split('\t')[1]
    if '.cif.gz' in hit_file:
        database = 'pdb100'

    return None
        
def register_foldseek_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or, ListOf, FloatArg, SaveFolderNameArg
    from chimerax.atomic import ChainArg
    TrimArg = Or(ListOf(EnumOf(['chains', 'sequence', 'ligands'])), BoolArg)
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('database', EnumOf(foldseek_databases)),
                   ('trim', TrimArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('save_directory', SaveFolderNameArg),
                   ('wait', BoolArg),
                   ('show_table', BoolArg),
                   ],
        synopsis = 'Search for proteins with similar folds using Foldseek web service'
    )
    register('foldseek', desc, foldseek_search, logger=logger)
