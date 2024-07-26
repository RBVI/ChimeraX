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

def foldseek(session, chain, database = 'pdb100', trim = None, alignment_cutoff_distance = None,
             save_directory = None, wait = False):
    '''Submit a Foldseek search for similar structures and display results in a table.'''
    global _query_in_progress
    if _query_in_progress:
        from chimerax.core.errors import UserError
        raise UserError('Foldseek search in progress.  Cannot run another search until current one completes.')

    if save_directory is None:
        from os.path import expanduser
        save_directory = expanduser(f'~/Downloads/ChimeraX/Foldseek/{chain.structure.name}_{chain.chain_id}')

    FoldseekWebQuery(session, chain, database=database,
                     trim=trim, alignment_cutoff_distance=alignment_cutoff_distance,
                     save_directory = save_directory, wait=wait)

foldseek_databases = ['pdb100', 'afdb50', 'afdb-swissprot', 'afdb-proteome']

#foldseek_databases = ['pdb100', 'afdb50', 'afdb-swissprot', 'afdb-proteome',
#                      'bfmd', 'cath50', 'mgnify_esm30', 'gmgcl_id']

_query_in_progress = False

class FoldseekWebQuery:

    def __init__(self, session, chain, database = 'pdb100', trim = True,
                 alignment_cutoff_distance = 2.0, save_directory = None, wait = False,
                 foldseek_url = 'https://search.foldseek.com/api'):
        self.session = session
        self.chain = chain
        self.database = database
        self.trim = trim
        self.alignment_cutoff_distance = alignment_cutoff_distance
        self.save_directory = save_directory
        self.foldseek_url = foldseek_url
        self._last_check_time = None
        self._status_interval = 1.0	# Seconds.  Frequency to check for search completion
        from chimerax.core import version as cx_version
        self._user_agent = {'User-Agent': f'ChimeraX {cx_version}'}	# Identify ChimeraX to Foldseek server
        self._download_chunk_size = 64 * 1024	# bytes

        # Use cached 8jnb results for developing user interface so I don't have to wait for server every test.
#        with open('/Users/goddard/ucsf/chimerax/src/bundles/foldseek_example/alis_pdb100.m8', 'r') as mfile:
#            results = {database: mfile.readlines()}
#        self.report_results(results)
#        return
        
        mmcif_string = _mmcif_as_string(chain)
        self._save('query.cif', mmcif_string)
        self._save_query_path(chain)
        if wait:
            ticket_id = self.submit_query(mmcif_string, databases = [database])
            self.wait_for_results(ticket_id)
            results = self.download_results(ticket_id, report_progress = self._report_progress)
            self.report_results(results)
        else:
            self.query_in_thread(mmcif_string, database)

    def report_results(self, results):
        hits = results[self.database]
        show_foldseek_hits(self.session, hits, self.database, self.chain,
                           trim = self.trim, alignment_cutoff_distance = self.alignment_cutoff_distance)
        self._log_open_results_command()

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
        '''
        result_url = self.foldseek_url + f'/result/download/{ticket_id}'
        import requests
        results = requests.get(result_url, stream = True, headers = self._user_agent)
        if report_progress:
            total_bytes = results.headers.get('Content-length')
            of_total = '' if total_bytes is None else f'of {"%.1f" % (total_bytes / (1024 * 1024))}'
            from time import time
            start_time = time()
        # Result is a tar gzip file containing a .m8 tab-separated value file.
        import tempfile
        rfile = tempfile.NamedTemporaryFile(prefix = 'foldseek_results_', suffix = '.tar.gz')
        size = 0
        with rfile as f:
            for chunk in results.iter_content(chunk_size=self._download_chunk_size):
                f.write(chunk)
                size += len(chunk)
                if report_progress:
                    size_mb = size / (1024 * 1024)
                    elapsed = time() - start_time
                    report_progress(f'Reading Foldseek results {"%.1f" % size_mb}{of_total} Mbytes downloaded in {"%.0f" % elapsed} seconds')
            f.flush()

            # Extract tar file making a dictionary of results for each database searched
            m8_results = {}
            import tarfile
            tfile = tarfile.open(f.name)
            for filename in tfile.getnames():
                mfile = tfile.extractfile(filename)
                dbname = filename.replace('alis_', '').replace('.m8', '')
                m8_results[dbname] = [line.decode('utf-8') for line in mfile.readlines()]

        # Save results to a file
        for dbname, hit_lines in m8_results.items():
            self._save(dbname + '.m8', ''.join(hit_lines))

        return m8_results

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

    def _save(self, filename, data):
        save_directory = self.save_directory
        if save_directory is None:
            return
        from os import path, makedirs
        if not path.exists(save_directory):
            makedirs(save_directory)
        file_mode = 'w' if isinstance(data, str) else 'wb'
        with open(path.join(save_directory, filename), file_mode) as file:
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
        m8_path = join(self.save_directory, self.database + '.m8')
        cspec = self.chain.string(style='command')
        from chimerax.core.commands import log_equivalent_command, quote_path_if_necessary
        cmd = f'open {quote_path_if_necessary(m8_path)} database {self.database} chain {cspec}'
        log_equivalent_command(self.session, cmd)

def _mmcif_as_string(chain):
    structure = chain.structure.copy()
    cchain = [c for c in structure.chains if c.chain_id == chain.chain_id][0]
    extra_residues = structure.residues - cchain.existing_residues
    extra_residues.delete()
    import tempfile
    with tempfile.NamedTemporaryFile(prefix = 'foldseek_mmcif_', suffix = '.cif', mode = 'w+') as f:
        from chimerax.mmcif.mmcif_write import write_mmcif
        write_mmcif(chain.structure.session, f.name, models = [structure])
        mmcif_string = f.read()
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
    fields = line.split('\t')
    field_names = ['query', 'theader', 'pident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'prob', 'evalue', 'bits', 'qlen', 'tlen', 'qaln', 'taln', 'tca', 'tseq', 'taxid', 'taxname']
    values = dict(zip(field_names, fields))
    for int_field in ['alnlen', 'mismatch', 'qstart', 'qend', 'tstart', 'tend', 'qlen', 'tlen']:
        values[int_field] = int(values[int_field])
    for float_field in ['pident', 'gapopen', 'prob', 'evalue', 'bits']:
        values[float_field] = float(values[float_field])
    values['database'] = database
    if database == 'pdb100':
        values.update(parse_pdb100_theader(values['theader']))
    if database.startswith('afdb'):
        values.update(parse_alphafold_theader(values['theader']))
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
        
def open_hit(session, hit, query_chain, trim = True, align = True, alignment_cutoff_distance = 2.0,
             in_file_history = True, log = True):
    af_frag = hit.get('alphafold_fragment')
    if af_frag is not None and af_frag != 'F1':
        session.logger.warning(f'Foldseek AlphaFold database hit {hit["alphafold_id"]} was predicted in fragments and ChimeraX is not able to fetch the fragment structures')
        return

    db_id = hit.get('database_id')
    if in_file_history:
        from_db = 'from alphafold' if hit['database'].startswith('afdb') else ''
        open_cmd = f'open {db_id} {from_db}'
        if not log:
            open_cmd += ' logInfo false'
        from chimerax.core.commands import run
        structures = run(session, open_cmd, log = log)
    else:
        kw = {'from_database': 'afdb'} if hit['database'].startswith('afdb') else {}
        structures, status = session.open_command.open_data(db_id, log_info = log, **kw)
        session.models.add(structures)

    name = hit.get('database_full_id')
    # Can get multiple structures such as NMR ensembles from PDB.
    stats = []
    for structure in structures:
        structure.name = name
        _remember_sequence_alignment(structure, query_chain, hit)
        aligned_res = trim_structure(structure, hit, trim, log = log)
        _show_ribbons(structure)

        if query_chain is not None and align:
            # Align the model to the query structure using Foldseek alignment.
            # Foldseek server does not return transform by default, so compute from sequence alignment.
            res, query_res = alignment_residue_pairs(hit, aligned_res, query_chain)
            p, rms, npairs = alignment_transform(res, query_res, alignment_cutoff_distance)
            stats.append((rms, npairs))
            structure.position = p

    if log and query_chain is not None and align:
        chain_id = hit.get('chain_id')
        cname = '' if chain_id is None else f' chain {chain_id}'
        if len(structures) == 1:
            msg = f'Alignment of {db_id}{cname} to query has RMSD {"%.3g" % rms} using {npairs} of {len(res)} paired residues'
        else:
            rms = [rms for rms,npair in stats]
            rms_min, rms_max = min(rms), max(rms)
            npair = [npair for rms,npair in stats]
            npair_min, npair_max = min(npair), max(npair)
            msg = f'Alignment of {db_id}{cname} ensemble of {len(structures)} structures to query has RMSD {"%.3g" % rms_min} - {"%.3g" % rms_max} using {npair_min}-{npair_max} of {len(res)} paired residues'      
        if alignment_cutoff_distance is not None and alignment_cutoff_distance > 0:
            msg += f' within cutoff distance {alignment_cutoff_distance}'
        session.logger.info(msg)

    return structures

def _remember_sequence_alignment(structure, query_chain, hit):
    '''
    Remember the positions of the aligned residues as part of the hit dictionary,
    so that after hit residues are trimmed we can still deduce the hit to query
    residue pairing.  The Foldseek hit and query alignment only includes a subset
    of residues from the structures and only include residues that have atomic coordinates.
    '''
    chain_id = hit.get('chain_id')
    chain = _structure_chain_with_id(structure, chain_id)
    hit['aligned_residue_offsets'] = _indices_of_residues_with_coords(chain, hit['tstart'], hit['tend'])
    hit['query_residue_offsets'] = _indices_of_residues_with_coords(query_chain, hit['qstart'], hit['qend'])

def _indices_of_residues_with_coords(chain, start, end):
    seqi = []
    si = 0
    for i, r in enumerate(chain.residues):
        if r is not None and r.name in _foldseek_accepted_3_letter_codes:
            si += 1
            if si >= start and si <= end:
                seqi.append(i)
            elif si > end:
                break
    return seqi
    
def trim_structure(structure, hit, trim, ligand_range = 3.0, log = True):
    res, chain_res = residue_range(structure, hit, structure.session.logger)
    if res is None:
        name = hit.get('database_full_id')
        structure.session.logger.warning(f'Because of a sequence mismatch between the database structure {name} and the Foldseek output, the alignment and residue pairing of this structure is probably wrong, and ChimeraX did not trim the structure.')
        return res
    rnum_start, rnum_end = res[0].number, res[-1].number
    crnum_start, crnum_end = chain_res[0].number, chain_res[-1].number

    if not trim:
        return res
    
    chain_id = hit.get('chain_id')
    
    if (trim is True or 'chains' in trim) and chain_id is not None:
        if len(structure.chains) > 1:
            cmd = f'delete #{structure.id_string} & ~/{chain_id}'
            from chimerax.core.commands import run
            run(structure.session, cmd, log = log)

    trim_seq = (trim is True or 'sequence' in trim)
    if trim_seq:
        if res is not None and len(res) < len(chain_res):
            res_ranges = []
            if crnum_start < rnum_start:
                res_ranges.append(f'{crnum_start}-{rnum_start-1}')
            if crnum_end > rnum_end:
                res_ranges.append(f'{rnum_end+1}-{crnum_end}')
            if res_ranges:
                chain_spec = f'#{structure.id_string}' if chain_id is None else f'#{structure.id_string}/{chain_id}'
                cmd = f'delete {chain_spec}:{",".join(res_ranges)}'
                from chimerax.core.commands import run
                run(structure.session, cmd, log = log)

    if (trim is True or 'ligands' in trim) and hit['database'] == 'pdb100':
        if structure.num_residues > len(chain_res):
            # Delete non-polymer residues that are not in contact with trimmed chain residues.
            rstart, rend = (rnum_start, rnum_end) if trim_seq else (crnum_start, crnum_end)
            cmd = f'delete (#{structure.id_string}/{chain_id}:{rstart}-{rend} :> {ligand_range}) & #{structure.id_string} & (ligand | ions | solvent)'
            from chimerax.core.commands import run
            run(structure.session, cmd, log = log)

    return res

def residue_range(structure, hit, log):
    '''
    Return the residue range for the subset of residues used in the hit alignment.
    Foldseek tstart and tend target residue numbers are not PDB residue numbers.
    Instead they start at 1 and include only residues with atomic coordinates in the chain.
    '''
    db = hit['database']
    db_id = hit['database_id']
    chain_id = hit.get('chain_id')
    cname = '' if chain_id is None else f'chain {chain_id}'
    chain = _structure_chain_with_id(structure, chain_id)
    if chain is None:
        log.warning(f'Foldseek result {db} {db_id} does not have expected chain {chain_id}.')
        return None, None
    tstart, tend, tlen, tseq = hit['tstart'], hit['tend'], hit['tlen'], hit['tseq']
    res = alignment_residues(chain.existing_residues)
    seq = ''.join(r.one_letter_code for r in res)
    if len(res) != tlen:
        log.warning(f'Foldseek result {db} {db_id} {cname} number of residues {tlen} does not match residues in structure {len(res)}, sequences {tseq} and {seq}.')
        return None, None
    if seq != tseq:
        log.warning(f'Foldseek result {db_id} {cname} target sequence {tseq} does not match sequence from database {seq}.')
        # Sometimes ChimeraX reports X where Foldseek gives K.  ChimeraX bug #15653.
        for sc,tc in zip(seq, tseq):
            if sc != tc and sc != 'X' and tc != 'X':
                return None, None
    return res[tstart-1:tend], res

def _structure_chain_with_id(structure, chain_id):
    if chain_id is None:
        chains = structure.chains
    else:
        chains = [chain for chain in structure.chains if chain.chain_id == chain_id]
    chain = chains[0] if len(chains) == 1 else None
    return chain

# Foldseek recognized residues from
#   https://github.com/steineggerlab/foldseek/blob/master/src/strucclustutils/GemmiWrapper.cpp
_foldseek_accepted_3_letter_codes = set('ALA,ARG,ASN,ASP,CYS,GLN,GLU,GLY,HIS,ILE,LEU,LYS,MET,PHE,PRO,SER,THR,TRP,TYR,VAL,MSE,MLY,FME,HYP,TPO,CSO,SEP,M3L,HSK,SAC,PCA,DAL,CME,CSD,OCS,DPR,B3K,ALY,YCM,MLZ,4BF,KCX,B3E,B3D,HZP,CSX,BAL,HIC,DBZ,DCY,DVA,NLE,SMC,AGM,B3A,DAS,DLY,DSN,DTH,GL3,HY3,LLP,MGN,MHS,TRQ,B3Y,PHI,PTR,TYS,IAS,GPL,KYN,CSD,SEC,UNK'.split(','))

def alignment_residues(residues):
    '''
    Foldseek omits some non-standard residues.  It appears the ones it accepts are about 150
    hardcoded in a C++ file

        ​https://github.com/steineggerlab/foldseek/blob/master/lib/gemmi/resinfo.hpp
    '''
    rok = [r for r in residues if r.name in _foldseek_accepted_3_letter_codes]
    if len(rok) < len(residues):
        from chimerax.atomic import Residues
        residues = Residues(rok)
    if (residues.atoms.names == 'CA').sum() < len(residues):
        # Residue.principal_atom() does not return a CA if the N atom is missing.
        # For example, N missing in PDB 7w7g /B:654, bug #15668.
        from chimerax.atomic import Residues
        residues = Residues([r for r in residues if 'CA' in r.atoms.names])
    return residues
    
def alignment_residue_pairs(hit, aligned_res, query_chain):
    qstart, qend = hit['qstart'], hit['qend']
    qres_all = alignment_residues(query_chain.existing_residues)
    qres = qres_all[qstart-1:qend]
    qaln, taln = hit['qaln'], hit['taln']
    ti = qi = 0
    ati, aqi = [], []
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            taa_db = aligned_res[ti].one_letter_code
            if taa_db != taa:
                if taa_db != 'X':  # ChimeraX sometimes reports an X while foldseek a K, bug #15653
                    from chimerax.core.errors import UserError
                    raise UserError(f'Database structure {hit["database_full_id"]} sequence does not match Foldseek alignment sequence.  Database structure has residue {taa_db}{aligned_res[ti].number} where Foldseek alignment has amino acid type {taa} at position {ti+1}')
            qaa_db = qres[qi].one_letter_code
            if qaa_db != qaa:
                if qaa_db != 'X':  # ChimeraX sometimes reports an X while foldseek a K, bug #15653
                    from chimerax.core.errors import UserError
                    raise UserError(f'Query chain {query_chain.string()} sequence does not match Foldseek alignment sequence. Query chain has residue {qaa_db}{qres[qi].number} where Foldseek alignment has amino acid type {qaa} at position {qi+1}')
            ati.append(ti)
            aqi.append(qi)
        if taa != '-':
            ti += 1
        if qaa != '-':
            qi += 1
                    
    return aligned_res.filter(ati), qres.filter(aqi)

def query_alignment_range(hits):
    '''Return the range of query residue numbers (qstart, qend) that includes all the hit alignments.'''
    qstarts = []
    qends = []
    for hit in hits:
        qstarts.append(hit['qstart'])
        qends.append(hit['qend'])
    qstart, qend = min(qstarts), max(qends)
    return qstart, qend

def sequence_alignment(hits, qstart, qend):
    '''
    Return the sequence alignment between hits and query as a numpy 2D array of bytes.
    The first row is the query sequence with one subsequent row for each hit.
    There are no gaps in the query, and gaps in the hit sequences have 0 byte values.
    '''
    nhit = len(hits)
    qlen = qend - qstart + 1
    from numpy import zeros, byte
    alignment = zeros((nhit+1, qlen), byte)
    for h, hit in enumerate(hits):
        qaln, taln = hit['qaln'], hit['taln']
        qi = hit['qstart']
        for qaa, taa in zip(qaln, taln):
            if qaa != '-' and taa != '-' and qi >= qstart and qi <= qend:
                ai = qi-qstart
                alignment[0,ai] = ord(qaa)
                alignment[h+1,ai] = ord(taa)
            if qaa != '-':
                qi += 1
    return alignment

def alignment_coordinates(hits, qstart, qend):
    '''
    Return C-alpha atom coordinates for aligned sequences.
    Also return a mask indicating positions that are not sequence gaps.
    '''
    nhit = len(hits)
    qlen = qend - qstart + 1
    from numpy import zeros, float32
    xyz = zeros((nhit, qlen, 3), float32)
    mask = zeros((nhit, qlen), bool)
    for h, hit in enumerate(hits):
        qaln, taln = hit['qaln'], hit['taln']
        qi = hit['qstart']
        hi = hit['tstart']
        hxyz = hit_coords(hit)
        for qaa, taa in zip(qaln, taln):
            if qaa != '-' and taa != '-' and qi >= qstart and qi <= qend:
                ai = qi-qstart
                xyz[h,ai,:] = hxyz[hi-1]
                mask[h,ai] = True
            if qaa != '-':
                qi += 1
            if taa != '-':
                hi += 1
    return xyz, mask

def alignment_transform(res, query_res, cutoff_distance = None):
    qxyz = query_res.existing_principal_atoms.scene_coords
    txyz = res.existing_principal_atoms.coords
    p, rms, npairs = align_xyz_transform(txyz, qxyz, cutoff_distance = cutoff_distance)
    return p, rms, npairs

def align_xyz_transform(txyz, qxyz, cutoff_distance = None):
    if cutoff_distance is None or cutoff_distance <= 0:
        from chimerax.geometry import align_points
        p, rms = align_points(txyz, qxyz)
        npairs = len(txyz)
    else:
        from chimerax.std_commands.align import align_and_prune
        p, rms, indices = align_and_prune(txyz, qxyz, cutoff_distance)
#        look_for_more_alignments(txyz, qxyz, cutoff_distance, indices)
        npairs = len(indices)
    return p, rms, npairs

def look_for_more_alignments(txyz, qxyz, cutoff_distance, indices):
    sizes = [len(txyz), len(indices)]
    from numpy import ones
    mask = ones(len(txyz), bool)
#    from chimerax.std_commands.align import align_and_prune, IterationError
    while True:
        mask[indices] = 0
        if mask.sum() < 3:
            break
        try:
#            p, rms, indices = align_and_prune(txyz, qxyz, cutoff_distance, mask.nonzero()[0])
            p, rms, indices, close_mask = align_and_prune(txyz, qxyz, cutoff_distance, mask.nonzero()[0])
            sizes.append(len(indices))
            nclose = close_mask.sum()
            if nclose > len(indices):
                sizes.append(-nclose)
                sizes.append(-(mask*close_mask).sum())
        except IterationError:
            break
    print (' '.join(str(s) for s in sizes))

from chimerax.core.errors import UserError
class IterationError(UserError):
    pass

def align_and_prune(xyz, ref_xyz, cutoff_distance, indices = None):

    import numpy
    if indices is None:
        indices = numpy.arange(len(xyz))
    axyz, ref_axyz = xyz[indices], ref_xyz[indices]
    from chimerax.geometry import align_points
    p, rms = align_points(axyz, ref_axyz)
    dxyz = p*axyz - ref_axyz
    d2 = (dxyz*dxyz).sum(axis=1)
    cutoff2 = cutoff_distance * cutoff_distance
    i = d2.argsort()
    if d2[i[-1]] <= cutoff2:
        dxyz = p*xyz - ref_xyz
        d2 = (dxyz*dxyz).sum(axis=1)
        close_mask = (d2 <= cutoff2)
        return p, rms, indices, close_mask

    # cull 10% or...
    index1 = int(len(d2) * 0.9)
    # cull half the long pairings
    index2 = int(((d2 <= cutoff2).sum() + len(d2)) / 2)
    # whichever is fewer
    index = max(index1, index2)
    survivors = indices[i[:index]]

    if len(survivors) < 3:
        raise IterationError("Alignment failed;"
            " pruning distances > %g left less than 3 atom pairs" % cutoff_distance)
    return align_and_prune(xyz, ref_xyz, cutoff_distance, survivors)
    
def compute_rmsds(hits, query_xyz, cutoff_distance = None):
    for hit in hits:
        hxyz = hit_coords(hit)
        hi, qi = hit_residue_pairing(hit)
#        print(hit['database_full_id'])
        p, rms, npairs = align_xyz_transform(hxyz[hi], query_xyz[qi], cutoff_distance=cutoff_distance)
        hit['rmsd'] = rms
        hit['close'] = 100*npairs/len(hi)
        hit['cutoff_distance'] = cutoff_distance
        hit['coverage'] = 100 * len(qi) / len(query_xyz)

def hit_coords(hit):
    from numpy import array, float32
    xyz = array([float(x) for x in hit['tca'].split(',')], float32)
    n = len(xyz)//3
    hxyz = xyz.reshape((n,3))
    return hxyz

def hit_residue_pairing(hit, offset = True):
    qaln, taln = hit['qaln'], hit['taln']
    ti = qi = 0
    ati, aqi = [], []
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            ati.append(ti)
            aqi.append(qi)
        if taa != '-':
            ti += 1
        if qaa != '-':
            qi += 1
    from numpy import array, int32
    ati, aqi = array(ati, int32), array(aqi, int32)
    if offset:
        ati += hit['tstart']-1
        aqi += hit['qstart']-1
    return ati, aqi
    
def _show_ribbons(structure):
    for c in structure.chains:
        cres = c.existing_residues
        if not cres.ribbon_displays.any():
            cres.ribbon_displays = True
            cres.atoms.displays = False

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

    show_foldseek_hits(session, hit_lines, database, query_chain)

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
    
def show_foldseek_hits(session, hit_lines, database, query_chain = None,
                       trim = None, alignment_cutoff_distance = None):
    msg = f'Foldseek search for similar structures to {query_chain} in {database} found {len(hit_lines)} hits'
    session.logger.info(msg)

    hits = [parse_search_result(hit, database) for hit in hit_lines]
    if query_chain is not None:
        # Compute percent coverage and percent close C-alpha values per hit.
        qres = alignment_residues(query_chain.existing_residues)
        qxyz = qres.existing_principal_atoms.coords
        compute_rmsds(hits, qxyz, cutoff_distance = 2)
    from .gui import foldseek_panel, Foldseek
    fp = foldseek_panel(session)
    if fp:
        fp.show_results(hits, query_chain = query_chain, database = database,
                        trim = trim, alignment_cutoff_distance = alignment_cutoff_distance)
    else:
        fp = Foldseek(session, query_chain = query_chain, database = database,
                      hits = hits, trim = trim, alignment_cutoff_distance = alignment_cutoff_distance)
    return fp

def foldseek_open(session, hit_name, trim = None, align = True, alignment_cutoff_distance = None,
                  in_file_history = True, log = True):
    from .gui import foldseek_panel
    fp = foldseek_panel(session)
    if fp is None:
        from chimerax.core.errors import UserError
        raise UserError('No foldseek results are available')
    query_chain = fp.results_query_chain
    if trim is None:
        trim = fp.trim
    if alignment_cutoff_distance is None:
        alignment_cutoff_distance = fp.alignment_cutoff_distance
    for hit in fp.hits:
        if hit['database_full_id'] == hit_name:
            open_hit(session, hit, query_chain, trim = trim,
                     align = align, alignment_cutoff_distance = alignment_cutoff_distance,
                     in_file_history = in_file_history, log = log)
            break

def foldseek_show(session, hit_name):
    '''Show table row for this hit.'''
    hit, panel = _foldseek_hit_by_name(session, hit_name)
    if hit:
        fp.select_table_row(hit)

def _foldseek_hit_by_name(session, hit_name):
    from .gui import foldseek_panel
    fp = foldseek_panel(session)
    if fp is None:
        from chimerax.core.errors import UserError
        raise UserError('No foldseek results table is shown')
    for hit in fp.hits:
        if hit['database_full_id'] == hit_name:
            return hit, fp
    return None, None
    
def foldseek_pairing(session, hit_structure, color = None, radius = None,
                     halfbond_coloring = None):
    '''Show pseudobonds between foldseek hit and query paired residues.'''
    hit_name = hit_structure.name
    hit, panel = _foldseek_hit_by_name(session, hit_name)
    if hit is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any Foldseek hit {hit_name}')

    query_chain = panel.results_query_chain
    r_pairs = hit_and_query_residue_pairs(hit_structure, query_chain, hit)
    ca_pairs = []
    for hr, qr in r_pairs:
        hca = hr.principal_atom
        qca = qr.principal_atom
        if hca and qca:
            ca_pairs.append((hca, qca))

    if len(ca_pairs) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'Did not find any residues to pair for Foldseek hit {hit_name}')

    g = session.pb_manager.get_group(f'{hit_name} pairing')
    if g.id is not None:
        g.clear()

    for hca, qca in ca_pairs:
        b = g.new_pseudobond(hca, qca)
        if color is not None:
            b.color = color
        if radius is not None:
            b.radius = radius
        if halfbond_coloring is not None:
            b.halfbond = halfbond_coloring

    if g.id is None:
        session.models.add([g])

    return g

def hit_and_query_residue_pairs(hit_structure, query_chain, hit):
    ho = hit.get('aligned_residue_offsets')
    qo = hit.get('query_residue_offsets')
    if ho is None or qo is None:
        from chimerax.core.errors import UserError
        raise UserError(f'Could not find Foldseek alignment offsets for {hit_name}')
    
    qres = query_chain.residues
    hit_chain = _structure_chain_with_id(hit_structure, hit.get('chain_id'))
    hres = hit_chain.residues
    ahi, aqi = hit_residue_pairing(hit, offset = False)
    r_pairs = []
    for hi, qi in zip(ahi, aqi):
        hr = hres[ho[hi]]
        qr = qres[qo[qi]]
        if hr is not None and qr is not None:
            r_pairs.append((hr, qr))

    return r_pairs
        
def register_foldseek_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or, ListOf, FloatArg, SaveFolderNameArg, StringArg, Color8Arg
    from chimerax.atomic import ChainArg, StructureArg
    TrimArg = Or(ListOf(EnumOf(['chains', 'sequence', 'ligands'])), BoolArg)
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('database', EnumOf(foldseek_databases)),
                   ('trim', TrimArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('save_directory', SaveFolderNameArg),
                   ('wait', BoolArg),
                   ],
        synopsis = 'Search for proteins with similar folds using Foldseek web service'
    )
    register('foldseek', desc, foldseek, logger=logger)

    desc = CmdDesc(
        required = [('hit_name', StringArg)],
        keyword = [('trim', TrimArg),
                   ('align', BoolArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('in_file_history', BoolArg),
                   ('log', BoolArg),
                   ],
        synopsis = 'Open Foldseek result structure and align to query'
    )
    register('foldseek open', desc, foldseek_open, logger=logger)

    desc = CmdDesc(
        required = [('hit_name', StringArg)],
        synopsis = 'Show Foldseek result table row'
    )
    register('foldseek show', desc, foldseek_show, logger=logger)

    desc = CmdDesc(
        required = [('hit_structure', StructureArg)],
        keyword = [('color', Color8Arg),
                   ('radius', FloatArg),
                   ('halfbond_coloring', BoolArg),
                   ],
        synopsis = 'Show Foldseek result table row'
    )
    register('foldseek pairing', desc, foldseek_pairing, logger=logger)
