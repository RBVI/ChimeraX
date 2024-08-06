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
        hit_lines = results[self.database]
        results = FoldseekResults(hit_lines, self.database, self.chain)
        show_foldseek_results(self.session, results, trim = self.trim,
                              alignment_cutoff_distance = self.alignment_cutoff_distance)
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
        This may be done in a separate thread, so don't do any user interface calls that would use Qt.
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
            filename = self._results_file_name(dbname)
            self._save(filename, ''.join(hit_lines))

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
    from numpy import array, float32
    xyz = array([float(x) for x in values['tca'].split(',')], float32)
    n = len(xyz)//3
    values['tca'] = xyz.reshape((n,3))
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
    chain_res, ri_start, ri_end = _residue_range(structure, hit, structure.session.logger)
    if chain_res is None:
        name = hit.get('database_full_id')
        structure.session.logger.warning(f'Because of a sequence mismatch between the database structure {name} and the Foldseek output, the alignment and residue pairing of this structure is probably wrong, and ChimeraX did not trim the structure.')
        return None

    res = chain_res[ri_start:ri_end+1]
    if not trim:
        return res

    msg = []
    
    trim_lig = (trim is True or 'ligands' in trim) and hit['database'] == 'pdb100'
    chain_id = hit.get('chain_id')
    if (trim is True or 'chains' in trim) and chain_id is not None:
        if trim_lig:
            # Delete polymers in other chains, but not ligands from other chains.
            ocres = [c.existing_residues for c in tuple(structure.chains) if c.chain_id != chain_id]
            if ocres:
                from chimerax.atomic import concatenate
                ocres = concatenate(ocres)
        else:
            # Delete everything from other chain ids, including ligands.
            sres = structure.residues
            ocres = sres[sres.chain_ids != chain_id]
        if ocres:
            msg.append(f'{structure.num_chains-1} extra chains')
            ocres.delete()

    trim_seq = (trim is True or 'sequence' in trim)
    if trim_seq:
        # Trim end first because it changes the length of the res array.
        if ri_end < len(chain_res)-1:
            cterm = chain_res[ri_end+1:]
            msg.append(f'{len(cterm)} C-terminal residues')
            cterm.delete()
        if ri_start > 0:
            nterm = chain_res[:ri_start]
            msg.append(f'{len(nterm)} N-terminal residues')
            nterm.delete()

    if trim_lig:
        if structure.num_residues > len(chain_res):
            sres = structure.residues
            from chimerax.atomic import Residue
            npres = sres[sres.polymer_types == Residue.PT_NONE]
            npnear = _find_close_residues(npres, chain_res, ligand_range)
            npfar = npres.subtract(npnear)
            if npfar:
                msg.append(f'{len(npfar)} non-polymer residues more than {ligand_range} Angstroms away')
                npfar.delete()
                
    if log and msg:
        structure.session.logger.info(f'Deleted {", ".join(msg)}.')

    return res

def _find_close_residues(residues1, residues2, distance):
    a1 = residues1.atoms
    axyz1 = a1.scene_coords
    axyz2 = residues2.atoms.scene_coords
    from chimerax.geometry import find_close_points
    ai1, ai2 = find_close_points(axyz1, axyz2, distance)
    r1near = a1[ai1].unique_residues
    return r1near

def _residue_range(structure, hit, log):
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
        return None, None, None
    tstart, tend, tlen, tseq = hit['tstart'], hit['tend'], hit['tlen'], hit['tseq']
    res = alignment_residues(chain.existing_residues)
    seq = ''.join(r.one_letter_code for r in res)
    if len(res) != tlen:
        log.warning(f'Foldseek result {db} {db_id} {cname} number of residues {tlen} does not match residues in structure {len(res)}, sequences {tseq} and {seq}.')
        return None, None, None
    if seq != tseq:
        if seq.replace('?', 'X') != tseq:  # ChimeraX uses one letter code "?" for UNK residues while Foldseek uses "X"
            log.warning(f'Foldseek result {db_id} {cname} target sequence {tseq} does not match sequence from database {seq}.')
        # Sometimes ChimeraX reports X where Foldseek gives K.  ChimeraX bug #15653.
        for sc,tc in zip(seq, tseq):
            if sc != tc and sc != 'X' and tc != 'X':
                return None, None, None
    return res, tstart-1, tend-1

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
    qatoms = query_res.find_existing_atoms('CA')
    qxyz = qatoms.scene_coords
    tatoms = res.find_existing_atoms('CA')
    txyz = tatoms.coords
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

def hit_coords(hit):
    hxyz = hit['tca']
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

    results = FoldseekResults(hit_lines, database, query_chain)
    show_foldseek_results(session, results)

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
    
def show_foldseek_results(session, results, trim = None, alignment_cutoff_distance = None):
    msg = f'Foldseek search for similar structures to {results.query_chain} in {results.database} found {len(results.hits)} hits'
    session.logger.info(msg)

    from .gui import foldseek_panel
    fp = foldseek_panel(session, create = True)
    fp.set_trim_options(trim)
    fp.set_alignment_cutoff_option(alignment_cutoff_distance)
    fp.show_results(results)
    return fp

class FoldseekResults:
    def __init__(self, hit_lines, database, query_chain):
        self.hits = [parse_search_result(hit, database) for hit in hit_lines]
        self.database = database
        self._query_chain = query_chain
        self._query_residues = None
        self._query_alignment_range = None
        self._sequence_alignment_array = None
        self._alignment_coordinates = None
        self._lddt_score_array = None

    @property
    def num_hits(self):
        return len(self.hits)

    @property
    def session(self):
        qc = self.query_chain
        return qc.structure.session if qc else None

    @property
    def query_chain(self):
        qc = self._query_chain
        if qc is not None and qc.structure is None:
            self._query_chain = qc = None
        return qc

    @property
    def query_residues(self):
        qres = self._query_residues
        if qres is None:
            self._query_residues = qres = alignment_residues(self._query_chain.existing_residues)
        return qres

    def query_alignment_range(self):
        '''Return the range of query residue numbers (qstart, qend) that includes all the hit alignments.'''
        qar = self._query_alignment_range
        if qar is not None:
            return qar
        qstarts = []
        qends = []
        for hit in self.hits:
            qstarts.append(hit['qstart'])
            qends.append(hit['qend'])
        qar = qstart, qend = min(qstarts), max(qends)
        return qar

    def sequence_alignment_array(self):
        saa = self._sequence_alignment_array
        if saa is not None:
            return saa
        qstart, qend = self.query_alignment_range()
        self._sequence_alignment_array = saa = sequence_alignment(self.hits, qstart, qend)
        return saa

    def alignment_coordinates(self):
        ac = self._alignment_coordinates
        if ac is not None:
            return ac
        qstart, qend = self.query_alignment_range()
        self._alignment_coordinates = hits_xyz, hits_mask = alignment_coordinates(self.hits, qstart, qend)
        return hits_xyz, hits_mask
    
    def compute_rmsds(self, alignment_cutoff_distance = None):
        if self.query_chain is None:
            return False
        # Compute percent coverage and percent close C-alpha values per hit.
        qres = alignment_residues(self.query_chain.existing_residues)
        qatoms = qres.find_existing_atoms('CA')
        query_xyz = qatoms.coords
        for hit in self.hits:
            hxyz = hit_coords(hit)
            hi, qi = hit_residue_pairing(hit)
            p, rms, npairs = align_xyz_transform(hxyz[hi], query_xyz[qi],
                                                 cutoff_distance=alignment_cutoff_distance)
            hit['rmsd'] = rms
            hit['close'] = 100*npairs/len(hi)
            hit['cutoff_distance'] = alignment_cutoff_distance
            hit['coverage'] = 100 * len(qi) / len(query_xyz)
        return True

    def set_coverage_attribute(self):
        if self.query_chain is None  or getattr(self, '_coverage_attribute_set', False):
            return
        self._coverage_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_coverage', "Foldseek", attr_type = int)

        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend:
                ai = ri-(qstart-1)
                alignment_array = self.sequence_alignment_array()
                count = count_nonzero(alignment_array[1:,ai])
            else:
                count = 0
            r.foldseek_coverage = count

    def set_conservation_attribute(self):
        if self.query_chain is None  or getattr(self, '_conservation_attribute_set', False):
            return
        self._conservation_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_conservation', "Foldseek", attr_type = float)

        alignment_array = self.sequence_alignment_array()
        seq, count, total = _consensus_sequence(alignment_array)
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend:
                ai = ri-(qstart-1)
                conservation = count[ai] / total[ai]
            else:
                conservation = 0
            r.foldseek_conservation = conservation

    def set_entropy_attribute(self):
        if self.query_chain is None or getattr(self, '_entropy_attribute_set', False):
            return
        self._entropy_attribute_set = True
        
        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_entropy', "Foldseek", attr_type = float)

        alignment_array = self.sequence_alignment_array()
        entropy = _sequence_entropy(alignment_array)
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend:
                ai = ri-(qstart-1)
                r.foldseek_entropy = entropy[ai]

    def set_lddt_attribute(self):
        if self.query_chain is None or getattr(self, '_lddt_attribute_set', False):
            return
        self._lddt_attribute_set = True

        from chimerax.atomic import Residue
        Residue.register_attr(self.session, 'foldseek_lddt', "Foldseek", attr_type = float)

        lddt_scores = self.lddt_scores()
        qstart, qend = self.query_alignment_range()
        from numpy import count_nonzero
        for ri,r in enumerate(self.query_residues):
            if ri >= qstart-1 and ri <= qend:
                ai = ri-(qstart-1)
                alignment_array = self.sequence_alignment_array()
                nscores = count_nonzero(alignment_array[1:,ai])
                ave_lddt = lddt_scores[:,ai].sum() / nscores
            else:
                ave_lddt = 0
            r.foldseek_lddt = ave_lddt

    def lddt_scores(self):
        lddt_scores = self._lddt_score_array
        if lddt_scores is None:
            qstart, qend = self.query_alignment_range()
            qres = self.query_residues
            qatoms = qres.find_existing_atoms('CA')
            query_xyz = qatoms.coords[qstart-1:qend,:]
            hits_xyz, hits_mask = self.alignment_coordinates()
            from . import lddt
            lddt_scores = lddt.local_distance_difference_test(query_xyz, hits_xyz, hits_mask)
            self._lddt_score_array = lddt_scores
        return lddt_scores

    def take_snapshot(self, session, flags):
        data = {'hits': self.hits,
                'query_chain': self.query_chain,
                'database': self.database,
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        return FoldseekResults(data['hits'], data['database'], data['query_chain'])

# -----------------------------------------------------------------------------
#
def _consensus_sequence(alignment_array):
    seqlen = alignment_array.shape[1]
    from numpy import count_nonzero, bincount, argmax, empty, byte, int32
    seq = empty((seqlen,), byte)
    count = empty((seqlen,), int32)
    total = empty((seqlen,), int32)
    for i in range(seqlen):
        aa = alignment_array[:,i]
        total[i] = count_nonzero(aa)
        bc = bincount(aa)
        mi = argmax(bc[1:]) + 1
        seq[i] = mi
        count[i] = bc[mi]
    return seq, count, total

# -----------------------------------------------------------------------------
#
def _sequence_entropy(alignment_array):
    seqlen = alignment_array.shape[1]
    from numpy import bincount, empty, float32, array, int32
    entropy = empty((seqlen,), float32)
    aa_1_letter_codes = 'ARNDCEQGHILKMFPSTWYV'
    aa_char = array([ord(c) for c in aa_1_letter_codes], int32)
    bins = aa_char.max() + 1
    for i in range(seqlen):
        aa = alignment_array[:,i]
        bc = bincount(aa, minlength = bins)[aa_char]
        entropy[i] = _entropy(bc)
    return entropy
    
# -----------------------------------------------------------------------------
#
def _entropy(bin_counts):
    total = bin_counts.sum()
    if total == 0:
        return 0.0
    nonzero = (bin_counts > 0)
    p = bin_counts[nonzero] / total
    from numpy import log2
    e = -(p*log2(p)).sum()
    return e

def foldseek_open(session, hit_name, trim = None, align = True, alignment_cutoff_distance = None,
                  in_file_history = True, log = True):
    from .gui import foldseek_panel
    fp = foldseek_panel(session)
    if fp is None or len(fp.hits) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No foldseek results are available')
    query_chain = fp.results.query_chain
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

def foldseek_scroll_to(session, hit_name):
    '''Show table row for this hit.'''
    hit, panel = _foldseek_hit_by_name(session, hit_name)
    if hit:
        panel.select_table_row(hit)

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

    query_chain = panel.results.query_chain
    r_pairs = hit_and_query_residue_pairs(hit_structure, query_chain, hit)
    ca_pairs = []
    for hr, qr in r_pairs:
        hca = hr.find_atom('CA')
        qca = qr.find_atom('CA')
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
    register('foldseek scrollto', desc, foldseek_scroll_to, logger=logger)

    desc = CmdDesc(
        required = [('hit_structure', StructureArg)],
        keyword = [('color', Color8Arg),
                   ('radius', FloatArg),
                   ('halfbond_coloring', BoolArg),
                   ],
        synopsis = 'Show Foldseek result table row'
    )
    register('foldseek pairing', desc, foldseek_pairing, logger=logger)
