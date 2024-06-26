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

def foldseek(session, chain, database = 'pdb100', trim = None, alignment_cutoff_distance = None, wait = False):
    '''Submit a Foldseek search for similar structures and display results in a table.'''
    FoldseekWebQuery(session, chain, database=database,
                     trim=trim, alignment_cutoff_distance=alignment_cutoff_distance, wait=wait)

foldseek_databases = ['pdb100', 'afdb50', 'afdb-swissprot', 'afdb-proteome']

#foldseek_databases = ['pdb100', 'afdb50', 'afdb-swissprot', 'afdb-proteome',
#                      'bfmd', 'cath50', 'mgnify_esm30', 'gmgcl_id']

class FoldseekWebQuery:

    def __init__(self, session, chain, database = 'pdb100', trim = True,
                 alignment_cutoff_distance = 2.0, wait = False,
                 foldseek_url = 'https://search.foldseek.com/api'):
        self.session = session
        self.chain = chain
        self.database = database
        self.trim = trim
        self.alignment_cutoff_distance = alignment_cutoff_distance
        self.foldseek_url = foldseek_url
        self._last_check_time = None
        self._status_interval = 1.0	# Seconds.  Frequency to check for search completion
        
        # Use cached 8jnb results for developing user interface so I don't have to wait for server every test.
#        with open('/Users/goddard/ucsf/chimerax/src/bundles/foldseek_example/alis_pdb100.m8', 'r') as mfile:
#            results = {database: mfile.readlines()}
#        self.report_results(results)
#        return
        
        mmcif_string = _mmcif_as_string(chain)
        ticket_id = self.submit_query(mmcif_string, databases = [database])
        if wait:
            self.wait_for_results(ticket_id)
            results = self.download_results(ticket_id)
            self.report_results(results)
        else:
            self.poll_for_results(ticket_id)

    def report_results(self, results):
        hit_summary = ', '.join([f'in {self.database} found {len(search_results)} hits'
                                 for database, search_results in results.items()])
        msg = f'Foldseek search for similar structures to {self.chain} {hit_summary}'
        self.session.logger.info(msg)

        hit_lines = results[self.database]
        hits = [parse_search_result(hit, self.database) for hit in hit_lines]
        from .gui import foldseek_panel, Foldseek
        fp = foldseek_panel(self.session)
        if fp:
            fp.show_results(hits, query_chain = self.chain, database = self.database,
                            trim = self.trim, alignment_cutoff_distance = self.alignment_cutoff_distance)
        else:
            Foldseek(self.session, query_chain = self.chain, database = self.database,
                     hits = hits, trim = self.trim, alignment_cutoff_distance = self.alignment_cutoff_distance)

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
        r = requests.post(query_url, files=files, data=data)
        if r.status_code != 200:
            error_msg = r.text
            from chimerax.core.errors import UserError
            raise UserError(f'Foldseek search failed: {error_msg}')

        ticket = r.json()
        ticket_id = ticket['id']
        return ticket_id

    def wait_for_results(self, ticket_id):
        # poll until the job is successful or fails
        import time
        while not self.check_for_results(ticket_id):
            time.sleep(self._status_interval)

    def poll_for_results(self, ticket_id):
        '''Poll for results during new frame callback and report them when complete.'''
        self.session.triggers.add_handler('new frame', lambda *args, t=ticket_id: self._new_frame_callback(t))

    def _new_frame_callback(self, ticket_id):
        from time import time
        t = time()
        if self._last_check_time and t - self._last_check_time < self._status_interval:
            return
        self._last_check_time = t

        from chimerax.core.errors import UserError
        try:
            complete = self.check_for_results(ticket_id)
        except UserError as e:
            self.session.logger.warning(str(e))
            return 'delete handler'

        if complete:
            results = self.download_results(ticket_id)
            self.report_results(results)
            return 'delete handler'

    def check_for_results(self, ticket_id):
        # poll until the job was successful or failed
        status_url = self.foldseek_url + f'/ticket/{ticket_id}'
        import requests
        r = requests.get(status_url)
        status = r.json()

        if status['status'] == "ERROR":
            from chimerax.core.errors import UserError
            raise UserError(f'FoldSeek jobs failed {status}')

        if status['status'] == "COMPLETE":
            return True

        return False
    
    def download_results(self, ticket_id):
        '''
        Return a dictionary mapping database name to results as m8 format tab-separated values.
        '''
        result_url = self.foldseek_url + f'/result/download/{ticket_id}'
        import requests
        results = requests.get(result_url, stream = True)

        # Result is a tar gzip file containing a .m8 tab-separated value file.
        import tempfile
        rfile = tempfile.NamedTemporaryFile(prefix = 'foldseek_results_', suffix = '.tar.gz')
        size = 0
        with rfile as f:
            for chunk in results.iter_content(chunk_size=16384):
                f.write(chunk)
                size += len(chunk)
                self.session.logger.status(f'Reading Foldseek results {size} bytes')
            f.flush()

            # Extract tar file making a dictionary of results for each database searched
            m8_results = {}
            import tarfile
            tfile = tarfile.open(f.name)
            for filename in tfile.getnames():
                mfile = tfile.extractfile(filename)
                dbname = filename.replace('alis_', '').replace('.m8', '')
                m8_results[dbname] = [line.decode('utf-8') for line in mfile.readlines()]

        return m8_results

def _mmcif_as_string(chain):
    structure = chain.structure.copy()
    cchain = [c for c in structure.chains if c.chain_id == chain.chain_id][0]
    extra_residues = structure.residues - cchain.existing_residues
    extra_residues.delete()
    import tempfile
    with tempfile.NamedTemporaryFile(prefix = 'foldseek_mmcif_', suffix = '.cif') as f:
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
    
    values = {
        'pdb_id': theader[:4],
        'pdb_assembly_id': theader[ia+9:iz],
        'chain_id': theader[iz+8:id],
        'description': theader[id+1:],
    }
    values['database_id'] = values['pdb_id']
    values['database_full_id'] = values['pdb_id'] + '_' + values['chain_id']

    return values

def parse_alphafold_theader(theader):
    '''Example: "AF-A7E3S4-F1-model_v4 RAF proto-oncogene serine/threonine-protein kinase"'''
    if not theader.startswith('AF-'):
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not contain expected string "AF-"')

    im = theader.find('-F1-model_v')
    if im == -1:
        from chimerax.core.errors import UserError
        raise UserError(f'Foldseek results target header "{theader}" did not contain expected string "-F1-model_v"')
    
    values = {
        'alphafold_id': theader[3:im],
        'alphafold_version': theader[im+11:im+12],
        'description': theader[im+13:],
    }
    values['database_id'] = values['alphafold_id']
    values['database_full_id'] = values['alphafold_id']
    
    return values
        
def open_hit(session, hit, query_chain, trim = True, alignment_cutoff_distance = 2.0):
    db_id = hit.get('database_id')
    from_db = 'from alphafold' if hit['database'].startswith('afdb') else ''
    from chimerax.core.commands import run
    structure = run(session, f'open {db_id} {from_db}')[0]
    aligned_res = trim_structure(structure, hit, trim)
    _show_ribbons(structure)
    
    # Align the model to the query structure using Foldseek alignment.
    # Foldseek server does not return transform by default, so compute from sequence alignment.
    res, query_res = alignment_residue_pairs(hit, aligned_res, query_chain)
    p, rms, npairs = alignment_transform(res, query_res, alignment_cutoff_distance)
    structure.position = p
    chain_id = hit.get('chain_id')
    cname = '' if chain_id is None else f'chain {chain_id}'
    msg = f'Alignment of {db_id}{cname} to query has RMSD {"%.3g" % rms} using {npairs} of {len(res)} paired residues'
    if alignment_cutoff_distance is not None and alignment_cutoff_distance > 0:
        msg += f' within cutoff distance {alignment_cutoff_distance}'
    session.logger.info(msg)

def trim_structure(structure, hit, trim):
    res, chain_res = residue_range(structure, hit, structure.session.logger)
    rnum_start, rnum_end = res[0].number, res[-1].number
    crnum_start, crnum_end = chain_res[0].number, chain_res[-1].number

    if not trim:
        return res
    
    chain_id = hit.get('chain_id')
    
    if (trim is True or 'chains' in trim) and chain_id is not None:
        if len(structure.chains) > 1:
            cmd = f'delete #{structure.id_string} & ~/{chain_id}'
            from chimerax.core.commands import run
            run(structure.session, cmd)

    trim_seq = (trim is True or 'sequence' in trim)
    if trim_seq:
        if res is not None and len(res) < len(chain_res):
            res_ranges = []
            if crnum_start < rnum_start:
                res_ranges.append(f'{crnum_start}-{rnum_start-1}')
            if crnum_end > rnum_end:
                res_ranges.append(f'{rnum_end+1}-{crnum_end}')
            chain_spec = f'#{structure.id_string}' if chain_id is None else f'#{structure.id_string}/{chain_id}'
            cmd = f'delete {chain_spec}:{",".join(res_ranges)}'
            from chimerax.core.commands import run
            run(structure.session, cmd)

    if (trim is True or 'ligands' in trim) and hit['database'] == 'pdb100':
        if structure.num_residues > len(chain_res):
            # Delete non-polymer residues that are not in contact with trimmed chain residues.
            rstart, rend = (rnum_start, rnum_end) if trim_seq else (crnum_start, crnum_end)
            cmd = f'delete (#{structure.id_string}/{chain_id}:{rstart}-{rend} :> 3) & #{structure.id_string} & (ligand | ions | solvent)'
            from chimerax.core.commands import run
            run(structure.session, cmd)

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
    schains = structure.chains if chain_id is None else [chain for chain in structure.chains
                                                         if chain.chain_id == chain_id]
    chain = schains[0] if len(schains) == 1 else None
    if chain is None:
        log.warning(f'Foldseek result {db} {db_id} does not have expected chain {chain_id}.')
        return None, None
    tstart, tend, tlen, tseq = hit['tstart'], hit['tend'], hit['tlen'], hit['tseq']
    res = chain.existing_residues
    if len(res) != tlen:
        log.warning(f'Foldseek result {db} {db_id} {cname} number of residues {tlen} does not match residues in structure {len(res)}.')
        return None, None
    seq = ''.join(r.one_letter_code for r in res)
    if seq != tseq:
        log.warning(f'Foldseek result {db_id} {cname} target sequence {tseq} does not match sequence from database {seq}.')
        return None, None
    all = (tend == len(tseq))
    return res[tstart-1:tend], res

def alignment_residue_pairs(hit, aligned_res, query_chain):
    qstart, qend = hit['qstart'], hit['qend']
    qres = query_chain.existing_residues[qstart-1:qend]
    qaln, taln = hit['qaln'], hit['taln']
    ti = qi = 0
    ati, aqi = [], []
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            if aligned_res[ti].one_letter_code != taa:
                from chimerax.core.errors import UserError
                raise UserError(f'Amino acid at aligned sequence position {ti} is {taa} which does not match target PDB structure residue {aligned_res[ti].one_letter_code}{aligned_res[ti].number}')
            if qres[qi].one_letter_code != qaa:
                from chimerax.core.errors import UserError
                raise UserError(f'Amino acid at aligned sequence position {qi} is {qaa} which does not match query PDB structure residue {qres[qi].one_letter_code}{qres[qi].number}')
            ati.append(ti)
            aqi.append(qi)
        if taa != '-':
            ti += 1
        if qaa != '-':
            qi += 1
                    
    return aligned_res.filter(ati), qres.filter(aqi)

def alignment_transform(res, query_res, cutoff_distance = None):
    # TODO: Do iterative pruning to get better core alignment.
    qxyz = query_res.existing_principal_atoms.scene_coords
    txyz = res.existing_principal_atoms.coords
    if cutoff_distance is None or cutoff_distance <= 0:
        from chimerax.geometry import align_points
        p, rms = align_points(txyz, qxyz)
        npairs = len(txyz)
    else:
        from chimerax.std_commands.align import align_and_prune
        p, rms, indices = align_and_prune(txyz, qxyz, cutoff_distance)
        npairs = len(indices)
    return p, rms, npairs

def _show_ribbons(structure):
    for c in structure.chains:
        cres = c.existing_residues
        if not cres.ribbon_displays.any():
            cres.ribbon_displays = True
            cres.atoms.displays = False
            
def register_foldseek_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or, ListOf, FloatArg
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [('chain', ChainArg)],
        keyword = [('database', EnumOf(foldseek_databases)),
                   ('trim', Or(ListOf(EnumOf(['chains', 'sequence', 'ligands'])), BoolArg)),
                   ('alignment_cutoff_distance', FloatArg),
                   ],
        synopsis = 'Search for proteins with similar folds using Foldseek web service'
    )
    
    register('foldseek', desc, foldseek, logger=logger)
