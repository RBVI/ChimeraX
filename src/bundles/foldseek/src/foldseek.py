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

def foldseek(session, structure, database = 'pdb100', trim = True):

    mmcif_string = _mmcif_as_string(structure)
    results = foldseek_web_query(mmcif_string, databases = [database])
    msg = 'Foldseek ' + ', '.join([f'{database} {len(search_results)} hits'
                                   for database, search_results in results.items()])
    session.logger.info(msg)

    if database == 'pdb100':
        hit_lines = results[database]
        hits = [parse_search_result(hit, database) for hit in hit_lines[:5]]
        from .gui import FoldseekPDBResults
        FoldseekPDBResults(session, query_structure = structure, pdb_hits = hits, trim = trim)

        '''
        hlines = []
        for hit in hits[:5]:
            values = parse_search_result(hit, database)
            hmsg = f'PDB {values["pdb_id"]} chain {values["pdb_chain_id"]} assembly {values["pdb_assembly_id"]} description {values["pdb_description"]}'
            hlines.append(hmsg)
        session.logger.info('\n'.join(hlines))
        '''
    
    '''
    For monomer search
    curl -X POST -F q=@PATH_TO_FILE -F 'mode=3diaa' -F 'database[]=afdb50' -F 'database[]=afdb-swissprot' -F 'database[]=afdb-proteome' -F 'database[]=bfmd' -F 'database[]=cath50' -F 'database[]=mgnify_esm30' -F 'database[]=pdb100' -F 'database[]=gmgcl_id' https://search.foldseek.com/api/ticket

    For multimer search.
    curl -X POST -F q=@PATH_TO_FILE -F 'mode=complex-3diaa' -F 'database[]=bfmd' -F 'database[]=pdb100' https://search.foldseek.com/api/ticket

    https://search.mmseqs.com/docs/
    '''

foldseek_databases = ['pdb100', 'afdb50', 'afdb-swissprot', 'afdb-proteome',
                      'bfmd', 'cath50', 'mgnify_esm30', 'gmgcl_id']

def foldseek_web_query(mmcif_string,
                       foldseek_url = 'https://search.foldseek.com/api',
#                       databases = ['pdb100', 'afdb50', 'afdb-swissprot' 'afdb-proteome'],
                       databases = ['pdb100'],
                       status_interval = 1.0):
    '''
    Use an https post to start at search using the Foldseek REST API.
    Then wait for results, download them, and return a dictionary mapping
    database name to results as m8 format tab-separated values.
    '''

    query_url = foldseek_url + '/ticket'
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
    
    # poll until the job was successful or failed
    status_url = foldseek_url + f'/ticket/{ticket_id}'
    while True:
        r = requests.get(status_url)
        status = r.json()

        if status['status'] == "ERROR":
            from chimerax.core.errors import UserError
            raise UserError(f'FoldSeek jobs failed {status}')

        if status['status'] == "COMPLETE":
            break

        # wait a short time between poll requests
        import time
        time.sleep(status_interval)

    # get results
    result_url = foldseek_url + f'/result/download/{ticket_id}'
    results = requests.get(result_url, stream = True)

    # Result is a tar gzip file containing a .m8 tab-separated value file.
    import tempfile
    rfile = tempfile.NamedTemporaryFile(prefix = 'foldseek_results_', suffix = '.tar.gz')
    with rfile as f:
        for chunk in results.iter_content(chunk_size=128):
            f.write(chunk)
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

def _mmcif_as_string(structure):
    import tempfile
    with tempfile.NamedTemporaryFile(prefix = 'foldseek_mmcif_', suffix = '.cif') as f:
        from chimerax.mmcif.mmcif_write import write_mmcif
        write_mmcif(structure.session, f.name, models = [structure])
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
    if database != 'pdb100':
        from chimerax.core.errors import UserError
        raise UserError('Foldseek command currently can only parse pdb100 database results.')
    
    fields = line.split('\t')
    field_names = ['query', 'theader', 'pident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'prob', 'evalue', 'bits', 'qlen', 'tlen', 'qaln', 'taln', 'tca', 'tseq', 'taxid', 'taxname']
    values = dict(zip(field_names, fields))
    values.update(parse_pdb100_theader(values['theader']))
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
        'pdb_chain_id': theader[iz+8:id],
        'pdb_description': theader[id+1:],
    }
    return values

def open_pdb_hit(session, pdb_hit, query_structure, trim = True):
    from chimerax.core.commands import run
    pdb_id = pdb_hit["pdb_id"]
    structure = run(session, f'open {pdb_id}')[0]
    aligned_res = trim_pdb_structure(structure, pdb_hit, trim)

    # TODO: Align the model to the query structure using Foldseek alignment
#    run(session, f'matchmaker #{structure.id_string} to #{query_structure.id_string}')
    p = alignment_transform(aligned_res, pdb_hit, session.logger)
    if p is not None:
        structure.position = p

def trim_pdb_structure(structure, pdb_hit, trim):
    pdb_id = pdb_hit["pdb_id"]
    chain_id = pdb_hit["pdb_chain_id"]
    
    if trim == 'chains' or trim is True:
        if len(structure.chains) > 1:
            cmd = f'delete #{structure.id_string} & ~/{chain_id}'
            from chimerax.core.commands import run
            run(structure.session, cmd)

    res, all = pdb_residue_range(structure, pdb_hit)
    if trim == 'sequence' or trim is True:
        if not all and res is not None:
            rnum_start, rnum_end = res[0].number, res[-1].number
            cmd = f'delete #{structure.id_string}/{chain_id} & ~:{rnum_start}-{rnum_end}'
            from chimerax.core.commands import run
            run(structure.session, cmd)

    return res

def pdb_residue_range(structure, pdb_hit):
    '''
    Foldseek tstart and tend target residue numbers are not PDB residue numbers.
    Instead they start at 1 and include only residues with atomic coordinates in the PDB chain.
    This routine converts these to PDB residue numbers.
    '''
    pdb_id = pdb_hit['pdb_id']
    chain_id = pdb_hit['pdb_chain_id']
    schains = [chain for chain in structure.chains if chain.chain_id == chain_id]
    chain = schains[0] if len(schains) == 1 else None
    if chain is None:
        structure.session.logger.warning(f'Foldseek result PDB {pdb_id} does not have a chain {chain_id}.')
        return None, None
    tstart, tend, tlen, tseq = int(pdb_hit['tstart']), int(pdb_hit['tend']), int(pdb_hit['tlen']), pdb_hit['tseq']
    res = chain.existing_residues
    if len(res) != tlen:
        structure.session.logger.warning(f'Foldseek result PDB {pdb_id} chain {chain_id} number of residues {tlen} does not match residues in structure {len(res)}.')
        return None, None
    seq = ''.join(r.one_letter_code for r in res)
    if seq != tseq:
        structure.session.logger.warning(f'Foldseek result PDB {pdb_id} chain {chain_id} target sequence {tseq} does not match PDB structure existing residue sequence {seq}.')
        return None, None
    all = (tend == len(tseq))
    return res[tstart-1:tend], all

def alignment_transform(res, pdb_hit, log):
    if res is None:
        return None
    pdb_id = pdb_hit['pdb_id']
    chain_id = pdb_hit['pdb_chain_id']
    tca = [float(x) for x in pdb_hit['tca'].split(',')]
    from numpy import array, float64
    tca = array(tca, float64).reshape((len(tca)//3, 3))
    tstart, tend = int(pdb_hit['tstart']), int(pdb_hit['tend'])
    tca = tca[tstart-1:tend,:]	# tca includes coordinates for all residues even those not used in alignment
    if len(tca) != len(res):
        log.warning(f'Foldseek result PDB {pdb_id} chain {chain_id} has {len(tca)} transform coordinates for {len(res)} residues.')
        return None
    ca_xyz = res.existing_principal_atoms.coords
    from chimerax.geometry import align_points
    p, rms = align_points(ca_xyz, tca)
    log.info(f'Foldseek alignment for PDB {pdb_id} chain {chain_id} of {len(res)} C-alpha atoms has RMSD {rms}')
    print ('alignment matrix', p.matrix)
    return p
    
def register_foldseek_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, BoolArg, Or
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('structure', AtomicStructureArg)],
        keyword = [('database', EnumOf(foldseek_databases)),
                   ('trim', Or(EnumOf(['chains', 'sequence']), BoolArg)),
                   ],
        synopsis = 'Search for proteins with similar folds using Foldseek web service'
    )
    
    register('foldseek', desc, foldseek, logger=logger)
