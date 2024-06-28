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

def foldseek(session, chain, database = 'pdb100', trim = True, alignment_cutoff_distance = 2.0):

    mmcif_string = _mmcif_as_string(chain)
    results = foldseek_web_query(mmcif_string, databases = [database])
    # Use cached 8jnb results for developing user interface so I don't have to wait for server every test.
#    with open('/Users/goddard/ucsf/chimerax/src/bundles/foldseek_example/alis_pdb100.m8', 'r') as mfile:
#        results = {database: mfile.readlines()}
    msg = 'Foldseek ' + ', '.join([f'{database} {len(search_results)} hits'
                                   for database, search_results in results.items()])
    session.logger.info(msg)

    if database == 'pdb100':
        hit_lines = results[database]
        hits = [parse_search_result(hit, database) for hit in hit_lines]
        from .gui import FoldseekPDBResults
        FoldseekPDBResults(session, query_chain = chain, pdb_hits = hits, trim = trim,
                           alignment_cutoff_distance = alignment_cutoff_distance)

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

def open_pdb_hit(session, pdb_hit, query_chain, trim = True, alignment_cutoff_distance = 2.0):
    from chimerax.core.commands import run
    pdb_id = pdb_hit["pdb_id"]
    structure = run(session, f'open {pdb_id}')[0]
    aligned_res = trim_pdb_structure(structure, pdb_hit, trim)
    _show_ribbons(structure)
    
    # Align the model to the query structure using Foldseek alignment.
    # Foldseek server does not return transform by default, so compute from sequence alignment.
    res, query_res = alignment_residue_pairs(pdb_hit, aligned_res, query_chain)
    p, rms, npairs = alignment_transform(res, query_res, alignment_cutoff_distance)
    structure.position = p
    chain_id = pdb_hit["pdb_chain_id"]
    msg = f'Alignment of {pdb_id} chain {chain_id} to query has RMSD {"%.3g" % rms} using {npairs} of {len(res)} paired residues'
    if alignment_cutoff_distance is not None and alignment_cutoff_distance > 0:
        msg += f' within cutoff distance {alignment_cutoff_distance}'
    session.logger.info(msg)

def trim_pdb_structure(structure, pdb_hit, trim):
    pdb_id = pdb_hit["pdb_id"]
    chain_id = pdb_hit["pdb_chain_id"]
    
    if trim is True or 'chains' in trim:
        if len(structure.chains) > 1:
            cmd = f'delete #{structure.id_string} & ~/{chain_id}'
            from chimerax.core.commands import run
            run(structure.session, cmd)

    res, chain_res = pdb_residue_range(structure, pdb_hit)
    rnum_start, rnum_end = res[0].number, res[-1].number
    crnum_start, crnum_end = chain_res[0].number, chain_res[-1].number

    trim_seq = (trim is True or 'sequence' in trim)
    if trim_seq:
        if res is not None and len(res) < len(chain_res):
            res_ranges = []
            if crnum_start < rnum_start:
                res_ranges.append(f'{crnum_start}-{rnum_start-1}')
            if crnum_end > rnum_end:
                res_ranges.append(f'{rnum_end+1}-{crnum_end}')
            cmd = f'delete #{structure.id_string}/{chain_id}:{",".join(res_ranges)}'
            from chimerax.core.commands import run
            run(structure.session, cmd)

    if trim is True or 'ligands' in trim:
        if structure.num_residues > len(chain_res):
            # Delete non-polymer residues that are not in contact with trimmed chain residues.
            rstart, rend = (rnum_start, rnum_end) if trim_seq else (crnum_start, crnum_end)
            cmd = f'delete #{structure.id_string}/{chain_id}:{rstart}-{rend} :> 3 & (ligand | ions | solvent)'
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
    return res[tstart-1:tend], res

def alignment_residue_pairs(pdb_hit, aligned_res, query_chain):
    qstart, qend = int(pdb_hit['qstart']), int(pdb_hit['qend'])
    qres = query_chain.existing_residues[qstart-1:qend]
    qaln, taln = pdb_hit['qaln'], pdb_hit['taln']
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
