# vim: set expandtab shiftwidth=4 softtabstop=4:

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

'''
Development of this command is in progress, Jan 24, 2024.

I noticed that the selected residues when loading structures from cncalcium/dimers for Hiten Madhani
seemed wrong -- the residue weren't at interfaces at all.

It would be useful to have an option that only gives the highest ranked Alphafold model in the output
table for each dimer from among the 5 predicted dimers.

It would be useful to have an option to load all the table structures.

It would be nice to see the protein-protein interaction graph of the found dimers using matplotlib.

I'd like to be able to align all the loaded dimers by aligning equivalent monomers.  This needs
some scoring method whenever there are loops in the protein-protein interaction network to
try to build a consistent complex.  If there are multiple dimers for one pair of proteins we
would like to take the most consistent one with the other structures.

For large proteins (e.g. > 2000 amino acids) I'd like to be able chop them in half, but then
reassemble the pieces using monomer predictions of those large proteins.
'''

# -----------------------------------------------------------------------------
#
def alphafold_interfaces(session, directory, distance = 4, max_pae = 5, min_conf_pairs = 10,
                         open = False, results_file = 'interfaces.csv', short_names = True):
    '''
    Evaluate AlphaFold predicted aligned error scores at dimer interfaces for
    several predictions produced by the alphafold dimers command.  Make a table
    of results for the confident interfaces with links to load and show interfaces.
    '''
    iclist = interface_confidence(session, directory, distance = distance, max_pae = max_pae,
                                  results_file = results_file)
    if short_names:
        use_short_sequence_names(iclist)
    giclist = group_by_sequences(iclist)
    ciclist = [ic for ic in iclist if ic.num_confident_pairs >= min_conf_pairs]
    gciclist = group_by_sequences(ciclist)

    msg = (f'{len(gciclist)} of {len(giclist)} dimers have {min_conf_pairs} or more confident residue interactions\n' +
           f'spanning <= {distance} Angstroms with predicted aligned error <= {max_pae} Angstroms.')
    session.logger.info(msg)

    if gciclist:
        html = interface_table(gciclist, directory, distance, max_pae, min_conf_pairs)
        session.logger.info(html, is_html = True)

    if open:
        paths = [gic[0].pdb_path for gic in gciclist]
        open_structures(session, paths)

# -----------------------------------------------------------------------------
#
def interface_confidence(session, directory, distance = 4, max_pae = 5, results_file = None):
    if results_file is not None:
        from os.path import join, isfile
        results_path = join(directory, results_file)
        if isfile(results_path):
            iclist = _read_interfaces(results_path)
            if iclist and iclist[0].distance == distance and iclist[0].max_pae == max_pae:
                return iclist

    from os import listdir
    pdb_files = [f for f in listdir(directory) if f.endswith('.pdb') or f.endswith('.cif')]

    iclist = []
    for fi, file in enumerate(pdb_files):
        from os.path import join
        pdb_path = join(directory, file)
        pae_file = _pae_filename_from_structure_filename(file)
        pae_path = join(directory, pae_file)
        dc = dimer_confidence(session, pdb_path, pae_path, distance=distance, max_pae=max_pae)
        iclist.append(dc)
        session.logger.status(f'Evaluating {dc.sequence_names} ({fi+1} of {len(pdb_files)})')

    if results_file is not None:
        _write_interfaces(results_path, iclist)

    return iclist

# -----------------------------------------------------------------------------
#
def _pae_filename_from_structure_filename(structure_filename):
    if structure_filename.endswith('.pdb'):
        # Colabfold / AlphaFold 2
        pae_filename = structure_filename.replace('unrelaxed', 'scores').replace('.pdb','.json')
    elif structure_filename.endswith('.cif'):
        pae_filename = structure_filename.replace('model', 'full_data').replace('.cif','.json')
    return pae_filename

# -----------------------------------------------------------------------------
#
def dimer_confidence(session, pdb_path, pae_path, distance = 4, max_pae = 5):
    '''
    Load AlphaFold PDB dimer predictions and PAE files and score the confidence of interactions between
    the two chains.
    '''
    if pdb_path.endswith('.pdb'):
        from chimerax.pdb import open_pdb
        m = open_pdb(session, pdb_path, log_info = False)[0][0]
    elif pdb_path.endswith('.cif'):
        from chimerax.mmcif import open_mmcif
        m = open_mmcif(session, pdb_path, log_info = False)[0][0]
    chains = m.chains
    if len(chains) == 2:
        res1 = chains[0].existing_residues
        res2 = chains[1].existing_residues
    elif len(chains) == 3:
        from chimerax.atomic import concatenate
        res1 = concatenate((chains[0].existing_residues, chains[1].existing_residues))
        res2 = chains[2].existing_residues
    else:
        raise ValueError(f'Got {len(chains)} chains, expected 2 or 3')

    n1, n2 = len(res1), len(res2)
    r1, r2, ni1, ni2 = contacting_residue_pairs(res1, res2, distance)
    npair = len(r1)
    allres = m.residues
    r1i = allres.indices(r1)
    r2i = allres.indices(r2)
    from chimerax.alphafold.pae import AlphaFoldPAE
    pae = AlphaFoldPAE(pae_path, m).pae_matrix
    from numpy import minimum
    pae12 = minimum(pae[r1i,r2i], pae[r2i,r1i])
    nconf = (pae12 <= max_pae).sum()

    # Make residue specifier for interface residues for aligning to other models.
    low_pae = (pae12 <= max_pae)
#    r1nums, r2nums = r1[low_pae].numbers, r2[low_pae].numbers
    r1nums, r2nums = list(set(r1[low_pae].numbers)), list(set(r2[low_pae].numbers))
    r1nums.sort()
    r2nums.sort()
    m.delete()
    return InterfaceConfidence(pdb_path, pae_path, n1, n2, ni1, ni2, npair, nconf, r1nums, r2nums,
                               distance, max_pae)

# -----------------------------------------------------------------------------
#
class InterfaceConfidence:
    def __init__(self, pdb_path, pae_path, n1, n2, ni1, ni2, npair, nconf, r1nums, r2nums,
                 distance, max_pae):
        self.pdb_path = pdb_path
        self.sequence_names = sequences_name_from_file_name(pdb_path)
        self.pae_path = pae_path
        self.distance = distance
        self.max_pae = max_pae
        self.num_residues1 = n1
        self.num_residues2 = n2
        self.num_interface_residues1 = ni1
        self.num_interface_residues2 = ni2
        self.num_interface_residue_pairs = npair
        self.num_confident_pairs = nconf
        self.interface_residue_numbers1 = r1nums  # Confident, no duplicates
        self.interface_residue_numbers2 = r2nums  # Confident, no duplicates

    csv_header = '# pdb_path, pae_path, distance, max_pae, num_res1, num_res2, num_interface_res1, num_interface_res2, num_interface_res_pairs, num_confident_pairs, interface_res_num1, interface_res_num2'

    def to_csv(self):
        from os.path import basename
        pdb_file = basename(self.pdb_path)
        pae_file = basename(self.pae_path)
        d, maxp, n1, n2, ni1, ni2, npair, nconf, r1nums, r2nums = (
            self.distance, self.max_pae,
            self.num_residues1, self.num_residues2,
            self.num_interface_residues1, self.num_interface_residues2,
            self.num_interface_residue_pairs, self.num_confident_pairs,
            self.interface_residue_numbers1, self.interface_residue_numbers2)
        r1list, r2list = ' '.join(str(i) for i in r1nums), ' '.join(str(i) for i in r2nums)
        line = f'{pdb_file},{pae_file},{d},{maxp},{n1},{n2},{ni1},{ni2},{npair},{nconf},{r1list},{r2list}'
        return line

    @staticmethod
    def from_csv(line, directory):
        fields = line.strip().split(',')
        pdb_file, pae_file = fields[0:2]
        from os.path import join
        pdb_path = join(directory, pdb_file)
        pae_path = join(directory, pae_file)
        distance, max_pae = [float(x) for x in fields[2:4]]
        n1, n2, ni1, ni2, npair, nconf = [int(x) for x in fields[4:10]]
        r1nums, r2nums = [(tuple(int(rnum) for rnum in rnums.split(' ')) if rnums else ())
                          for rnums in fields[10:12]]
        ic = InterfaceConfidence(pdb_path, pae_path, n1, n2, ni1, ni2, npair, nconf, r1nums, r2nums,
                                 distance, max_pae)
        return ic
    
# -----------------------------------------------------------------------------
#
def sequences_name_from_file_name(file):
    from os.path import basename
    file = basename(file)
    if file.endswith('.pdb'):
        i = file.find('_unrelaxed')  # AlphaFold 2
        seqs_name = file[:i] if i >= 0 else file
    elif file.endswith('.cif'):
        i = file.find('_model')  # AlphaFold 3
        seqs_name = file[:i] if i >= 0 else file
        if seqs_name.startswith('fold_'):
            seqs_name = seqs_name[5:]
    return seqs_name

# -----------------------------------------------------------------------------
#
def _write_interfaces(results_path, iclist):
    lines = [ic.to_csv() for ic in iclist]
    with open(results_path, 'w') as file:
        file.write(InterfaceConfidence.csv_header + '\n')
        file.write('\n'.join(lines))

# -----------------------------------------------------------------------------
#
def _read_interfaces(results_path):
    with open(results_path, 'r') as file:
        lines = file.readlines()
    from os.path import dirname
    directory = dirname(results_path)
    iclist = [InterfaceConfidence.from_csv(line, directory) for line in lines[1:]]
    return iclist

# -----------------------------------------------------------------------------
#
def interface_table(giclist, directory, distance, max_pae, min_conf_pairs):

    lines = ['<table border=1 cellpadding=2 cellspacing=0>',
             '<th>Sequences<th>Models<th>Confident pairs<th>#Res1<th> #Res2']
    open_cmds = []
    for gic in giclist:
        ic = gic[0]
        seqs_name = ic.sequence_names
        quoted_pdb_path = '\\"' + ic.pdb_path + '\\"' if ' ' in ic.pdb_path else ic.pdb_path
        quoted_pae_path = '\\"' + ic.pae_path + '\\"' if ' ' in ic.pae_path else ic.pae_path
        res1 =  ','.join(str(i) for i in ic.interface_residue_numbers1)
        res2 =  ','.join(str(i) for i in ic.interface_residue_numbers2)
        res = f'/A:{res1}/B:{res2}'
        cmds = [
            f'open {quoted_pdb_path}',
            f"rename last-opened '{seqs_name}'",
            'color last-opened bypolymer',
            f'select {res} & last-opened',
            f'alphafold pae last-opened file {quoted_pae_path}',
            f'alphafold contacts last-opened & /A to last-opened & /B distance {ic.distance} maxPae {ic.max_pae}'
        ]
        cmd = ' ; '.join(cmds)
        # TODO: Make models link open all models for a dimer align them, name them by their rank, show PAE lines.
        lines.extend(
            [f'<tr><td><a href="cxcmd:{cmd}">{seqs_name}</a>',
             f'<td align=center>{len(gic)}',
             f'<td align=center>{ic.num_confident_pairs}',
             f'<td align=center>{len(ic.interface_residue_numbers1)}',
             f'<td align=center>{len(ic.interface_residue_numbers2)}']
        )
        open_cmds.extend(cmds)
    lines.append('</table>')
    open_cmds = [((cmd + ' plot false') if cmd.startswith('alphafold pae') else cmd)
                 for cmd in open_cmds]  # Don't show PAE plots for every model.
    open_cmds.extend(['wait 1', 'tile', 'label all models'])
    # TODO: Add separate tile and and model slider links.
    # TODO: How about an align link?
    # How about a show graph link.
    # Seems like I'd be better off with a special purpose gui.
    # That would avoid the annoying log behavior that the table gets lost above many commands.
    open_cmd = ' ; '.join(open_cmds)
    lines.append(f'<a href="cxcmd:{open_cmd}">Open best</a>.')
    hide_cmd = 'hide @@bfactor<=50 ribbon'
    show_cmd = 'show @@bfactor<=50 ribbon'
    lines.append(f'<a href="cxcmd:{hide_cmd}">Hide</a> or <a href="cxcmd:{show_cmd}">show</a> disordered loops (pLLDT &lt;= 50).')
    msg = '\n'.join(lines)
    return msg

# -----------------------------------------------------------------------------
#
def short_sequence_names(long_sequence_names):
    '''
    Return map of sequence pair name to shorter sequence names which
    are single components separated by "_" in the original sequence names
    that uniquely name a sequence
    '''
    seq_pairs = long_sequence_names
    seq_names = set()
    for seq_pair in seq_pairs:
        seqs = seq_pair.split('.')
        if len(seqs) == 2:
            for seq in seqs:
                seq_names.add(seq)
    seqmap = {}
    for seq in seq_names:
        parts = seq.split('_')
        uparts = [part for part in parts if len([s for s in seq_names if part in s]) == 1]
        if uparts:
            uparts.sort(key = lambda p: len([c for c in p if c.isdigit()]))
            seqmap[seq] = uparts[0]
    seq_pair_map = {}
    for seq_pair in seq_pairs:
        seqs = seq_pair.split('.')
        if len(seqs) == 2:
            seq_pair_map[seq_pair] = ' '.join(seqmap.get(seq, seq) for seq in seqs)
    return seq_pair_map

# -----------------------------------------------------------------------------
#
def use_short_sequence_names(iclist):
    seq_names = [ic.sequence_names for ic in iclist]
    short = short_sequence_names(seq_names)
    for ic in iclist:
        if ic.sequence_names in short:
            ic.sequence_names = short[ic.sequence_names]

# -----------------------------------------------------------------------------
#
def contacting_residue_pairs(res1, res2, distance):
    a1, a2 = res1.atoms, res2.atoms
    from chimerax.geometry import find_close_points
    i1, i2 = find_close_points(a1.coords, a2.coords, distance)
    r1, r2 = a1[i1].unique_residues, a2[i2].unique_residues
    ra2 = r2.atoms
    axyz2 = ra2.coords
    rpairs = []
    for r in r1:
        i1, i2 = find_close_points(r.atoms.coords, axyz2, distance)
        rpairs.extend((r,rc) for rc in ra2[i2].unique_residues)
    from chimerax.atomic import Residues
    rpair1 = Residues([rp1 for rp1, rp2 in rpairs])
    rpair2 = Residues([rp2 for rp1, rp2 in rpairs])
    return rpair1, rpair2, len(r1), len(r2)

# -----------------------------------------------------------------------------
#
def group_by_sequences(iclist):
    smap = {}
    for ic in iclist:
        seq_names = ic.sequence_names
        if seq_names in smap:
            smap[seq_names].append(ic)
        else:
            smap[seq_names] = [ic]
    for dic in smap.values():
        dic.sort(key = lambda ic: file_name_rank(ic.pdb_path))
    seq_names = list(smap.keys())
    seq_names.sort()
    giclist = [smap[sname] for sname in seq_names]
    return giclist

# -----------------------------------------------------------------------------
#
def file_name_rank(path):
    from os.path import basename
    filename = basename(path)
    i = filename.find('_rank_')
    rank = int(filename[i+6:i+9]) if i >= 0 else 0
    return rank

# -----------------------------------------------------------------------------
#
def open_structures(session, paths):
    from chimerax.core.commands import run, quote_path_if_necessary
    qpaths = ' '.join(quote_path_if_necessary(p) for p in paths)
    cmd = f'open {qpaths}'
    run(session, cmd)
    
# -----------------------------------------------------------------------------
#
def register_alphafold_interfaces_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFolderNameArg, FloatArg, IntArg, BoolArg, SaveFileNameArg
    desc = CmdDesc(
        required = [('directory', OpenFolderNameArg)],
        keyword = [('distance', FloatArg),
                   ('max_pae', FloatArg),
                   ('min_conf_pairs', IntArg),
                   ('open', BoolArg),
                   ('results_file', SaveFileNameArg),
                   ('short_names', BoolArg)],
        synopsis = 'Evaluate AlphaFold PAE at interfaces for a directory of predicted models'
    )
    register('alphafold interfaces', desc, alphafold_interfaces, logger=logger)
