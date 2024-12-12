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

def diffplot(session, embedding_path = None, alignment = None, residues = None, structures = None,
             drop = 'structures', coords_path = None, output_embedding_path = None,
             plot = True, cluster = None, replace = True, verbose = False):

    if embedding_path is not None:
        pdb_names, umap_xy = _read_embedding(embedding_path)
        if cluster:
            from chimerax.umap import k_means_clusters, color_by_cluster
            cluster_numbers = k_means_clusters(umap_xy, cluster)
            colors = color_by_cluster(cluster_numbers)
            _report_clusters(pdb_names, cluster_numbers, session.logger)
        else:
            colors = None
        _plot_embedding(session, pdb_names, umap_xy, colors, replace=replace)
        return
        
    if alignment is None:
        alignment = _get_open_sequence_alignment(session)

    from chimerax.core.errors import UserError
    if structures is None:
        structures = _structures_matching_alignment_sequence_names(session, alignment)
        if len(structures) == 0:
            raise UserError('No structures with names matching alignment sequence names')

    chains = [_chain_from_structure_name(s) for s in structures]
    if len(chains) == 0:
        raise UserError('No chains matching alignment sequence names')

    if residues is None:
        residues = chains[0].existing_residues

    _associate_chains_to_sequence_alignment_by_name(alignment, chains)

    aligned_residues, atoms, res_columns = _residue_alignment_columns(alignment, residues)

    if drop == 'residues':
        log = session.logger if verbose else None
        dcols = _drop_columns(chains, res_columns, alignment, log)
        if dcols:
            dcolstring = _concise_columns(sorted(list(dcols)))
            ucols = len(res_columns) - len(dcols)
            msg = f'\nDropping {len(dcols)} residues in columns {dcolstring} because not all structures had them. Using {ucols} residues.'
            session.logger.info(msg)
            if len(dcols) == len(res_columns):
                raise UserError('No residues were in all the structures')
            from numpy import array
            mask = array([(c not in dcols) for c in res_columns], bool)
            aligned_residues = aligned_residues[mask]
            atoms = atoms[mask]
            res_columns = [c for c in res_columns if c not in dcols]

    rxyz = atoms.scene_coords

    diffs = _chain_atom_motions(chains, res_columns, alignment, aligned_residues, rxyz, session, verbose)

    if coords_path is not None:
        with open(coords_path, 'w') as f:
            lines = [c.structure.name + ',' + ','.join('%.3f'%x for x in xyz.flat) for c,xyz in diffs]
            f.write('\n'.join(lines))

    from chimerax.umap import install_umap, umap_embed
    install_umap(session)

    coord_diffs = [diff_xyz for chain, diff_xyz in diffs]
    xyz_diffs = _flatten_xyz_lists(coord_diffs)
    umap_xy = umap_embed(xyz_diffs.copy())

    diff_chains = [chain for chain, diff_xyz in diffs]
    pdb_names = [chain.structure.name for chain in diff_chains]
    if output_embedding_path is not None:
        _write_embedding(pdb_names, umap_xy, output_embedding_path)

    if plot:
        if cluster:
            from chimerax.umap import k_means_clusters, color_by_cluster
            cluster_numbers = k_means_clusters(umap_xy, cluster)
#            cluster_numbers = k_means_clusters(xyz_diffs, cluster)  # Test if UMAP gives same clusters as k-means. Yes!
            colors = color_by_cluster(cluster_numbers)
            _report_clusters(diff_chains, pdb_names, cluster_numbers, session.logger)
        else:
            colors = None
        _plot_embedding(session, pdb_names, umap_xy, colors, replace=replace)
        
def _get_open_sequence_alignment(session):
    from chimerax.core.errors import UserError
    na = len(session.alignments.alignments) if hasattr(session, 'alignments') else 0
    if na == 0:
        raise UserError('No sequence alignment specified.')
    elif na > 1:
        raise UserError(f'No sequence alignment specified and {na} alignments are open.')
    alignment = session.alignments.alignments[0]
    return alignment

def _structures_matching_alignment_sequence_names(session, alignment):
    from chimerax.atomic import all_atomic_structures
    nsmap = {s.name:s for s in all_atomic_structures(session)}
    structures = []
    found = set()
    for aseq in alignment.seqs:
        s = nsmap.get(aseq.name)
        if s is not None and s not in found:
            structures.append(s)
            found.add(s)
    return structures

def _chain_from_structure_name(structure):
    if '_' not in structure.name:
        if structure.num_chains == 1:
            return structure.chains[0]
    else:
        chain_id = structure.name.split('_')[1]
        chains = [c for c in structure.chains if c.chain_id == chain_id]
        if len(chains) != 1:
            from chimerax.core.errors import UserError
            raise UserError(f'Structure {structure} has {len(chains)} chains with chain id {chain_id}')
        return chains[0]

def _associate_chains_to_sequence_alignment_by_name(alignment, chains):
    alignment_sequences = {seq.name:seq for seq in alignment.seqs}

    # Make sure association makes an exact match, using Needleman-Wunsch if needed, bug #15106
    from chimerax.seqalign.settings import settings
    original_aer = settings.assoc_error_rate
    settings.assoc_error_rate = 10000

    for chain in chains:
        name = chain.structure.name
        if name not in alignment_sequences:
            from chimerax.core.errors import UserError
            raise UserError(f'Chain {name} not in alignment')
        aseq = alignment_sequences[name]
        if chain not in alignment.associations or alignment.associations[chain] != aseq:
            alignment.associate(chain, aseq)

    settings.assoc_error_rate = original_aer	# Restore original association error tolerance
    
    # Remove associations of all other chains
    chainset = set(chains)
    for chain in tuple(alignment.associations.keys()):
        if chain not in chainset:
            alignment.disassociate(chain)

def _residue_alignment_columns(alignment, residues):
    ruc = residues.unique_chains
    from chimerax.core.errors import UserError
    if len(ruc) != 1:
        raise UserError(f'Must specify residues in a single chain, got {len(residues)} in {len(ruc)} chains')
    ref_chain = ruc[0]

    if ref_chain not in alignment.associations:
        raise UserError(f'Chain {ref_chain} is not in alignment {alignment.description}')
    
# TODO: Need to fix automatic BLAST associations so correct structure is matched with corresponding row
#       in the sequence alignment.

    ref_seq = alignment.associations[ref_chain]
    ref_match = ref_seq.match_maps[ref_chain]
    rp = ref_match.res_to_pos
    aligned = [r for r in residues if r in rp]
    if len(aligned) < len(residues):
        print (f'Only {len(aligned)} of {len(residues)} residues are aligned')

    from chimerax.atomic import Residues
    resa = Residues(aligned)
    rpa = resa.existing_principal_atoms
    if len(rpa) < len(resa):
        print(f'Only {len(rpa)} of {len(resa)} have CA atom')
        resa = Residues([r for r,a in zip(resa, resa.principal_atoms) if a is not None])
    columns = [ref_seq.ungapped_to_gapped(rp[r]) for r in resa]

    return resa, rpa, columns

def _alignment_atoms(chain, alignment, columns):
    if chain not in alignment.associations:
        from chimerax.core.errors import UserError
        raise UserError(f'Chain {chain} is not associated with any sequence in alignment {alignment.description}')
    aseq = alignment.associations[chain]
    mm = aseq.match_maps[chain]
    cares = mm.pos_to_res
    ugcolumns = [aseq.gapped_to_ungapped(col) for col in columns]
    missing_col = []
    for gcol, ugcol in zip(columns, ugcolumns):
        if ugcol not in cares:
            missing_col.append(gcol)
    if missing_col:
        mcolstring = _concise_columns(missing_col)
        missing_msg = f'Chain {chain} has no residue in alignment columns {mcolstring}'
        return None, missing_col, missing_msg
    from chimerax.atomic import Residues, Atoms
    cres = Residues([cares.get(col) for col in ugcolumns])
    cra = cres.principal_atoms
    if None in cra:
        nmissing = cra.count(None)
        missing_col = [columns[i] for i,a in enumerate(cra) if a is None]
        return None, missing_col, f'Chain {chain} is missing {nmissing} CA atoms'
#    print (chain.structure.name, 'residues', ''.join([r.one_letter_code for r in cres]))
    return Atoms(cra), [], None

def _drop_columns(chains, res_columns, alignment, log):
    '''Find columns where not all chains aligned or have C-alpha atom.'''
    missing = set()
    msgs = []
    for chain in chains:
        atoms, missing_columns, missing_message = _alignment_atoms(chain, alignment, res_columns)
        if missing_columns:
            missing.update(missing_columns)
        if missing_message:
            msgs.append(missing_message)
    if log:
        log.info('\n'.join(msgs))

    return missing

def _chain_atom_motions(chains, res_columns, alignment, aligned_residues, reference_xyz, session, verbose):
    diffs = []
    excluded_chains = []
    all_atoms = []
    for chain in chains:
        atoms, missing_columns, missing_message = _alignment_atoms(chain, alignment, res_columns)
        if missing_message:
            excluded_chains.append((chain, missing_message))
        else:
            diff_xyz = atoms.scene_coords - reference_xyz
            diffs.append((chain, diff_xyz))
            all_atoms.append(atoms)

    from chimerax.atomic import Atoms, concatenate
    diffplot_atoms = concatenate(all_atoms, Atoms)
    from chimerax.basic_actions.cmd import name_frozen
    from chimerax.core.objects import Objects
    name_frozen(session, 'diffatoms', Objects(atoms = diffplot_atoms))
    
    column_nums = _concise_columns(res_columns)
    from chimerax.atomic import concise_residue_spec
    residue_nums = concise_residue_spec(session, aligned_residues)
    struct_name = aligned_residues[0].structure.name
    comp_spec = ' '.join(chain.string(style='command') for chain, diff_xyz in diffs)
    show_hide_sel = (f'(<a href="cxcmd:show {comp_spec} model">show</a>'
                     f' / <a href="cxcmd:hide {comp_spec} model">hide</a>'
                     f' / <a href="cxcmd:select {comp_spec}">select</a>)')
    session.logger.info(f'<br>Compared {len(diffs)} structures {show_hide_sel} at {len(aligned_residues)} residues of {struct_name}<br>'
                        f'{residue_nums}<br><br>'
                        f'\nAlignment columns {column_nums}<br>',
                        is_html = True)

    if excluded_chains:
        excl_names = ' '.join(c.structure.name for c,msg in excluded_chains)
        excl_spec = ' '.join(c.string(style='command') for c,msg in excluded_chains)
        show_hide_sel = (f'(<a href="cxcmd:show {excl_spec} model">show</a>'
                         f' / <a href="cxcmd:hide {excl_spec} model">hide</a>'
                         f' / <a href="cxcmd:select {excl_spec}">select</a>)')
        session.logger.info(f'Excluded {len(excluded_chains)} structures {show_hide_sel} because they did not have alignment matches for all the residues being compared: {excl_names}', is_html=True)
        if verbose:
            session.logger.info('\n'.join(msg for c,msg in excluded_chains))

    return diffs

def _concise_columns(column_numbers):
    ranges = []
    last_c = None
    for c in column_numbers:
        c += 1	# Switch from 0-base to 1-base indexing
        if last_c is None:
            start = last_c = c
        elif c > last_c + 1:
            ranges.append(f'{start}-{last_c}' if last_c > start else f'{start}')
            start = last_c = c
        else:
            last_c = c
    if last_c is not None:
        ranges.append(f'{start}-{c}' if c > start else f'{start}')
    return ' '.join(ranges)

def _flatten_xyz_lists(xyz_lists):
    from numpy import array, float64
    nstruct, nxyz = len(xyz_lists), len(xyz_lists[0])
    coords = array(xyz_lists, float64).reshape((nstruct, 3*nxyz))
    return coords

def _write_embedding(pdb_names, umap_xy, embedding_path):
    lines = [f'{pdb_name},{xy[0]},{xy[1]}' for pdb_name, xy in zip(pdb_names, umap_xy)]
    with open(embedding_path, 'w') as f:
        f.write('\n'.join(lines))
    
def _read_embedding(embedding_path):
    pdb_names = []
    xy = []
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            fields = line.split(',')
            pdb_names.append(fields[0])
            xy.append(tuple(float(x) for x in fields[1:3]))
    from numpy import array, float64
    umap_xy = array(xy, float64)
    return pdb_names, umap_xy

def _report_clusters(chains, pdb_names, cluster_num, logger):
    n = max(cluster_num) + 1
    cluster_pdbs = [[] for i in range(n)]
    cluster_chains = [[] for i in range(n)]
    for i,k in enumerate(cluster_num):
        cluster_pdbs[k].append(pdb_names[i])
        cluster_chains[k].append(chains[i])
    lines = []
    for i,(chains,pdbs) in enumerate(zip(cluster_chains, cluster_pdbs)):
        cluster_spec = ' '.join(c.string(style='command') for c in chains)
        show_hide_sel = (f'(<a href="cxcmd:show {cluster_spec} model">show</a>'
                         f' / <a href="cxcmd:hide {cluster_spec} model">hide</a>'
                         f' / <a href="cxcmd:select {cluster_spec}">select</a>)')

        lines.append(f'<b>Cluster {i+1}</b> {show_hide_sel}: {" ".join(pdbs)}<br><br>')
    logger.info('\n' + '\n\n'.join(lines), is_html = True)

def _color_models(session, pdb_names, colors):
    from chimerax.atomic import all_atomic_structures
    smap = {s.name:s for s in all_atomic_structures(session)}
    for pdb_name, color in zip(pdb_names, colors):
        if pdb_name in smap:
            structure = smap[pdb_name]
            structure.residues.ribbon_colors = color
            atoms = structure.atoms
            c_atoms = atoms[atoms.element_names == 'C']
            c_atoms.colors = color

def _plot_embedding(session, pdb_names, umap_xy, colors = None, replace = False):
    if colors is not None:
        _color_models(session, pdb_names, colors)
    if replace and hasattr(session, '_last_diffplot') and session._last_diffplot.tool_window.ui_area is not None:
        plot = session._last_diffplot
    else:
        plot = StructurePlot(session, title = 'Structure UMAP Plot', tool_name = 'DiffPlot')
        session._last_diffplot = plot
    plot.set_nodes(pdb_names, umap_xy, colors)
    return plot

from chimerax.umap import UmapPlot
class StructurePlot(UmapPlot):
    def mouse_click(self, node, event):
        '''Control click handler.'''
        if node is None:
            return
        from chimerax.atomic import all_atomic_structures
        for s in all_atomic_structures(self.session):
            if s.name == node.name:
                s.display = True
    def fill_context_menu(self, menu, item):
        if item is not None and self._have_colors:
            self.add_menu_entry(menu, 'Show cluster', lambda self=self, item=item: self._show_cluster(item))
            self.add_menu_entry(menu, 'Hide cluster', lambda self=self, item=item: self._hide_cluster(item))
            self.add_menu_entry(menu, 'Select cluster', lambda self=self, item=item: self._select_cluster(item))
            self.add_menu_entry(menu, f'Show {item.name}', lambda self=self, item=item: self._show_structure(item))
            self.add_menu_entry(menu, f'Hide {item.name}', lambda self=self, item=item: self._show_structure(item))
            self.add_menu_entry(menu, f'Select {item.name}', lambda self=self, item=item: self._show_structure(item))
        else:
            self.add_menu_entry(menu, f'Show all structures', self._show_all_structures)
            self.add_menu_entry(menu, f'Hide extra chains', self._hide_extra_chains)
            self.add_menu_entry(menu, f'Thin ribbons', self._thin_ribbons)
            self.add_menu_entry(menu, f'Show comparison atoms', self._show_comparison_atoms)
            self.add_menu_entry(menu, f'Hide comparison atoms', self._hide_comparison_atoms)
            self.add_menu_entry(menu, f'Select comparison atoms', self._select_comparison_atoms)
            self.add_menu_entry(menu, 'Save Plot As...', self.save_plot_as)
    def _show_cluster(self, node):
        self._run_cluster_command(node, 'show %s model')
    def _hide_cluster(self, node):
        self._run_cluster_command(node, 'hide %s model')
    def _select_cluster(self, node):
        self._run_cluster_command(node, 'select %s')
    def _run_cluster_command(self, node, command):
        spec = self._node_cluster_spec(node)
        if spec:
            self._run_command(command % spec)
    def _node_cluster_spec(self, node):
        from chimerax.atomic import all_atomic_structures
        smap = {m.name:m for m in all_atomic_structures(self.session)}
        structures_spec = ' '.join(smap[n.name].string(style='command') for n in self.nodes
                                   if n.color == node.color and n.name in smap)
        return structures_spec
    def _show_structure(self, node):
        self._run_structure_command(node, 'show %s model')
    def _hide_structure(self, node):
        self._run_structure_command(node, 'hide %s model')
    def _select_structure(self, node):
        self._run_structure_command(node, 'select %s')
    def _run_structure_command(self, node, command):
        spec = self._node_structure_spec(node)
        if spec:
            self._run_command(command % spec)
    def _node_structure_spec(self, node):
        from chimerax.atomic import all_atomic_structures
        smap = {m.name:m for m in all_atomic_structures(self.session)}
        structure_spec = smap[node.name].string(style='command') if node.name in smap else None
        return structure_spec
    def _show_all_structures(self):
        self._run_all_structures_command('show %s model')
    def _hide_extra_chains(self):
        self._run_all_structures_command('diffplot hideextrachains %s')
    def _thin_ribbons(self):
        self._run_all_structures_command('cartoon style %s width 0.3 thickness 0.1')
    def _show_comparison_atoms(self):
        self._show_ribbon_backbone_atoms()
        self._ribbon_through_c_alphas()
        self._run_command('show diffatoms')
    def _show_ribbon_backbone_atoms(self):
        for s in self._node_structures():
            s.residues.ribbon_hide_backbones = False
    def _ribbon_through_c_alphas(self):
        smooth = 0
        for s in self._node_structures():
            s.residues.ribbon_adjusts = smooth
    def _hide_comparison_atoms(self):
        self._run_command('hide diffatoms')
    def _select_comparison_atoms(self):
        self._run_command('select diffatoms')
    def _run_all_structures_command(self, command):
        spec = self._all_structures_spec()
        if spec:
            self._run_command(command % spec)
    def _all_structures_spec(self):
        structures_spec = ' '.join(s.string(style='command') for s in self._node_structures())
        return structures_spec
    def _node_structures(self):
        from chimerax.atomic import all_atomic_structures
        smap = {m.name:m for m in all_atomic_structures(self.session)}
        structures = [smap[node.name] for node in self.nodes if node.name in smap]
        return structures
    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)

def associate_by_name(session, alignment, structures = None):

    for chain in tuple(alignment.associations.keys()):
        alignment.disassociate(chain)

    if structures is None:
        from chimerax.atomic import AtomicStructure
        structures = session.models.list(type = AtomicStructure)

    alignment_sequences = {seq.name:seq for seq in alignment.seqs}
    for structure in structures:
        name = structure.name
        if name not in alignment_sequences:
            print (f'structure {name} not in alignment')
            continue
        if '_' not in name:
            print (f'structure {name} does not have "_"')
            continue
        chain_id = name.split('_')[1]
        chain = _structure_chain(structure, chain_id)
        if chain is None:
            print (f'no chain {chain_id} in {structure}')
            continue
        alignment.associate(chain, alignment_sequences[name])

def _structure_chain(structure, chain_id):
    for chain in structure.chains:
        if chain.chain_id == chain_id:
            return chain
    return None
    
def hide_extra_chains(session, structures):
    for structure in structures:
        if '_' not in structure.name:
            continue
        chain_id = structure.name.split('_')[1]
        res = structure.residues
        other_chains = (res.chain_ids != chain_id)
        res[other_chains].ribbon_displays = False

def register_diffplot_command(logger):
    from chimerax.core.commands import register, CmdDesc, OpenFileNameArg, SaveFileNameArg, IntArg, BoolArg, EnumOf
    from chimerax.atomic import ResiduesArg, AtomicStructuresArg
    from chimerax.seqalign import AlignmentArg
    desc = CmdDesc(required = [],
                   optional = [('embedding_path', OpenFileNameArg)],
                   keyword = [('alignment', AlignmentArg),
                              ('residues', ResiduesArg),
                              ('structures', AtomicStructuresArg),
                              ('drop', EnumOf(('structures', 'residues'))),
                              ('cluster', IntArg),
                              ('coords_path', SaveFileNameArg),
                              ('output_embedding_path', SaveFileNameArg),
                              ('replace', BoolArg),
                              ('verbose', BoolArg)],
                   synopsis='Measure structure differences')
    register('diffplot', desc, diffplot, logger=logger)

    desc = CmdDesc(required = [('structures', AtomicStructuresArg)],
                   synopsis='Hide chains not given in structure name')
    register('diffplot hideextrachains', desc, hide_extra_chains, logger=logger)

#register_diffplot_command(session.logger)
