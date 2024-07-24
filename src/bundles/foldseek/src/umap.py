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

def foldseek_umap(session, query_residues = None, align_with = None, cutoff_distance = 2.0,
                  cluster_count = None, cluster_distance = None, replace = True):
    from .gui import foldseek_panel
    fp = foldseek_panel(session)
    if fp is None:
        return

    query_chain = fp.results_query_chain
    if query_residues is None:
        from .foldseek import alignment_residues
        query_residues = alignment_residues(query_chain.existing_residues)

    _show_umap(session, fp.hits, query_chain, query_residues,
               align_with = align_with, cutoff_distance = cutoff_distance,
               cluster_count = cluster_count, cluster_distance = cluster_distance,
               replace = replace)

def _show_umap(session, hits, query_chain, query_residues, align_with = None, cutoff_distance = 2.0,
               cluster_count = None, cluster_distance = None, replace = True):

    coord_offsets, hit_names = _aligned_coords(hits, query_chain, query_residues,
                                               align_with = align_with, cutoff_distance = cutoff_distance)

    from chimerax.diffplot.diffplot import _umap_embed, _plot_embedding
    umap_xy = _umap_embed(coord_offsets)

    if cluster_count:
        from chimerax.diffplot.diffplot import _k_means_clusters, _color_by_cluster
        cluster_numbers = _k_means_clusters(umap_xy, cluster_count)
        colors = _color_by_cluster(cluster_numbers)
    elif cluster_distance is not None:
        from chimerax.diffplot.diffplot import _color_by_cluster
        cluster_numbers = _cluster_by_distance(umap_xy, cluster_distance)
        colors = _color_by_cluster(cluster_numbers)
    else:
        colors = None

    from chimerax.diffplot.diffplot import _plot_embedding
    p = _plot_embedding(session, hit_names, umap_xy, colors, replace=replace)
    p.query_residues = query_residues
    
    # Set the plot context menu to contain foldseek actions
    from types import MethodType
    p.fill_context_menu = MethodType(fill_context_menu, p)

def _foldseek_trace_models(session):
    from .traces import FoldseekTraces
    return session.models.list(type = FoldseekTraces)

def _aligned_coords(hits, query_chain, query_residues, align_with = None, cutoff_distance = 2.0):
    from .foldseek import alignment_residues, hit_coords, hit_residue_pairing, align_xyz_transform
    qres = alignment_residues(query_chain.existing_residues)
    qatoms = qres.existing_principal_atoms
    query_xyz = qatoms.coords
    qria = qatoms.indices(query_residues.existing_principal_atoms)
    qri = set(qria)
    qri.discard(-1)
    qres_xyz = query_xyz[qria]
    if align_with is not None:
        ai = set(qatoms.indices(align_with.existing_principal_atoms))
        ai.discard(-1)
        if len(ai) < 3:
            from chimerax.core.errors import UserError
            raise UserError('Foldseek umap align_with specifies fewer than 3 aligned query atoms')

    offsets = []
    names = []
    for hit in hits:
        hit_xyz = hit_coords(hit)
        hi, qi = hit_residue_pairing(hit)
        hri = [j for i,j in zip(qi,hi) if i in qri]
        if len(hri) < len(qri):
            continue
        hxyz = hit_xyz[hi]
        qxyz = query_xyz[qi]
        if align_with is None:
            ahxyz, aqxyz = hxyz, qxyz
        else:
            from numpy import array
            mask = array([(i in ai) for i in qi], bool)
            if mask.sum() < 3:
                continue	# Not enough atoms to align.
            ahxyz = hxyz[mask,:]
            aqxyz = qxyz[mask,:]
        p, rms, npairs = align_xyz_transform(ahxyz, aqxyz, cutoff_distance=cutoff_distance)
        hxyz_aligned = p.transform_points(hit_xyz[hri])
        hxyz_offset = (hxyz_aligned - qres_xyz).flat
        offsets.append(hxyz_offset)
        names.append(hit['database_full_id'])

    from numpy import array
    offsets = array(offsets)

    return offsets, names

def fill_context_menu(self, menu, item):
    if item is not None and self._have_colors:
        self.add_menu_entry(menu, f'Show traces for cluster {item.name}',
                            lambda self=self, item=item: _show_cluster_traces(self, item))
        self.add_menu_entry(menu, f'Show only traces for cluster {item.name}',
                            lambda self=self, item=item: _show_only_cluster_traces(self, item))
        self.add_menu_entry(menu, f'Hide traces for cluster {item.name}',
                            lambda self=self, item=item: _hide_cluster_traces(self, item))
        self.add_menu_entry(menu, 'Change cluster color',
                            lambda self=self, item=item: _change_cluster_color(self, item))
    self.add_menu_entry(menu, 'Color traces to match plot',
                        lambda self=self: _color_traces(self))
    self.add_menu_entry(menu, 'Show traces not on plot',
                        lambda self=self: _show_unplotted_traces(self))
    self.add_menu_entry(menu, 'Hide traces not on plot',
                        lambda self=self: _hide_unplotted_traces(self))
    self.add_menu_entry(menu, 'Show reference atoms',
                        lambda self=self: _show_reference_atoms(self))
    self.add_menu_entry(menu, 'Select reference atoms',
                        lambda self=self: _select_reference_atoms(self))
    
def _show_cluster_traces(structure_plot, node):
    _show_traces(structure_plot.session, _cluster_names(structure_plot, node))

def _show_only_cluster_traces(structure_plot, node):
    cnames = _cluster_names(structure_plot, node)
    _show_traces(structure_plot.session, cnames)
    _show_traces(structure_plot.session, cnames, show = False, other = True)

def _hide_cluster_traces(structure_plot, node):
    _show_traces(structure_plot.session, _cluster_names(structure_plot, node), show = False)

def _show_traces(session, names, show = True, other = False):
    tmodels = _foldseek_trace_models(session)
    names_set = set(names)
    for tmodel in tmodels:
        tmask = tmodel.triangle_mask
        if tmask is None:
            from numpy import ones
            tmask = ones((len(tmodel.triangles),), bool)
        for name, tstart, tend in tmodel.trace_triangle_ranges():
            change = (name not in names_set) if other else (name in names_set)
            if change:
                tmask[tstart:tend] = show
        tmodel.triangle_mask = tmask

def _show_unplotted_traces(structure_plot):
    _show_traces(structure_plot.session, [n.name for n in structure_plot.nodes],
                 other = True)

def _hide_unplotted_traces(structure_plot):
    _show_traces(structure_plot.session, [n.name for n in structure_plot.nodes],
                 show = False, other = True)

def _color_traces(structure_plot):
    tmodels = _foldseek_trace_models(structure_plot.session)
    if tmodels:
        from chimerax.core.colors import rgba_to_rgba8
        n2c = {node.name:rgba_to_rgba8(node.color) for node in structure_plot.nodes}
        for tmodel in tmodels:
            vc = tmodel.get_vertex_colors(create = True)
            for tname, vstart, vend in tmodel.trace_vertex_ranges():
                if tname in n2c:
                    vc[vstart:vend] = n2c[tname]
            tmodel.vertex_colors = vc

def _cluster_names(structure_plot, node):
    return [n.name for n in structure_plot.nodes if n.color == node.color]

def _change_cluster_color(structure_plot, node):
    cur_color = node.color
    from chimerax.core.colors import distinguish_from
    opacity = 1
    color = distinguish_from([cur_color]) + (opacity,)
    for n in structure_plot.nodes:
        if n.color == cur_color:
            n.color = color
    structure_plot.draw_graph()  # Redraw nodes.

def _cluster_by_distance(umap_xy, cluster_distance):
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(umap_xy)
    cluster_numbers = fcluster(Z, cluster_distance, criterion = 'distance')
    return cluster_numbers

def _show_reference_atoms(structure_plot):
    qres = structure_plot.query_residues
    qatoms = qres.existing_principal_atoms
    struct = qres[0].structure
    struct.display = True
    struct.atoms.displays = False
    qatoms.displays = True
    struct.residues.ribbon_displays = False

def _select_reference_atoms(structure_plot):
    qres = structure_plot.query_residues
    qatoms = qres.existing_principal_atoms
    struct = qres[0].structure
    struct.session.selection.clear()
    qatoms.selected = True

def register_foldseek_umap_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg, IntArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        optional = [('query_residues', ResiduesArg)],
        keyword = [('align_with', ResiduesArg),
                   ('cutoff_distance', FloatArg),
                   ('cluster_count', IntArg),
                   ('cluster_distance', FloatArg),
                   ('replace', BoolArg)],
        synopsis = 'Show umap plot of foldseek hit coordinates for specified residues.'
    )
    register('foldseek umap', desc, foldseek_umap, logger=logger)
