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

def similar_structures_cluster(session, query_residues = None, align_with = None, cutoff_distance = 2.0,
                               cluster_count = None, cluster_distance = None, replace = True):
    from .simstruct import similar_structure_results
    results = similar_structure_results(session)
    if results is None:
        return

    if not results.have_c_alpha_coordinates():
        from . import coords
        if not coords.similar_structures_fetch_coordinates(session, ask = True):
            return

    if query_residues is None:
        query_residues = results.query_residues

    if len(query_residues) == 0:
        from chimerax.core.errors import UserError
        raise UserError('Must specify at least 1 residue to compute similar structure clusters')
    
    _show_umap(session, results, query_residues,
               align_with = align_with, cutoff_distance = cutoff_distance,
               cluster_count = cluster_count, cluster_distance = cluster_distance,
               replace = replace)

def _show_umap(session, results, query_residues, align_with = None, cutoff_distance = 2.0,
               cluster_count = None, cluster_distance = None, replace = True):

    hits = results.hits
    coord_offsets, hit_names = _aligned_coords(results, query_residues,
                                               align_with = align_with, cutoff_distance = cutoff_distance)
    if len(coord_offsets) == 0:
        session.logger.error(f'Similar structure results contains no structures with all of the specified {len(query_residues)} residues')
        return

    from chimerax.diffplot.diffplot import _umap_embed, _plot_embedding, _install_umap
    _install_umap(session)
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
        cluster_numbers = None
        colors = None

    from chimerax.diffplot.diffplot import _plot_embedding
    p = _plot_embedding(session, hit_names, umap_xy, colors, replace=replace)
    p.query_residues = query_residues
    
    # Set the plot context menu to contain similar structure actions
    from types import MethodType
    p.fill_context_menu = MethodType(fill_context_menu, p)

    msg = f'Clustered {len(hit_names)} of {len(hits)} hits that have the specified {len(query_residues)} residues'
    if cluster_numbers is not None:
        msg += f' into {max(cluster_numbers)} groups'
    session.logger.info(msg)

def _backbone_trace_models(session):
    from .traces import BackboneTraces
    return session.models.list(type = BackboneTraces)

def _aligned_coords(results, query_residues, align_with = None, cutoff_distance = 2.0):
    hits = results.hits
    query_chain_residues = results.query_residues
    from .simstruct import hit_coords, align_xyz_transform
    qatoms = query_chain_residues.find_existing_atoms('CA')
    query_xyz = qatoms.coords
    qria = qatoms.indices(query_residues.find_existing_atoms('CA'))
    qri = set(qria)
    qri.discard(-1)
    qres_xyz = query_xyz[qria]
    if align_with is not None:
        ai = set(qatoms.indices(align_with.find_existing_atoms('CA')))
        ai.discard(-1)
        if len(ai) < 3:
            from chimerax.core.errors import UserError
            raise UserError('Similar structure clustering align_with option specifies fewer than 3 aligned query atoms')

    offsets = []
    names = []
    for hit in hits:
        hit_xyz = hit_coords(hit)
        hi, qi = results.hit_residue_pairing(hit)
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
    self.add_menu_entry(menu, 'Show all traces',
                        lambda self=self: _show_all_traces(self))
    self.add_menu_entry(menu, 'Show one trace per cluster',
                        lambda self=self: _show_one_trace_per_cluster(self))
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
    for tmodel in _backbone_trace_models(session):
        tmodel.show_traces(names, show=show, other=other)

def _show_all_traces(structure_plot):
    cnames = []
    _show_traces(structure_plot.session, cnames, show = True, other = True)

def _show_one_trace_per_cluster(structure_plot):
    cnames = _cluster_center_names(structure_plot)
    _show_traces(structure_plot.session, cnames)
    _show_traces(structure_plot.session, cnames, show = False, other = True)

def _cluster_center_names(structure_plot):
    cnodes = _nodes_by_color(structure_plot.nodes).values()
    center_names = []
    from numpy import array, float32, argmin
    for nodes in cnodes:
        umap_xy = array([n.position[:2] for n in nodes], float32)
        center = umap_xy.mean(axis = 0)
        umap_xy -= center
        d2 = (umap_xy * umap_xy).sum(axis = 1)
        i = argmin(d2)
        center_names.append(nodes[i].name)
    return center_names

def _nodes_by_color(nodes):
    c2n = {}
    for n in nodes:
        color = n.color
        if color in c2n:
            c2n[color].append(n)
        else:
            c2n[color] = [n]
    return c2n

def _show_unplotted_traces(structure_plot):
    _show_traces(structure_plot.session, [n.name for n in structure_plot.nodes],
                 other = True)

def _hide_unplotted_traces(structure_plot):
    _show_traces(structure_plot.session, [n.name for n in structure_plot.nodes],
                 show = False, other = True)

def _color_traces(structure_plot):
    tmodels = _backbone_trace_models(structure_plot.session)
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
    _show_color_panel(structure_plot, node)

_color_dialog = None
def _show_color_panel(structure_plot, node):
    global _color_dialog
    cd = _color_dialog
    from Qt import qt_object_is_deleted
    if cd is None or qt_object_is_deleted(cd):
        parent = structure_plot.tool_window.ui_area
        from Qt.QtWidgets import QColorDialog
        _color_dialog = cd = QColorDialog(parent)
        cd.setOption(cd.NoButtons, True)
    else:
        # On Mac, Qt doesn't realize when the color dialog has been hidden by the red 'X' button, so
        # "hide" it now so that Qt doesn't believe that the later show() is a no op.  Whereas on Windows
        # doing a hide followed by a show causes the dialog to jump back to it's original screen
        # position, so do the hide _only_ on Mac.
        import sys
        if sys.platform == "darwin":
            cd.hide()
        cd.currentColorChanged.disconnect()
    from Qt.QtGui import QColor
    cur_color = QColor.fromRgbF(*tuple(node.color))
    cd.setCurrentColor(cur_color)
    def use_color(color, *, structure_plot=structure_plot, node=node):
        rgba = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
        _color_cluster(structure_plot, node, rgba)
    cd.currentColorChanged.connect(use_color)
    cd.show()

def _color_cluster(structure_plot, node, color):
    cur_color = node.color
    for n in structure_plot.nodes:
        if n.color == cur_color:
            n.color = color
    structure_plot.draw_graph()  # Redraw nodes.

def _cluster_by_distance(umap_xy, cluster_distance):
    if len(umap_xy) <= 1:
        from numpy import ones, int32
        return ones((len(umap_xy),), int32)
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(umap_xy)
    cluster_numbers = fcluster(Z, cluster_distance, criterion = 'distance')
    return cluster_numbers

def _show_reference_atoms(structure_plot):
    qres = structure_plot.query_residues
    qatoms = qres.find_existing_atoms('CA')
    struct = qres[0].structure
    struct.display = True
    struct.atoms.displays = False
    qatoms.displays = True
    struct.residues.ribbon_displays = False

def _select_reference_atoms(structure_plot):
    qres = structure_plot.query_residues
    qatoms = qres.find_existing_atoms('CA')
    struct = qres[0].structure
    struct.session.selection.clear()
    qatoms.selected = True

def register_similar_structures_cluster_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg, IntArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        optional = [('query_residues', ResiduesArg)],
        keyword = [('align_with', ResiduesArg),
                   ('cutoff_distance', FloatArg),
                   ('cluster_count', IntArg),
                   ('cluster_distance', FloatArg),
                   ('replace', BoolArg)],
        synopsis = 'Show umap plot of similar structure hit coordinates for specified residues.'
    )
    register('similarstructures cluster', desc, similar_structures_cluster, logger=logger)
