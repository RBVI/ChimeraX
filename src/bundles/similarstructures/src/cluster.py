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

def similar_structures_cluster(session, query_residues = None, align_with = None, alignment_cutoff_distance = None,
                               cluster_count = None, cluster_distance = None, color_by_species = False,
                               replace = True, from_set = None, of_structures = None):
    from .simstruct import similar_structure_results
    results = similar_structure_results(session, from_set)
    hits = results.named_hits(of_structures)

    if not results.have_c_alpha_coordinates():
        from . import coords
        if not coords.similar_structures_fetch_coordinates(session, ask = True, from_set = from_set,
                                                           of_structures = of_structures):
            return

    if query_residues is None:
        query_residues = results.query_residues

    if len(query_residues) == 0:
        from chimerax.core.errors import UserError
        raise UserError('Must specify at least 1 residue to compute similar structure clusters')

    if alignment_cutoff_distance is None:
        alignment_cutoff_distance = results.alignment_cutoff_distance
    
    _show_umap(session, results, hits, query_residues,
               align_with = align_with, alignment_cutoff_distance = alignment_cutoff_distance,
               cluster_count = cluster_count, cluster_distance = cluster_distance,
               color_by_species = color_by_species, replace = replace)

def _show_umap(session, results, hits, query_residues, align_with = None, alignment_cutoff_distance = 2.0,
               cluster_count = None, cluster_distance = None, color_by_species = False, replace = True):

    coord_offsets, hit_names = _aligned_coords(results, hits, query_residues,
                                               align_with = align_with,
                                               cutoff_distance = alignment_cutoff_distance)
    if len(coord_offsets) == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'Similar structure results contains no structures with all of the specified {len(query_residues)} residues')
    if coord_offsets.shape[1] == 0:
        from chimerax.core.errors import UserError
        raise UserError(f'No query structure residues were specified.')

    from chimerax.umap import install_umap, umap_embed
    install_umap(session)
    umap_xy = umap_embed(coord_offsets)

    from chimerax.umap import k_means_clusters, cluster_by_distance, color_by_cluster
    if cluster_count:
        cluster_numbers = k_means_clusters(umap_xy, cluster_count)
        colors = color_by_cluster(cluster_numbers)
    elif cluster_distance is not None:
        cluster_numbers = cluster_by_distance(umap_xy, cluster_distance)
        colors = color_by_cluster(cluster_numbers)
    else:
        cluster_numbers = None
        colors = None

    plot = _find_similar_structure_plot(session, results.name) if replace else None
    if plot is None:
        plot = SimilarStructurePlot(session)

    plot.set_nodes(results.name, hit_names, umap_xy, colors, query_residues)

    if color_by_species:
        plot._color_by_species()

    msg = f'Clustered {len(hit_names)} of {len(hits)} hits that have the specified {len(query_residues)} residues'
    if cluster_numbers is not None:
        msg += f' into {max(cluster_numbers)} groups'
    session.logger.info(msg)

from chimerax.umap import UmapPlot
class SimilarStructurePlot(UmapPlot):
    def __init__(self, session):
        self._similar_structures_id = None
        self._query_residues = None
        self._cluster_colors = None
        self._species_to_color = None
        UmapPlot.__init__(self, session, title = 'Similar Structures Plot', tool_name = 'SimilarStructures')

    def set_nodes(self, similar_structures_id, structure_names, umap_xy, colors, query_residues):
        self._similar_structures_id = similar_structures_id
        self._query_residues = query_residues
        self._cluster_colors = dict(zip(structure_names, colors)) if colors is not None else None
        self._species_to_color = None
        UmapPlot.set_nodes(self, structure_names, umap_xy, colors)

    @property
    def results(self):
        from .simstruct import similar_structure_results
        return similar_structure_results(self.session, self._similar_structures_id)

    def fill_context_menu(self, menu, item):
        clustered_item = (item is not None and self._have_colors)
        if clustered_item:
            self.add_menu_entry(menu, f'Show traces for cluster {item.name}',
                                lambda self=self, item=item: self._show_cluster_traces(item))
            self.add_menu_entry(menu, f'Show only traces for cluster {item.name}',
                                lambda self=self, item=item: self._show_only_cluster_traces(item))
            self.add_menu_entry(menu, f'Hide traces for cluster {item.name}',
                                lambda self=self, item=item: self._hide_cluster_traces(item))
        self.add_menu_entry(menu, 'Show all traces', self._show_all_traces)
        self.add_menu_entry(menu, 'Show one trace per cluster', self._show_one_trace_per_cluster)
        self.add_menu_entry(menu, 'Show traces not on plot', self._show_unplotted_traces)
        self.add_menu_entry(menu, 'Hide traces not on plot', self._hide_unplotted_traces)

        self.add_menu_separator(menu)
        self.add_menu_entry(menu, 'Color traces to match plot', self._color_traces)
        if clustered_item:
            self.add_menu_entry(menu, f'Change cluster {item.name} color',
                                lambda self=self, item=item: self._change_cluster_color(item))
        self.add_menu_entry(menu, 'Color by cluster', self._color_by_cluster)
        self.add_menu_entry(menu, 'Color by species', self._color_by_species)

        if item is not None:
            self.add_menu_separator(menu)
            self.add_menu_entry(menu, f'Show table row for {item.name}',
                                lambda self=self, item=item: self._show_table_row(item))
            self.add_menu_entry(menu, f'Select rows for cluster {item.name}',
                                lambda self=self, item=item: self._select_cluster_table_rows(item))

        self.add_menu_separator(menu)
        self.add_menu_entry(menu, 'Show reference atoms', self._show_reference_atoms)
        self.add_menu_entry(menu, 'Select reference atoms', self._select_reference_atoms)

    def _show_cluster_traces(self, node):
        if self._create_traces():
            self._show_only_cluster_traces(node)
        else:
            self._show_traces(self._cluster_names(node))

    def _show_only_cluster_traces(self, node):
        cnames = self._cluster_names(node)
        self._show_traces(cnames)
        self._show_traces(cnames, show = False, other = True)

    def _hide_cluster_traces(self, node):
        self._show_traces(self._cluster_names(node), show = False)

    def _show_traces(self, names, show = True, other = False):
        self._create_traces()
        for tmodel in _backbone_trace_models(self.session, self._similar_structures_id):
            tmodel.show_traces(names, show=show, other=other)

    def _create_traces(self):
        if len(_backbone_trace_models(self.session, self._similar_structures_id)) == 0:
            from . import traces
            traces.similar_structures_traces(self.session, from_set = self._similar_structures_id)
            self._color_traces()
            return True
        return False

    def _show_all_traces(self):
        cnames = []
        self._show_traces(cnames, show = True, other = True)

    def _show_one_trace_per_cluster(self):
        cnames = self._cluster_center_names()
        self._show_traces(cnames)
        self._show_traces(cnames, show = False, other = True)

    def _cluster_center_names(self):
        cnodes = _nodes_by_color(self.nodes).values()
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

    def _show_unplotted_traces(self):
        self._show_traces([n.name for n in self.nodes], other = True)

    def _hide_unplotted_traces(self):
        self._show_traces([n.name for n in self.nodes], show = False, other = True)

    def _color_traces(self):
        self._create_traces()
        tmodels = _backbone_trace_models(self.session, self._similar_structures_id)
        if tmodels:
            from chimerax.core.colors import rgba_to_rgba8
            n2c = {node.name:rgba_to_rgba8(node.color) for node in self.nodes}
            for tmodel in tmodels:
                for c in tmodel.chains:
                    color = n2c.get(c.chain_id)
                    if color is not None:
                        r = c.existing_residues
                        r.ribbon_colors = color
                        r.atoms.colors = color

    def _cluster_names(self, node):
        return [n.name for n in self.nodes if n.color == node.color]

    def _change_cluster_color(self, node):
        self._show_color_panel(node)

    _color_dialog = None
    def _show_color_panel(self, node):
        cd = self._color_dialog
        from Qt import qt_object_is_deleted
        if cd is None or qt_object_is_deleted(cd):
            parent = self.tool_window.ui_area
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
        def use_color(color, *, self=self, node=node):
            rgba = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
            self._color_cluster(node, rgba)
        cd.currentColorChanged.connect(use_color)
        cd.show()

    def _color_cluster(self, node, color):
        cur_color = node.color
        cluster_colors = self._cluster_colors
        for n in self.nodes:
            if n.color == cur_color:
                n.color = color
                if cluster_colors:
                    from chimerax.core.colors import rgba_to_rgba8
                    cluster_colors[n.name] = rgba_to_rgba8(color)
        self.draw_graph()  # Redraw nodes.

    def _color_by_cluster(self, no_cluster_color = (178,178,178,255)):
        cluster_colors = self._cluster_colors
        if cluster_colors is None:
            return
        from chimerax.core.colors import rgba8_to_rgba
        for node in self.nodes:
            node.color = rgba8_to_rgba(cluster_colors.get(node.name, no_cluster_color))
        self.draw_graph()  # Redraw nodes.

    def _color_by_species(self):
        node_names = set(node.name for node in self.nodes)
        species = {hit['database_full_id']:hit.get('taxname') for hit in self.results.hits
                   if hit['database_full_id'] in node_names}
        species_colors = self._species_colors(species)
        for node in self.nodes:
            if node.name in species:
                node.color = species_colors[species[node.name]]
        self.draw_graph()  # Redraw nodes.

    def _species_colors(self, species):
        species_colors = self._species_to_color
        if species_colors is None:
            unique_species = list(set(species.values()))
            unique_species.sort()
            from chimerax.core.colors import random_colors, rgba8_to_rgba
            scolors8 = random_colors(len(unique_species))
            scolors = [rgba8_to_rgba(c) for c in scolors8]
            species_colors = dict(zip(unique_species, scolors))
            self._species_to_color = species_colors
            species_colors_html = '<br>'.join(f'{s} {_html_color_square(c)}' for s,c in zip(unique_species, scolors8))
            msg = f'Coloring {len(unique_species)} different species:<br>{species_colors_html}'
            self.session.logger.info(msg, is_html = True)
        return species_colors

    def _show_table_row(self, node):
        from .gui import similar_structures_panel
        ssp = similar_structures_panel(self.session)
        if ssp and self.results is ssp.results:
            hit_nums = [i for i,hit in enumerate(ssp.results.hits) if hit['database_full_id'] == node.name]
            if hit_nums:
                ssp.select_table_row(hit_nums[0])
        ssp.display(True)

    def _select_cluster_table_rows(self, node):
        from .gui import similar_structures_panel
        ssp = similar_structures_panel(self.session)
        if ssp and self.results is ssp.results:
            cnames = self._cluster_names(node)
            ssp.select_table_rows_by_names(cnames)
        ssp.display(True)

    def _show_reference_atoms(self):
        qres = self._query_residues
        qatoms = qres.find_existing_atoms('CA')
        struct = qres[0].structure
        struct.display = True
        struct.atoms.displays = False
        qatoms.displays = True
        qatoms.draw_modes = qatoms.SPHERE_STYLE
        struct.session.selection.clear()
        qatoms.selected = True
        struct.residues.ribbon_displays = False

    def _select_reference_atoms(self):
        qres = self._query_residues
        qatoms = qres.find_existing_atoms('CA')
        struct = qres[0].structure
        struct.session.selection.clear()
        qatoms.selected = True
    
    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        umap_data = UmapPlot.take_snapshot(self, session, flags)
        data = {'umap state': umap_data,
                'similar_structures_id': self._similar_structures_id,
                'query_residues': self._query_residues,
                'cluster_colors': self._cluster_colors,
                'species_to_color': self._species_to_color,
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        ssp = cls(session)
        UmapPlot.set_state_from_snapshot(ssp, session, data['umap state'])
        ssp._similar_structures_id = data['similar_structures_id']
        ssp._query_residues = data['query_residues']
        ssp._cluster_colors = data['cluster_colors']
        ssp._species_to_color = data['species_to_color']
        return ssp

def _find_similar_structure_plot(session, similar_structures_id):
    plots = [tool for tool in session.tools.list()
             if isinstance(tool, SimilarStructurePlot) and tool._similar_structures_id == similar_structures_id]
    return plots[-1] if plots else None

def _nodes_by_color(nodes):
    c2n = {}
    for n in nodes:
        color = n.color
        if color in c2n:
            c2n[color].append(n)
        else:
            c2n[color] = [n]
    return c2n

def _html_color_square(rgba8):
    from chimerax.core.colors import hex_color
    color = hex_color(rgba8)
    return f'&nbsp;<div style="width:10px; height:10px; display:inline-block; border:1px solid #000; background-color:{color}"></div>'

def _backbone_trace_models(session, similar_structures_id):
    from .traces import BackboneTraces
    btmodels = [bt for bt in session.models.list(type = BackboneTraces)
                if bt.similar_structures_id == similar_structures_id]
    return btmodels

def _aligned_coords(results, hits, query_residues, align_with = None, cutoff_distance = 2.0):
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
            ahxyz = hxyz[mask,:]
            aqxyz = qxyz[mask,:]
        if len(ahxyz) < 3:
                continue	# Not enough atoms to align.
        p, rms, npairs = align_xyz_transform(ahxyz, aqxyz, cutoff_distance=cutoff_distance)
        hxyz_aligned = p.transform_points(hit_xyz[hri])
        hxyz_offset = (hxyz_aligned - qres_xyz).flat
        offsets.append(hxyz_offset)
        names.append(hit['database_full_id'])

    from numpy import array
    offsets = array(offsets)

    return offsets, names

def register_similar_structures_cluster_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg, IntArg, StringArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        optional = [('query_residues', ResiduesArg)],
        keyword = [('align_with', ResiduesArg),
                   ('alignment_cutoff_distance', FloatArg),
                   ('cluster_count', IntArg),
                   ('cluster_distance', FloatArg),
                   ('color_by_species', BoolArg),
                   ('replace', BoolArg),
                   ('from_set', StringArg),
                   ('of_structures', StringArg),
                   ],
        synopsis = 'Show umap plot of similar structure hit coordinates for specified residues.'
    )
    register('similarstructures cluster', desc, similar_structures_cluster, logger=logger)
