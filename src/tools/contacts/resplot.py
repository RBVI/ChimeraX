# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# ------------------------------------------------------------------------------
#
from .gui import Plot
class ResiduePlot(Plot):
    
    def __init__(self, session, contact):

        self.contact = c = contact

        # Interface residues
        g1, g2 = c.group1, c.group2
        self.residues1 = r1 = c.contact_residues(g1)
        self.residues2 = r2 = c.contact_residues(g2)
        from chimerax.core.atomic import concatenate
        self.residues = res = concatenate((r1, r2))

        # Non-interface residues
        self.noninterface_residues1 = g1.atoms.unique_residues.subtract(r1)
        self.noninterface_residues2 = g2.atoms.unique_residues.subtract(r2)
        allres = concatenate((g1.atoms, g2.atoms)).unique_residues
        self.noninterface_residues = allres.subtract(res)
        
        # Create matplotlib panel
        bundle_info = session.toolshed.find_bundle('contacts')
        title = '%s %d residues and %s %d residues' % (g1.name, len(r1), g2.name, len(r2))
        Plot.__init__(self, session, bundle_info, title = title)

        # Create graph
        self.graph = self._make_graph()

        # Layout and plot graph
        self._draw_graph()

        # Don't squish plot if window is not square.
        self.equal_aspect()

        # Setup mousemodes
        c = self.canvas
        c.mousePressEvent = self._mouse_press
        c.mouseMoveEvent = self._mouse_move
        c.mouseReleaseEvent = self._mouse_release
        self._pan = None
        self._interface_shown = False
        
        self.tool_window.ui_area.contextMenuEvent = self._show_context_menu

    def _make_graph(self):
        import networkx as nx
        # Keep graph nodes in order so we can reproduce the same layout.
        from collections import OrderedDict
        class OrderedGraph(nx.Graph):
            node_dict_factory = OrderedDict
            adjlist_dict_factory = OrderedDict
        G = nx.OrderedGraph()

        c = self.contact
        for r in self.residues:
            G.add_node(r)
            
        return G

    def _draw_graph(self):
                
        G = self.graph
        axes = self.axes

        # Layout nodes
        proj = self._layout_projection()
        pos = self._layout_positions(self.residues, proj)

        # Draw nodes
        # Sizes are areas define by matplotlib.pyplot.scatter() s parameter documented as point^2.
#        node_sizes = tuple(0.03 * n.area for n in G)
        node_colors = tuple(tuple(r255/255 for r255 in r.ribbon_color) for r in G)
        import networkx as nx
        na = nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, ax=axes)
        na.set_picker(True)	# Generate mouse pick events for clicks on nodes
        self._node_artist = na

        nires = self.noninterface_residues1
        opos = self._layout_positions(nires, proj)
        ba = nx.draw_networkx_nodes(G, opos, nodelist=nires, node_size=800, node_color=(.7,.7,.7,1), linewidths=0,
                                    ax=axes)
        ba.set_zorder(-10)

        # Draw node labels
        short_names = {r:'%d'%r.number for r in G}
        nx.draw_networkx_labels(G, pos, labels=short_names, font_size=8, font_family='sans-serif', ax=axes)

        self.tight_layout()
        self.show()

    def _layout_projection(self):
        c = self.contact
        r1, r2 = (self.residues1, self.residues2)
        xyz1, xyz2 = [r.atoms.scene_coords.mean(axis = 0) for r in (r1,r2)]
        zaxis = xyz2 - xyz1
        center = 0.5 * (xyz1 + xyz2)
        from chimerax.core.geometry import orthonormal_frame
        f = orthonormal_frame(zaxis, origin = center)
        finv = f.inverse()
        return finv

    def _layout_positions(self, residues, proj):
        pos = {}
        for r in residues:
            rxyz = r.atoms.scene_coords.mean(axis = 0)
            x,y,z = proj * rxyz
            pos[r] = (x,y)
        return pos
        
    def _mouse_press(self, event):
        if self._clicked_item(event.x(), event.y()) is None:
            self._mouse_press_pan(event)

    def _mouse_move(self, event):
        self._mouse_move_pan(event)
    
    def _mouse_release(self, event):
        if self._mouse_release_pan(event):
            return

        from PyQt5.QtCore import Qt
        if event.button() != Qt.LeftButton:
            return	# Only handle left button.  Right button will post menu.

        r = self._clicked_item(event.x(), event.y())
#        shift_key = event.modifiers() & Qt.ShiftModifier
        if r:
            self._select_residue(r)
            if not self._interface_shown:
                self._show_interface()

    def _clicked_item(self, x, y):
        # Check for node click
        e = self.matplotlib_mouse_event(x,y)
        c,d = self._node_artist.contains(e)
        if c:
            i = d['ind'][0]
            r = self.graph.nodes()[i]
        else:
            r = None
        return r

    def _show_context_menu(self, event):
        r = self._clicked_item(event.x(), event.y())
        add = self.add_menu_item
        add('Show interface', self._show_interface)
        if r:
            add('Select residue', self._select_residue, r)
        self.post_menu(event)

    def _residue_name(self, r):
        return '%s %d' % (r.name, r.number)
    
    def _show_interface(self):
        c = self.contact
        aa1 = c.group1.atoms
        aa1.displays = True
        gray = (180,180,180,255)
        self.noninterface_residues1.atoms.colors = gray
        self.noninterface_residues2.atoms.displays = False
        a2 = self.residues2.atoms
        a2.displays = True
        a2.draw_modes = a2.STICK_STYLE
        v = self._session().main_view
        v.camera.position = self._layout_projection().inverse()
        v.view_all(aa1.scene_bounds)
        self._interface_shown = True
    
    def _select_residue(self, r):
        self._clear_selection()
        r.atoms.selected = True
            
    def _clear_selection(self):
        self._session().selection.clear()
