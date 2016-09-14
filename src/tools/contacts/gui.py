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
from chimerax.core.tools import ToolInstance
class Plot(ToolInstance):

    def __init__(self, session, bundle_info, *, title=None):
        ToolInstance.__init__(self, session, bundle_info)

        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self)
        if title is not None:
            tw.title = title
        self.tool_window = tw
        parent = tw.ui_area

        from matplotlib import figure
        self.figure = f = figure.Figure(dpi=100, figsize=(2,2))

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
        self.canvas = c = Canvas(f)
        c.setParent(parent)

        from PyQt5.QtWidgets import QHBoxLayout
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(c)
        parent.setLayout(layout)
        tw.manage(placement="side")

        self.axes = axes = f.gca()

        self._pan = None	# Pan/zoom mouse control
        self._menu = None	# Context menu
        
    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def tight_layout(self):
        '''Hide axes and reduce border padding.'''
        a = self.axes
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.axis('tight')
        self.figure.tight_layout(pad = 0, w_pad = 0, h_pad = 0)

    def equal_aspect(self):
        '''
        Make both axes use same scaling, pixels per plot unit.
        Without this if the window is not square, the plot squishes one axis.
        '''
        self.axes.set_aspect('equal', adjustable='datalim')

    def _mouse_press_pan(self, event):
        from PyQt5.QtCore import Qt
        if event.button() == Qt.LeftButton:
            # Initiate pan and zoom for left click on background
            h = self.tool_window.ui_area.height()
            x, y = event.x(), h-event.y()
            self.axes.start_pan(x, y, button = 1)
            self._pan = False

    def _mouse_move_pan(self, event):
        if self._pan is not None:
            from PyQt5.QtCore import Qt
            if event.modifiers() & Qt.ShiftModifier:
                # Zoom preserving aspect ratio
                button = 3
                key = 'control'
            else:
                # Pan in x and y
                button = 1
                key = None
            h = self.tool_window.ui_area.height()
            x, y = event.x(), h-event.y()
            self.axes.drag_pan(button, key, x, y)
            self._pan = True
            self.canvas.draw()
    
    def _mouse_release_pan(self, event):
        if self._pan is not None:
            self.axes.end_pan()
            did_pan = self._pan
            self._pan = None
            if did_pan:
                return True
        return False

    def matplotlib_mouse_event(self, x, y):
        '''Used for detecting clicked matplotlib canvas item using Artist.contains().'''
        h = self.tool_window.ui_area.height()
        from matplotlib.backend_bases import MouseEvent
        e = MouseEvent('context menu', self.canvas, x, h-y)
        return e

    def add_menu_item(self, text, callback, arg = None):
        '''Add menu item to context menu'''
        widget = self.tool_window.ui_area
        from PyQt5.QtWidgets import QAction
        a = QAction(text, widget)
        #a.setStatusTip("Info about this menu entry")
        args = () if arg is None else (arg,)
        a.triggered.connect(lambda checked, cb=callback, args=args: cb(*args))
        self._context_menu().addAction(a)

    def add_menu_separator(self):
        self._context_menu().addSeparator()

    def post_menu(self, event):
        self._context_menu().exec(event.globalPos())
        self._menu = None
        
    def _context_menu(self):
        m = self._menu
        if m is None:
            widget = self.tool_window.ui_area
            from PyQt5.QtWidgets import QMenu
            self._menu = m = QMenu(widget)
        return m

# ------------------------------------------------------------------------------
#
class ContactPlot(Plot):
    
    def __init__(self, session, groups, contacts):

        # Create matplotlib panel
        bundle_info = session.toolshed.find_bundle('contacts')
        title = '%d Chains %d Contacts' % (len(groups), len(contacts))
        Plot.__init__(self, session, bundle_info, title = title)

        self.groups = groups
        self.contacts = contacts
        
        # Create graph
        self.graph = self._make_graph(contacts)

        # Layout and plot graph
        self._node_artist = None	# Matplotlib PathCollection for node display
        self._edge_artist = None	# Matplotlib LineCollection for edge display
        self._labels = {}		# Maps group to Matplotlib Text object for labels
        self.undisplayed_color = (.8,.8,.8,1)	# Node color for undisplayed chains
        self._draw_graph()

        c = self.canvas
        c.mousePressEvent = self._mouse_press
        c.mouseMoveEvent = self._mouse_move
        c.mouseReleaseEvent = self._mouse_release

        self._handler = session.triggers.add_handler('atomic changes', self._atom_display_change)

        self.tool_window.ui_area.contextMenuEvent = self._show_context_menu
        
    def delete(self):
        self._session().triggers.remove_handler(self._handler)
        self._handler = None
        Plot.delete(self)

    def _make_graph(self, contacts):
        max_area = float(max(c.buried_area for c in contacts))
        import networkx as nx
        # Keep graph nodes in order so we can reproduce the same layout.
        from collections import OrderedDict
        class OrderedGraph(nx.Graph):
            node_dict_factory = OrderedDict
            adjlist_dict_factory = OrderedDict
        G = nx.OrderedGraph()
#        G = nx.Graph()
        for c in contacts:
            G.add_edge(c.group1, c.group2, weight = c.buried_area/max_area, contact=c)
        return G

    def _draw_graph(self):
        # Draw nodes
        node_pos = self._draw_nodes()
    
        # Draw edges
        self._draw_edges(node_pos)

        # Draw node labels
        self._draw_labels(node_pos)

        self.tight_layout()
        self.equal_aspect()	# Don't squish plot if window is not square.
        self.canvas.draw()

        self.show()	# Show graph panel

    def _draw_nodes(self):
        G = self.graph
        node_pos = self._node_layout_positions()
        # Sizes are areas define by matplotlib.pyplot.scatter() s parameter documented as point^2.
        node_sizes = tuple(0.03 * n.area for n in G)
        node_colors = tuple((n.color if n.shown() else self.undisplayed_color) for n in G)
        import networkx as nx
        na = nx.draw_networkx_nodes(G, node_pos, node_size=node_sizes, node_color=node_colors, ax=self.axes)
        na.set_picker(True)	# Generate mouse pick events for clicks on nodes
        if self._node_artist:
            self._node_artist.remove()
        self._node_artist = na	# matplotlib PathCollection object
        return node_pos

    def _node_layout_positions(self):
        # Project camera view positions of chains to x,y.
        proj = self._session().main_view.camera.position.inverse()
        ipos = {g : tuple((proj * g.centroid())[:2]) for g in self.groups}

        # Compute optimal distance between nodes
        from chimerax.core.geometry import distance
        d = sum(distance(ipos[c.group1], ipos[c.group2]) for c in self.contacts) / len(self.contacts)
            
        import networkx as nx
        pos = nx.spring_layout(self.graph, pos = ipos, k = d) # positions for all nodes
        from numpy import array
        self._layout_positions = array([pos[n] for n in self.groups])

        return pos

    def _draw_edges(self, node_pos):
        
        self._edge_contacts = ec = []
        edges = []
        widths = []
        styles = []
        G = self.graph
        for (u,v,d) in G.edges(data=True):
            ec.append(d['contact'])
            edges.append((u,v))
            large_area = d['weight'] > 0.5
            widths.append(3 if large_area else 2)
            styles.append('solid' if large_area else 'dotted')
        import networkx as nx
        ea = nx.draw_networkx_edges(G, node_pos, edgelist=edges, width=widths, style=styles, ax=self.axes)
        ea.set_picker(True)
        if self._edge_artist:
            self._edge_artist.remove()
        self._edge_artist = ea

    def _draw_labels(self, node_pos):
        short_names = {n:n.short_name for n in self.graph}
        import networkx as nx
        labels = nx.draw_networkx_labels(self.graph, node_pos, labels=short_names,
                                         font_size=12, font_family='sans-serif', ax=self.axes)
        if self._labels:
            # Remove existing labels.
            for t in self._labels.values():
                t.remove()
        self._labels = labels	# Dictionary mapping node to matplotlib Text objects.
            
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
        item = self._clicked_item(event.x(), event.y())
        nodes = self._item_nodes(item)
        shift_key = event.modifiers() & Qt.ShiftModifier

        if shift_key:
            self._select_nodes(nodes)
        else:
            n = len(nodes)
            if n == 0:
                self._show_all_atoms()
            elif n == 1:
                self._show_neighbors(nodes[0])
            else:
                # Edge clicked, pair of nodes
                self._show_node_atoms(nodes)

    def _clicked_item(self, x, y):
        # Check for node click
        e = self.matplotlib_mouse_event(x,y)
        c,d = self._node_artist.contains(e)
        if c:
            i = d['ind'][0]
            item = self.graph.nodes()[i]
        else:
            # Check for edge click
            ec,ed = self._edge_artist.contains(e)
            if ec:
                i = ed['ind'][0]
                c = self._edge_contacts[i]
                item = c
            else:
                # Background clicked
                item = None
        return item

    def _item_nodes(self, item):
        from .cmd import SphereGroup, Contact
        if item is None:
            nodes = []
        elif isinstance(item, SphereGroup):
            nodes = [item]
        elif isinstance(item, Contact):
            nodes = [item.group1, item.group2]
        return nodes
                    
    def _select_nodes(self, nodes):
        self._clear_selection()
        for g in nodes:
            g.atoms.selected = True

    def _select_neighbors(self, g):
        self._clear_selection()
        from .cmd import neighbors
        for n in neighbors(g, self.contacts):
            n.atoms.selected = True

    def _select_contact_residues(self, contacts, min_area = 1):
        self._clear_selection()        
        for c in contacts:
            for g in (c.group1, c.group2):
                atoms = c.contact_residue_atoms(g, min_area)
                atoms.selected = True
            
    def _clear_selection(self):
        self._session().selection.clear()

    def _node_contacts(self, n):
        from .cmd import neighbors
        nc = neighbors(n, self.contacts)	# Map neighbor node to Contact
        return tuple(nc.values())
        
    def _show_node_atoms(self, nodes):
        gset = set(nodes)
        for h in self.groups:
            h.atoms.displays = (h in gset)

    def _show_neighbors(self, g):
        from .cmd import neighbors
        ng = neighbors(g, self.contacts)
        ng[g] = None
        for h in self.groups:
            h.atoms.displays = (h in ng)

    def _show_contact_residues(self, g, min_area = 5, color = (255,255,255,255)):
        from .cmd import neighbors
        ng = neighbors(g, self.contacts)	# Map neighbor node to Contact
        from chimerax.core.atomic import Atom
        for h in self.groups:
            if h in ng:
                c = ng[h]
                atoms = c.contact_residue_atoms(h, min_area)
                h.atoms.displays = False
                atoms.displays = True	# Show only contacting residues
                atoms.draw_modes = Atom.STICK_STYLE
                gatoms = c.contact_residue_atoms(g, min_area)
                gatoms.draw_modes = Atom.STICK_STYLE
                gatoms.colors = color
            else:
                h.atoms.displays = (h is g)

    def _show_all_atoms(self):
        for g in self.groups:
            g.atoms.displays = True

    def _show_residue_plot(self, c):
        from .resplot import ResiduePlot
        ResiduePlot(self._session(), c)
        
    def _explode_all(self, scale = 2):
        gc = [(g,g.centroid()) for g in self.groups if g.shown()]
        if len(gc) < 2:
            return
        from numpy import mean
        center = mean([c for g,c in gc], axis = 0)
        for g,c in gc:
            g.move((scale-1)*(c - center))

    def _unexplode_all(self):
        for g in self.groups:
            g.unmove()

    def _explode_neighbors(self, n):
        for c in self._node_contacts(n):
            g = c.group2 if n is c.group1 else c.group1
            c.explode_contact(move_group = g)

    def _orient(self):
        gc = []
        la = []
        for g,(x,y) in zip(self.groups, self._layout_positions):
            if g.shown():
                gc.append(g.centroid())
                la.append((x,y,0))
        if len(gc) < 2:
            return
        from chimerax.core.geometry import align_points, translation
        from numpy import array, mean
        p, rms = align_points(array(gc), array(la))
        ra = p.zero_translation()
        center = mean(gc, axis=0)
        v = self._session().main_view
        rc = v.camera.position.zero_translation()
        rot = translation(center) * rc * ra * translation(-center)
        v.move(rot)
        
    def _atom_display_change(self, name, changes):
        if 'display changed' in changes.atom_reasons():
            # Atoms shown or hidden.  Color hidden nodes gray.
            node_colors = tuple((n.color if n.shown() else self.undisplayed_color) for n in self.graph)
            self._node_artist.set_facecolor(node_colors)
            self.canvas.draw()	# Need to ask canvas to redraw the new colors.

    def _show_context_menu(self, event):
        item = self._clicked_item(event.x(), event.y())
        nodes = self._item_nodes(item)
        node_names = ','.join(n.name for n in nodes)
        nn = len(nodes)
        
        # Show/hide menu entries
        add = self.add_menu_item
        if nodes:
            add('Show only %s' % node_names, self._show_node_atoms, nodes)

        if len(nodes) == 1:
            add('Show %s and neighbors' % node_names, self._show_neighbors, nodes[0])
            add('Show contact residues', self._show_contact_residues, nodes[0])
        
        add('Show all atoms', self._show_all_atoms)

        from .cmd import Contact, SphereGroup
        if isinstance(item, Contact):
            add('Residue plot', self._show_residue_plot, item)

        self.add_menu_separator()

        earg = None
        if isinstance(item, Contact):
            explode = item.explode_contact
        elif isinstance(item, SphereGroup):
            explode = self._explode_neighbors
            earg = item
        else:
            explode = self._explode_all
        add('Explode', explode, earg)
        add('Unxplode', self._unexplode_all)

        self.add_menu_separator()

        add('Layout matching structure', self._draw_graph)
        add('Orient structure', self._orient)
        
        self.add_menu_separator()
        
        # Selection menu entries
        if nodes:
            add('Select %s' % node_names, self._select_nodes, nodes)

        if nn == 1:
            add('Select neighbors', self._select_neighbors, nodes[0])

        if nn == 0:
            clist = self.contacts
        elif nn == 1:
            clist = self._node_contacts(nodes[0])
        elif nn == 2:
            clist = [item]
        add('Select contact residues', self._select_contact_residues, clist)

        add('Select all', self._select_nodes, self.groups)
        add('Clear selection', self._clear_selection)

        self.post_menu(event)
