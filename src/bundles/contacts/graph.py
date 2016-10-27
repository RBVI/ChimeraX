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

    def __init__(self, session, tool_name, *, title=None):
        ToolInstance.__init__(self, session, tool_name)

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

# ------------------------------------------------------------------------------
#
class Graph(Plot):
    '''Show a graph of labeled nodes and edges.'''
    
    def __init__(self, session, nodes, edges, tool_name, title):

        # Create matplotlib panel
        Plot.__init__(self, session, tool_name, title = title)
        self.tool_window.fill_context_menu = self._fill_context_menu

        self.nodes = nodes
        self.background_nodes = ()	# No click events
        self.edges = edges

        self.font_size = 12
        self.font_family = 'sans-serif'

        # Create graph
        self.graph = self._make_graph()

        # Layout and plot graph
        self._node_artist = None	# Matplotlib PathCollection for node display
        self._edge_artist = None	# Matplotlib LineCollection for edge display
        self._labels = {}		# Maps group to Matplotlib Text object for labels

        c = self.canvas
        c.mousePressEvent = self._mouse_press
        c.mouseMoveEvent = self._mouse_move
        c.mouseReleaseEvent = self._mouse_release

    def _make_graph(self):
        import networkx as nx
        # Keep graph nodes in order so we can reproduce the same layout.
        from collections import OrderedDict
        class OrderedGraph(nx.Graph):
            node_dict_factory = OrderedDict
            adjlist_dict_factory = OrderedDict
        G = nx.OrderedGraph()
        G.add_nodes_from(self.nodes)
        edges = self.edges
        if edges:
            max_weight = float(max(e.weight for e in edges))
            for e in edges:
                G.add_edge(e.nodes[0], e.nodes[1], weight = e.weight/max_weight, edge_object=e)
        return G

    def draw_graph(self):
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
        nodes = tuple(n for n in G if not n.background)
        node_sizes = tuple(n.size for n in nodes)
        node_colors = tuple(n.color for n in nodes)
        import networkx as nx
        na = nx.draw_networkx_nodes(G, node_pos, nodelist = nodes,
                                    node_size=node_sizes, node_color=node_colors, ax=self.axes)
        na.set_picker(True)	# Generate mouse pick events for clicks on nodes
        if self._node_artist:
            self._node_artist.remove()
        self._node_artist = na	# matplotlib PathCollection object

        if len(nodes) < len(G):
            # Unclickable background nodes
            bnodes = tuple(n for n in G if n.background)
            ba = nx.draw_networkx_nodes(G, node_pos, nodelist = bnodes,
                                        node_size=tuple(n.size for n in bnodes),
                                        node_color=tuple(n.color for n in bnodes),
                                        linewidths=0, ax=self.axes)
            ba.set_zorder(-10)

        return node_pos

    def recolor_nodes(self):
        node_colors = tuple(n.color for n in self.graph)
        self._node_artist.set_facecolor(node_colors)
        self.canvas.draw()	# Need to ask canvas to redraw the new colors.

    def _node_layout_positions(self):
        # Project camera view positions of chains to x,y.
        proj = self.layout_projection()
        ipos = {n : (proj * n.position)[:2] for n in self.nodes}

        ne = len(self.edges)
        if ne > 0:
            # Compute average distance between nodes
            from chimerax.core.geometry import distance
            d = sum(distance(ipos[e.nodes[0]], ipos[e.nodes[1]]) for e in self.edges) / ne
            import networkx as nx
            pos = nx.spring_layout(self.graph, pos = ipos, k = d) # positions for all nodes
        else:
            pos = ipos
        from numpy import array
        self._layout_positions = array([pos[n] for n in self.nodes])

        return pos

    def layout_projection(self):
        return self._session().main_view.camera.position.inverse()

    def _draw_edges(self, node_pos):
        
        self._edge_objects = eo = []
        edges = []
        widths = []
        styles = []
        G = self.graph
        for (u,v,d) in G.edges(data=True):
            e = d['edge_object']
            eo.append(e)
            edges.append((u,v))
            widths.append(e.width)
            styles.append(e.style)
        if len(edges) == 0:
            return
        import networkx as nx
        ea = nx.draw_networkx_edges(G, node_pos, edgelist=edges, width=widths,
                                    style=styles, ax=self.axes)
        ea.set_picker(True)
        if self._edge_artist:
            self._edge_artist.remove()
        self._edge_artist = ea

    def _draw_labels(self, node_pos):
        node_names = {n:n.name for n in self.graph if not n.background}
        import networkx as nx
        labels = nx.draw_networkx_labels(self.graph, node_pos, labels=node_names,
                                         font_size=self.font_size,
                                         font_family=self.font_family, ax=self.axes)

        elabel = [e for e in self.edges if e.label]
        if elabel:
            ed = {tuple(e.nodes):e.label for e in elabel}
            elab = nx.draw_networkx_edge_labels(self.graph, node_pos, edge_labels=ed,
                                                font_size=self.font_size,
                                                font_family=self.font_family,
                                                rotate = False, ax=self.axes)
            labels.update(elab)

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
        self.mouse_click(item, event)

    def mouse_click(self, node_or_edge, event):
        pass

    def is_shift_key_pressed(self, event):
        from PyQt5.QtCore import Qt
        return event.modifiers() & Qt.ShiftModifier

    def _clicked_item(self, x, y):
        # Check for node click
        e = self.matplotlib_mouse_event(x,y)
        c,d = self._node_artist.contains(e)
        item = None
        if c:
            i = d['ind'][0]
            item = self.graph.nodes()[i]
        elif self._edge_artist:
            # Check for edge click
            ec,ed = self._edge_artist.contains(e)
            if ec:
                i = ed['ind'][0]
                c = self._edge_objects[i]
                item = c
        return item

    def item_nodes(self, item):
        if item is None:
            nodes = []
        elif isinstance(item, Node):
            nodes = [item]
        elif isinstance(item, Edge):
            nodes = item.nodes
        return nodes

    def _fill_context_menu(self, menu, x, y):
        item = self._clicked_item(x, y)
        self.fill_context_menu(menu, item)

    def fill_context_menu(self, menu, item):
        pass

    def add_menu_entry(self, menu, text, callback, *args):
        '''Add menu item to context menu'''
        widget = self.tool_window.ui_area
        from PyQt5.QtWidgets import QAction
        a = QAction(text, widget)
        #a.setStatusTip("Info about this menu entry")
        a.triggered.connect(lambda checked, cb=callback, args=args: cb(*args))
        menu.addAction(a)

# ------------------------------------------------------------------------------
#
class Node:
    '''Node for Graph plot.'''
    name = ''		# Text shown on plot node.
    position = (0,0,0)	# 3d position of node, projected to xy for graph layout
    size = 1000		# Node size on plot in pixel area.
    color = (.8,.8,.8)	# RGB color for node, 0-1 range.
    background = False	# Whether to draw this mode behind other, unclickable

# ------------------------------------------------------------------------------
#
class Edge:
    '''Edge for Graph plot.'''
    weight = 1		# Edge weight for spring layout
    nodes = (None,None) # Two Node objects connected by this edge
    style = 'solid'	# Line style: 'solid', 'dotted', 'dashed'...
    width = 3		# Line width in pixels
    label = None	# Text label for edge
