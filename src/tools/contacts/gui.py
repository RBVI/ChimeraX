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

    def __init__(self, session, bundle_info, *, title='Plot'):
        ToolInstance.__init__(self, session, bundle_info)

        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from matplotlib import figure
        self.figure = f = figure.Figure(dpi=100, figsize=(2,2))

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
        self.canvas = c = Canvas(f)
        c.setParent(parent)

	# To get keys held down during mouse event need to accept focus
        from PyQt5.QtCore import Qt
        c.setFocusPolicy(Qt.ClickFocus)

        from PyQt5.QtWidgets import QHBoxLayout
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(c)
        parent.setLayout(layout)
        tw.manage(placement="right")

        self.axes = axes = f.gca()

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

# ------------------------------------------------------------------------------
#
class ContactPlot(Plot):
    
    def __init__(self, session, groups, edge_weights, spring_constant, graph_click_callback):

        # Create matplotlib panel
        bundle_info = session.toolshed.find_bundle('contacts')
        Plot.__init__(self, session, bundle_info)

        # Create graph
        self.graph = self._make_graph(edge_weights)

        # Layout and plot graph
        self.undisplayed_color = (.8,.8,.8,1)	# Node color for undisplayed chains
        self._draw_graph(spring_constant)

        self._graph_click_callback = graph_click_callback
        if graph_click_callback:
            self.canvas.mpl_connect('button_press_event', self._pick)

        self._handler = session.triggers.add_handler('atomic changes', self._atom_display_change)

    def delete(self):
        self._session().triggers.remove_handler(self._handler)
        self._handler = None
        Plot.delete(self)

    def _make_graph(self, edge_weights):
        max_w = float(max(w for g1,g2,w in edge_weights))
        import networkx as nx
        G = nx.Graph()
        for g1, g2, w in edge_weights:
            G.add_edge(g1, g2, weight = w/max_w)
        return G

    def _draw_graph(self, spring_constant):
                
        G = self.graph
        axes = self.axes
        
        # Layout nodes
        kw = {} if spring_constant is None else {'k':spring_constant}
        import networkx as nx
        pos = nx.spring_layout(G, **kw) # positions for all nodes

        # Draw nodes
        # Sizes are areas define by matplotlib.pyplot.scatter() s parameter documented as point^2.
        node_sizes = tuple(0.05 * n.area for n in G)
        node_colors = tuple((n.color if n.shown() else self.undisplayed_color) for n in G)
        na = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=axes)
        na.set_picker(True)	# Generate mouse pick events for clicks on nodes
        self._node_artist = na
    
        # Draw edges
        self._edges = edges = []
        widths = []
        styles = []
        for (u,v,d) in G.edges(data=True):
            edges.append((u,v))
            large_area = d['weight'] >0.1
        widths.append(3 if large_area else 2)
        styles.append('solid' if large_area else 'dotted')
        ea = nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, style=styles, ax=axes)
        ea.set_picker(True)
        self._edge_artist = ea

        # Draw node labels
        short_names = {n:n.short_name for n in G}
        nx.draw_networkx_labels(G, pos, labels=short_names, font_size=16, font_family='sans-serif', ax=axes)

        # Hide axes and reduce border padding
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.axis('tight')
        self.figure.tight_layout(pad = 0, w_pad = 0, h_pad = 0)
        self.show()

    def _pick(self, event):
        # Check for node click
        c,d = self._node_artist.contains(event)
        if c:
            i = d['ind'][0]
            n = [self.graph.nodes()[i]]
        else:
            # Check for edge click
            ec,ed = self._edge_artist.contains(event)
            if ec:
                i = ed['ind'][0]
                n = self._edges[i]	# Two nodes connected by this edge
            else:
                # Background clicked
                n = []
        self._graph_click_callback(n, event)

    def _atom_display_change(self, name, changes):
        if 'display changed' in changes.atom_reasons():
            # Atoms shown or hidden.  Color hidden nodes gray.
            node_colors = tuple((n.color if n.shown() else self.undisplayed_color) for n in self.graph)
            self._node_artist.set_facecolor(node_colors)
            self.canvas.draw()	# Need to ask canvas to redraw the new colors.

