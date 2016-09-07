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

def show_contact_graph(groups, edge_weights, spring_constant, node_click_callback, session):

    # Create graph
    max_w = float(max(w for g1,g2,w in edge_weights))
    import networkx as nx
    G = nx.Graph()
    for g1, g2, w in edge_weights:
        G.add_edge(g1, g2, weight = w/max_w)

    # Layout nodes
    kw = {} if spring_constant is None else {'k':spring_constant}
    pos = nx.spring_layout(G, **kw) # positions for all nodes

    # Create matplotlib panel
    bundle_info = session.toolshed.find_bundle('contacts')
    p = Plot(session, bundle_info)
    a = p.axes

    # Draw nodes
    # Sizes are areas define by matplotlib.pyplot.scatter() s parameter documented as point^2.
    node_sizes = tuple(0.05 * n.area for n in G)
    node_colors = tuple(n.color for n in G)
    na = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=a)
    na.set_picker(True)	# Generate mouse pick events for clicks on nodes

    # Draw edges
    edges = []
    widths = []
    styles = []
    for (u,v,d) in G.edges(data=True):
        edges.append((u,v))
        large_area = d['weight'] >0.1
        widths.append(3 if large_area else 2)
        styles.append('solid' if large_area else 'dotted')
    ea = nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, style=styles, ax=a)
    ea.set_picker(True)

    # Draw node labels
    short_names = {n:n.short_name for n in G}
    nx.draw_networkx_labels(G, pos, labels=short_names, font_size=16, font_family='sans-serif', ax=a)

    # Hide axes and reduce border padding
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.axis('tight')
    p.figure.tight_layout(pad = 0, w_pad = 0, h_pad = 0)
    p.show()

    if node_click_callback:
        def pick(event, nodes = G.nodes(), cb=node_click_callback,
                 node_artist = na, edge_artist = ea):
            # Check for node click
            c,d = node_artist.contains(event)
            if c:
                n = [nodes[d['ind'][0]]]
            else:
                # Check for edge click
                ec,ed = ea.contains(event)
                if ec:
                    i = ed['ind'][0]
                    n = edges[i]	# Two nodes connected by this edge
                else:
                    # Background clicked
                    n = []
            cb(n, event)
        p.canvas.mpl_connect('button_press_event', pick)
