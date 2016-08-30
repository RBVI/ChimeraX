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

def show_contact_graph(node_weights, edge_weights, short_names, colors, spring_constant, session):

    # Create graph
    max_w = float(max(w for nm1,nm2,w in edge_weights))
    import networkx as nx
    G = nx.Graph()
    for name1, name2, w in edge_weights:
        G.add_edge(name1, name2, weight = w/max_w)

    # Layout nodes
    kw = {} if spring_constant is None else {'k':spring_constant}
    pos = nx.spring_layout(G, **kw) # positions for all nodes

    # Create matplotlib panel
    bundle_info = session.toolshed.find_bundle('contacts')
    p = Plot(session, bundle_info)
    a = p.axes

    # Draw nodes
    from math import sqrt
    w = dict(node_weights)
    node_sizes = tuple(10*sqrt(w[n]) for n in G)
    node_colors = tuple(colors[short_names[n]] for n in G)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=a)

    # Draw edges
    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.1]
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=2, style='dotted', ax=a)
    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.1]
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3, ax=a)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=short_names, font_size=16, font_family='sans-serif', ax=a)

    # Hide axes and reduce border padding
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.axis('tight')
    p.figure.tight_layout(pad = 0, w_pad = 0, h_pad = 0)
    p.show()
