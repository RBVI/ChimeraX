# vim: set expandtab ts=4 sw=4:

# ------------------------------------------------------------------------------
#
from chimera.core.tools import ToolInstance
class Plot(ToolInstance):

    SIZE = (300, 300)

    def __init__(self, session, tool_info, *, restoring=False, title='Plot'):
        if not restoring:
            ToolInstance.__init__(self, session, tool_info)

        from chimera.core.ui import MainToolWindow
        tw = MainToolWindow(self, size=self.SIZE)
        self.tool_window = tw
        parent = tw.ui_area

        from matplotlib import figure
        self.figure = f = figure.Figure(dpi=100, figsize=(2,2))
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
        self.canvas = Canvas(parent, -1, f)

        import wx
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas,1,wx.EXPAND)
        parent.SetSizerAndFit(sizer)

        tw.manage(placement="right")

        self.axes = axes = f.gca()

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        data = {
            "ti": ToolInstance.take_snapshot(self, session, flags),
            "shown": self.tool_window.shown
        }
        return self.tool_info.session_write_version, data

    def restore_snapshot_init(self, session, tool_info, version, data):
        if version not in tool_info.session_versions:
            from chimera.core.state import RestoreError
            raise RestoreError("unexpected version")
        ti_version, ti_data = data["ti"]
        ToolInstance.restore_snapshot_init(
            self, session, tool_info, ti_version, ti_data)
        self.__init__(session, tool_info, restoring=True)
        self.display(data["shown"])

    def reset_state(self, session):
        pass

def show_contact_graph(node_weights, edge_weights, short_names, session):

    # Create graph
    max_w = float(max(w for nm1,nm2,w in edge_weights))
    import networkx as nx
    G = nx.Graph()
    for name1, name2, w in edge_weights:
        G.add_edge(name1, name2, weight = w/max_w)

    # Layout nodes
    pos = nx.spring_layout(G) # positions for all nodes

    # Create matplotlib panel
    tool_info = session.toolshed.find_tool('contacts')
    p = Plot(session, tool_info)
    a = p.axes

    # Draw nodes
    from math import sqrt
    w = dict(node_weights)
    node_sizes = tuple(10*sqrt(w[n]) for n in G)
    from chimera.core.colors import chain_rgba
    node_colors = tuple(chain_rgba(short_names[n]) for n in G)
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
