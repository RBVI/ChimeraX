# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# ------------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class Plot(ToolInstance):

    def __init__(self, session, tool_name, *, title=None):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        if title is not None:
            tw.title = title
        self.tool_window = tw
        parent = tw.ui_area

        from matplotlib import figure
        self.figure = f = figure.Figure(dpi=100, figsize=(2,2))

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
        self.canvas = c = Canvas(f)
        parent.setMinimumHeight(1)  # Matplotlib gives divide by zero error when plot resized to 0 height.
        c.setParent(parent)

        from Qt.QtWidgets import QHBoxLayout
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

    def move(self, delta_x, delta_y):
        '''Move plot objects by delta values in window pixels.'''
        win = self.tool_window.ui_area
        w,h = win.width(), win.height()
        if w == 0 or h == 0:
            return
        a = self.axes
        x0, x1 = a.get_xlim()
        xs = delta_x/w * (x1-x0)
        nx0, nx1 = x0-xs, x1-xs
        y0, y1 = a.get_ylim()
        ys = delta_y/h * (y1-y0)
        ny0, ny1 = y0-ys, y1-ys
        a.set_xlim(nx0, nx1)
        a.set_ylim(ny0, ny1)
        self.canvas.draw()

    def zoom(self, factor):
        '''
        Zoom plot objects by specified factor by changing
        the displayed limits of the plot.  Objects do not change size.
        '''
        a = self.axes
        x0, x1 = a.get_xlim()
        xmid, xsize = 0.5*(x0+x1), x1-x0
        xh = 0.5*xsize/factor
        nx0, nx1 = xmid-xh, xmid+xh
        y0, y1 = a.get_ylim()
        ymid, ysize = 0.5*(y0+y1), y1-y0
        yh = 0.5*ysize/factor
        ny0, ny1 = ymid-yh, ymid+yh
        a.set_xlim(nx0, nx1)
        a.set_ylim(ny0, ny1)
        self.canvas.draw()

    def matplotlib_mouse_event(self, x, y):
        '''Used for detecting clicked matplotlib canvas item using Artist.contains().'''
        h = self.tool_window.ui_area.height()
        # TODO: matplotlib 2.0.2 bug on mac retina displays, requires 2x scaling
        # for picking objects to work. ChimeraX ticket #762.
        pr = self.tool_window.ui_area.devicePixelRatio()
        from matplotlib.backend_bases import MouseEvent
        e = MouseEvent('context menu', self.canvas, pr*x, pr*(h-y))
        return e

    def save_plot_as(self):
        fmts = [self.session.data_formats[fmt_name] for fmt_name in ('Portable Network Graphics',
                                                                     'Scalable Vector Graphics',
                                                                     'Portable Document Format')]
        parent = self.tool_window.ui_area
        from chimerax.ui.open_save import SaveDialog
        save_dialog = SaveDialog(self.session, parent, "Save Plot", data_formats=fmts)
        if not save_dialog.exec():
            return
        filename = save_dialog.selectedFiles()[0]
        if not filename:
            from chimerax.core.errors import UserError
            raise UserError("No file specified for saving plot")
        format = {'PDF document (*.pdf)':'pdf',
                  'SVG image (*.svg)':'svg',
                  'PNG image (*.png)':'png'}[save_dialog.selectedNameFilter()]
        from os.path import splitext
        if splitext(filename)[1] == '':
            filename += '.' + format
        self.save_plot(filename, format = format)

    def save_plot(self, path, dpi = 300, pad_inches = 0.1, format = None):
        self.figure.savefig(path, dpi = dpi, pad_inches = pad_inches, format = format)

    def add_menu_entry(self, menu, text, callback, *args):
        '''Add menu item to context menu'''
        widget = self.tool_window.ui_area
        from Qt.QtGui import QAction
        a = QAction(text, widget)
        #a.setStatusTip("Info about this menu entry")
        a.triggered.connect(lambda *, cb=callback, args=args: cb(*args))
        menu.addAction(a)

# ------------------------------------------------------------------------------
#
class Graph(Plot):
    '''
    Show a graph of labeled nodes and edges.
    Left mouse click shows context menu.
    Ctrl-left or ctrl-right click calls mouse_click() method on object.
    Scroll zooms the plot.
    Middle and right mouse drags move the plotted objects.
    '''
    
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
        self._node_objects = []		# For looking up mouse clicked object by index
        self._edge_artist = None	# Matplotlib LineCollection for edge display
        self._edge_objects = []		# For looking up mouse clicked object by index
        self._labels = {}		# Maps group to Matplotlib Text object for labels

        c = self.canvas
        c.mousePressEvent = self._mouse_press
        c.mouseMoveEvent = self._mouse_move
        c.mouseReleaseEvent = self._mouse_release
        c.wheelEvent = self._wheel_event
        c.contextMenuEvent = lambda event: None		# Don't show context menu on right click.
        self._last_mouse_xy = None
        self._dragged = False
        self._min_drag = 10	# pixels
        self._drag_mode = None

    def _make_graph(self):
        import networkx as nx
        # Keep graph nodes in order so we can reproduce the same layout.
        G = nx.Graph()
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
        self._node_objects = nodes

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
            from chimerax.geometry import distance
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
        pos = event.pos()
        self._last_mouse_xy = (pos.x(), pos.y())
        self._dragged = False
        b = event.button()
        from Qt.QtCore import Qt
        if b == Qt.LeftButton:
            if self.is_ctrl_key_pressed(event):
                drag_mode = 'select'	# Click on object.
            elif self.is_alt_key_pressed(event) or self.is_command_key_pressed(event):
                drag_mode = 'translate'
            else:
                self.tool_window._show_context_menu(event)
                drag_mode = 'menu'
        elif b == Qt.MiddleButton:
            drag_mode = 'translate'
        elif b == Qt.RightButton:
            if self.is_ctrl_key_pressed(event):
                drag_mode = 'select'	# Click on object (same as ctrl-left)
            else:
                drag_mode = 'translate'
        else:
            drag_mode = None

        self._drag_mode = drag_mode
        
    def _mouse_move(self, event):
        if self._last_mouse_xy is None:
            self._mouse_press(event)
            return 	# Did not get mouse down

        pos = event.pos()
        x, y = pos.x(), pos.y()
        lx, ly = self._last_mouse_xy
        dx, dy = x-lx, y-ly
        if abs(dx) < self._min_drag and abs(dy) < self._min_drag:
            return
        self._last_mouse_xy = (x,y)
        self._dragged = True

        mode = self._drag_mode
        if mode == 'zoom':
            # Zoom
            h = self.tool_window.ui_area.height()
            from math import exp
            factor = exp(3*dy/h)
            self.zoom(factor)
        elif mode == 'translate':
            # Translate plot
            self.move(dx, -dy)
    
    def _mouse_release(self, event):
        if not self._dragged and self._drag_mode == 'select':
            pos = event.pos()
            item = self._clicked_item(pos.x(), pos.y())
            self.mouse_click(item, event)

        self._last_mouse_xy = None
        self._dragged = False
        self._drag_mode = None
        
    def _wheel_event(self, event):
        delta = event.angleDelta().y()  # Typically 120 per wheel click, positive down.
        from math import exp
        factor = exp(delta / 1200)
        self.zoom(factor)

    def mouse_click(self, node_or_edge, event):
        pass

    def is_alt_key_pressed(self, event):
        from Qt.QtCore import Qt
        return event.modifiers() & Qt.AltModifier

    def is_command_key_pressed(self, event):
        from Qt.QtCore import Qt
        import sys
        if sys.platform == 'darwin':
            # Mac command-key gives Qt control modifier.
            return event.modifiers() & Qt.ControlModifier
        return False

    def is_ctrl_key_pressed(self, event):
        from Qt.QtCore import Qt
        import sys
        if sys.platform == 'darwin':
            # Mac ctrl-key gives Qt meta modifier and Mac Command key gives Qt ctrl modifier.
            return event.modifiers() & Qt.MetaModifier
        return event.modifiers() & Qt.ControlModifier

    def _clicked_item(self, x, y):
        # Check for node click
        e = self.matplotlib_mouse_event(x,y)
        c,d = self._node_artist.contains(e)
        item = None
        if c:
            i = d['ind'][0]
            item = self._node_objects[i]
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
        self.add_menu_entry(menu, 'Save Plot As...', self.save_plot_as)
        
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
