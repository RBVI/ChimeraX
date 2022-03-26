# vim: set expandtab shiftwidth=4 softtabstop=4:

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

# -----------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class AlphaFoldHeatmap(ToolInstance):

    help = 'help:user/tools/alphafold.html'

    def __init__(self, session, tool_name):

        self._alphafold_model = None
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        heading = ('<html>Show AlphaFold predicted residue-residue alignment errors (PAE).'
                   '<br>Drag a box to color model residues.</html>')
        from Qt.QtWidgets import QLabel
        self._heading = hl = QLabel(heading)
        layout.addWidget(hl)

        self._heatmap = gv = Heatmap(parent, self._rectangle_select, self._rectangle_clear)
        from Qt.QtWidgets import QSizePolicy
        from Qt.QtCore import Qt
        gv.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#        gv.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(gv)

        from Qt.QtWidgets import QGraphicsScene
        self._scene = gs = QGraphicsScene(gv)
        gs.setSceneRect(0, 0, 500, 500)
        gv.setScene(gs)

        self._pixmap_item = None	# Heatmap image

        # Color Domains button
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)
        
        layout.addStretch(1)    # Extra space at end

        tw.manage(placement=None)	# Start floating
        
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Color PAE Domains', self._color_domains),
                        ('Color pLDDT', self._color_plddt),
                        ('Help', self._show_help)],
                       spacing = 10)
        return f

    # ---------------------------------------------------------------------------
    #
    def _color_domains(self):
        m = self._alphafold_model
        if m is None:
            return
        from .heatmap import pae_domains, color_by_pae_domain
        if self._clusters is None:
            self._clusters = pae_domains(self._pae_matrix)
            from chimerax.core.colors import random_colors
            self._cluster_colors = random_colors(len(self._clusters), seed=0)
        color_by_pae_domain(m.residues, self._clusters, colors=self._cluster_colors)
        
    # ---------------------------------------------------------------------------
    #
    def _color_plddt(self):
        m = self._alphafold_model
        if m is None:
            return

        cmd = 'color bfactor #%s palette alphafold log false' % m.id_string
        from chimerax.core.commands import run
        run(self.session, cmd, log = False)

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    # ---------------------------------------------------------------------------
    #
    def set_heatmap(self, pae_path, model = None):
        title = (f'<html>Predicted residue-residue distance errors (PAE) for {model.name}'
                 '<br>Drag a box to color model residues.</html>')
        self._heading.setText(title)
        pi = self._pixmap_item
        if pi is not None:
            self._scene.removeItem(pi)
        from .heatmap import read_pae_matrix, pae_rgb, pae_pixmap
        self._pae_matrix = matrix = read_pae_matrix(pae_path)
        self._clusters = None
        rgb = pae_rgb(matrix)
        pixmap = pae_pixmap(rgb)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._heatmap._pixmap = self._pixmap_item
        self._alphafold_model = model

    # ---------------------------------------------------------------------------
    #
    def _rectangle_select(self, xy1, xy2):
        x1,y1 = xy1
        x2,y2 = xy2
        r1, r2 = int(min(x1,x2)), int(max(x1,x2))
        r3, r4 = int(min(y1,y2)), int(max(y1,y2))
        if r2 < r3 or r4 < r1:
            # Use two colors
            self._color_residues(r1, r2, 'lime')
            self._color_residues(r3, r4, 'magenta')
        else:
            # Use single color
            self._color_residues(min(r1,r3), max(r2,r4), 'lime')

    # ---------------------------------------------------------------------------
    #
    def _color_residues(self, r1, r2, colorname):
        m = self._alphafold_model
        if m is None:
            return

        from chimerax.core.colors import BuiltinColors
        color = BuiltinColors[colorname].uint8x4()
        residues = m.residues[r1:r2+1]
        residues.ribbon_colors = color
        residues.atoms._colors = color
        
    # ---------------------------------------------------------------------------
    #
    def _rectangle_clear(self):
        self._color_plddt()
        
    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, AlphaFoldHeatmap, 'AlphaFoldHeatmap', create=create)

from Qt.QtWidgets import QGraphicsView
class Heatmap(QGraphicsView):
    def __init__(self, parent, rectangle_select_cb=None, rectangle_clear_cb=None):
        QGraphicsView.__init__(self, parent)
        self.click_callbacks = []
        self.drag_callbacks = []
        self.mouse_down = False
        self._drag_box = None
        self._down_xy = None
        self._rectangle_select_callback = rectangle_select_cb
        self._rectangle_clear_callback = rectangle_clear_cb
    def sizeHint(self):
        from Qt.QtCore import QSize
        return QSize(500,500)
    def resizeEvent(self, event):
        # Rescale histogram when window resizes
        self.fitInView(self.sceneRect())
        QGraphicsView.resizeEvent(self, event)
    def mousePressEvent(self, event):
        self.mouse_down = True
        for cb in self.click_callbacks:
            cb(event)
        self._down_xy = self._scene_position(event)
        self._clear_drag_box()
        if self._rectangle_clear_callback:
            self._rectangle_clear_callback()
    def mouseMoveEvent(self, event):
        self._drag(event)
    def _drag(self, event):
        # Only process mouse move once per graphics frame.
        for cb in self.drag_callbacks:
            cb(event)
        self._draw_drag_box(event)
    def mouseReleaseEvent(self, event):
        self.mouse_down = False
        self._drag(event)
        if self._rectangle_select_callback and self._down_xy:
            self._rectangle_select_callback(self._down_xy, self._scene_position(event))
    def _scene_position(self, event):
        p = self.mapToScene(event.pos())
        return p.x(), p.y()
    def _draw_drag_box(self, event):
        x1,y1 = self._down_xy
        x2,y2 = self._scene_position(event)
        x,y,w,h = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
        box = self._drag_box
        if box is None:
            scene = self.scene()
            self._drag_box = scene.addRect(x,y,w,h)
        else:
            box.setRect(x,y,w,h)
    def _clear_drag_box(self):
        box = self._drag_box
        if box:
            self.scene().removeItem(box)
            self._drag_box = None
            
# -----------------------------------------------------------------------------
#
def heatmap_panel(session, create = False):
    return AlphaFoldHeatmap.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_heatmap_panel(session):
    return heatmap_panel(session, create = True)
