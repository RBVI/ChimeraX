# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class ShortcutPanel(ToolInstance):

    shortcuts = []
    SESSION_ENDURING = True

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        self._icon_size = 40
        self._icons_per_row = 13
        self.tool_window = None

        from .shortcuts import keyboard_shortcuts
        self.keyboard_shortcuts = keyboard_shortcuts(session)

        parent = session.ui.main_window
        self.buttons = self.create_toolbar(parent)

    def create_toolbar(self, parent):
        from Qt.QtWidgets import QToolBar
        from Qt.QtGui import QIcon, QAction
        from Qt.QtCore import Qt, QSize
        tb = QToolBar(self.display_name, parent)
        tb.setStyleSheet('QToolBar{spacing:0px;}\n'
                         'QToolButton{padding:0px; margin:0px; background-color:white; border:none;}')
        s = self._icon_size
        tb.setIconSize(QSize(s,s))
        parent.add_tool_bar(self, Qt.TopToolBarArea, tb)
        for keys, icon_file, descrip in self.shortcuts:
            from os import path
            icon_dir = path.join(path.dirname(__file__), 'icons')
            action = QAction(QIcon(path.join(icon_dir, icon_file)), descrip, tb)
            def button_press_cb(event, keys=keys, ks=self.keyboard_shortcuts):
                ks.run_shortcut(keys)
            action.triggered.connect(button_press_cb)
            tb.addAction(action)
        tb.show()
        return tb

    def create_button_panel(self):
        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        p = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout
        layout = QVBoxLayout(p)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        b = self.create_panel_buttons(p)
        p.setLayout(layout)
        layout.addWidget(b)
        tw.manage(placement="side")

    def create_panel_buttons(self, parent):
        from Qt.QtWidgets import QFrame, QGridLayout, QToolButton, QActionGroup
        from Qt.QtGui import QIcon, QAction
        from Qt.QtCore import Qt, QSize
        tb = QFrame(parent)
        layout = QGridLayout(tb)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tb.setStyleSheet('QFrame{spacing:0px;}\n'
                         'QToolButton{padding:0px; margin:0px; border:none;}')
        group = QActionGroup(tb)
        s = self._icon_size
        columns = self._icons_per_row
        from os import path
        icon_dir = path.join(path.dirname(__file__), 'icons')
        for snum, (keys, icon_file, descrip) in enumerate(self.shortcuts):
            b = QToolButton(tb)
            b.setIconSize(QSize(s,s))
            action = QAction(QIcon(path.join(icon_dir, icon_file)), descrip, tb)
            b.setDefaultAction(action)
            def button_press_cb(event, keys=keys, ks=self.keyboard_shortcuts):
                ks.run_shortcut(keys)
            action.triggered.connect(button_press_cb)
            group.addAction(action)
            row, column = snum//columns, snum%columns
            layout.addWidget(b, row, column)

        return tb

    def display(self, show):
        if show:
            f = self.buttons.show
        else:
            f = self.buttons.hide
        self.session.ui.thread_safe(f)

    def displayed(self):
        return not self.buttons.isHidden()
        
    def display_panel(self, show):
        tw = self.tool_window
        if show:
            if tw is None:
                self.create_button_panel()
            self.tool_window.shown = True
        elif tw:
            tw.shown = False

    @classmethod
    def get_singleton(cls, session):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, cls.tool_name)

class MoleculeDisplayPanel(ShortcutPanel):
    tool_name = 'Molecule Display Toolbar'
    shortcuts = (
        ('da', 'atomshow.webp', 'Show atoms'),
        ('ha', 'atomhide.webp', 'Hide atoms'),
        ('rb', 'ribshow.webp', 'Show cartoon'),
        ('hr', 'ribhide.webp', 'Hide cartoon'),
        ('ms', 'surfshow.webp', 'Show molecular surface'),
        ('hs', 'surfhide.webp', 'Hide molecular surface'),
        ('st', 'stick.webp', 'Use stick style'),
        ('sp', 'sphere.webp', 'Use sphere style'),
        ('bs', 'ball.webp', 'Use ball-and-stick style'),
        ('ce', 'colorbyelement.webp', 'Color atoms by element'),
        ('cc', 'colorbychain.webp', 'Color atoms by chain'),
        ('hb', 'hbonds.webp', 'Find hydrogen bonds'),
    )
    help = "help:user/tools/moldisplay.html"

class GraphicsPanel(ShortcutPanel):
    tool_name = 'Graphics Toolbar'
    shortcuts = (
        ('wb', 'whitebg.webp', 'White background'),
        ('gb', 'graybg.webp', 'Gray background'),
        ('bk', 'blackbg.webp', 'Black background'),
        ('ls', 'simplelight.webp', 'Simple lighting'),
        ('sh', 'shadow.webp', 'Single shadow'),
        ('la', 'softlight.webp', 'Soft lighting'),
        ('lf', 'fulllight.webp', 'Full lighting'),
        ('lF', 'flat.webp', 'Flat lighting'),
        ('se', 'silhouette.webp', 'Silhouette edges'),
        ('va', 'viewall.webp', 'View all'),
        ('dv', 'orient.webp', 'Standard orientation'),
        ('sx', 'camera.webp', 'Save snapshot to desktop'),
        ('vd', 'video.webp', 'Record spin movie'),
    )
    help = "help:user/tools/graphics.html"

class DensityMapPanel(ShortcutPanel):
    tool_name = 'Density Map Toolbar'
    shortcuts = (
        ('sM', 'showmap.webp', 'Show map'),
        ('hM', 'hidemap.webp', 'Hide map'),
        ('fl', 'mapsurf.webp', 'Map as surface'),
        ('me', 'mesh.webp', 'Map as mesh'),
        ('gs', 'mapimage.webp', 'Map as image'),
        ('s1', 'step1.webp', 'Map step 1'),
        ('s2', 'step2.webp', 'Map step 2'),
#        ('s4', 'step4.webp', 'Map step 4'),
        ('fT', 'fitmap.webp', 'Fit map in map'),
        ('sb', 'diffmap.webp', 'Compute difference map'),
        ('gf', 'smooth.webp', 'Smooth map'),
        ('tt', 'icecube.webp', 'Transparent surface'),
        ('ob', 'outlinebox.webp', 'Show outline box'),
        ('pl', 'plane.webp', 'Show one plane'),
        ('o3', 'orthoplanes.webp', 'Orthogonal planes'),
        ('pa', 'fullvolume.webp', 'Show full volume'),
        ('zs', 'xyzslice.webp', 'Volume xyz slices'),
        ('ps', 'perpslice.webp', 'Volume perpendicular slices'),
        ('aw', 'airways.webp', 'Airways preset'),
        ('as', 'ear.webp', 'Skin preset'),
        ('dc', 'initialcurve.webp', 'Default volume curve'),
    )
    help = "help:user/tools/densitymaps.html"

panel_classes = {
    MoleculeDisplayPanel.tool_name: MoleculeDisplayPanel,
    GraphicsPanel.tool_name: GraphicsPanel,
    DensityMapPanel.tool_name: DensityMapPanel,
}
