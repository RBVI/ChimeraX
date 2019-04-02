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

from chimerax.core.tools import ToolInstance
from chimerax.core.settings import Settings


class ToolbarSettings(Settings):
    AUTO_SAVE = {
        "show_button_labels": True,
        "show_group_labels": True,
    }


class ToolbarTool(ToolInstance):

    SESSION_ENDURING = True
    SESSION_SAVE = False        # No session saving for now
    PLACEMENT = "top"
    CUSTOM_SCHEME = "toolbar"
    help = "help:user/tools/Toolbar.html"  # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Toolbar"
        self.settings = ToolbarSettings(session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self._build_ui()
        self.tool_window.fill_context_menu = self.fill_context_menu
        # kludge to hide title bar
        from PyQt5.QtWidgets import QWidget
        self.tool_window._kludge.dock_widget.setTitleBarWidget(QWidget())

        # TODO: Temporarily remove default toolbars
        from chimerax.core.commands import run
        # run(session, "toolshed hide 'Density Map Toolbar'", log=False)
        run(session, "toolshed hide 'Graphics Toolbar'", log=False)
        run(session, "toolshed hide 'Molecule Display Toolbar'", log=False)
        run(session, "toolshed hide 'Mouse Modes for Right Button'", log=False)

    def _build_ui(self):
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        margins = layout.contentsMargins()
        margins.setTop(0)
        margins.setBottom(0)
        layout.setContentsMargins(margins)
        self.ttb = TabbedToolbar(
            self.tool_window.ui_area, show_group_titles=self.settings.show_group_labels,
            show_button_titles=self.settings.show_button_labels)
        layout.addWidget(self.ttb)
        self._build_buttons()
        self.tool_window.ui_area.setLayout(layout)
        self.tool_window.manage(self.PLACEMENT)

    def fill_context_menu(self, menu, x, y):
        # avoid having actions destroyed when this routine returns
        # by stowing a reference in the menu itself
        from PyQt5.QtWidgets import QAction
        button_labels = QAction("Show button labels", menu)
        button_labels.setCheckable(True)
        button_labels.setChecked(self.settings.show_button_labels)
        button_labels.toggled.connect(lambda arg, f=self._set_button_labels: f(arg))
        menu.addAction(button_labels)
        group_labels = QAction("Show group labels", menu)
        group_labels.setCheckable(True)
        group_labels.setChecked(self.settings.show_group_labels)
        group_labels.toggled.connect(lambda arg, f=self._set_group_labels: f(arg))
        menu.addAction(group_labels)

    def _set_button_labels(self, show_button_labels):
        self.settings.show_button_labels = show_button_labels
        self.ttb.set_show_button_titles(show_button_labels)

    def _set_group_labels(self, show_group_labels):
        self.settings.show_group_labels = show_group_labels
        self.ttb.set_show_group_titles(show_group_labels)

    def handle_scheme(self, cmd):
        # First check that the path is a real command
        kind, value = cmd.split(':', maxsplit=1)
        if kind == "shortcut":
            self.session.keyboard_shortcuts.run_shortcut(value)
        elif kind == "mouse":
            button_to_bind = 'right'
            from chimerax.core.commands import run
            run(self.session, f'ui mousemode {button_to_bind} "{value}"')
        elif kind == "cmd":
            from chimerax.core.commands import run
            run(self.session, f'{value}')
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown toolbar command: %s" % cmd)

    def _build_buttons(self):
        import os
        import chimerax.shortcuts
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap, QIcon
        shortcut_icon_dir = os.path.join(chimerax.shortcuts.__path__[0], 'icons')
        dir_path = os.path.join(os.path.dirname(__file__), 'icons')
        for tab in _Toolbars:
            help_url, info = _Toolbars[tab]
            for (section, compact) in info:
                shortcuts = info[(section, compact)]
                for what, icon_file, descrip, tooltip in shortcuts:
                    kind, value = what.split(':', 1)
                    if kind == "mouse":
                        m = self.session.ui.mouse_modes.named_mode(value)
                        if m is None:
                            continue
                        icon_path = m.icon_path
                    else:
                        icon_path = os.path.join(shortcut_icon_dir, icon_file)
                        if not os.path.exists(icon_path):
                            icon_path = os.path.join(dir_path, icon_file)
                    pm = QPixmap(icon_path)
                    # Toolbutton will scale down, but not up, so give large icons
                    icon = QIcon(pm.scaledToHeight(128, Qt.SmoothTransformation))
                    if not tooltip:
                        tooltip = descrip
                    self.ttb.add_button(
                        tab, section, descrip,
                        lambda e, what=what, self=self: self.handle_scheme(what),
                        icon, tooltip, compact)
        self.ttb.show_category('Home')


_Toolbars = {
    "Home": (
        None,
        {
            ("Atoms", True): [
                ("shortcut:da", "atomshow.png", "Show", "Show atoms"),
                ("shortcut:ha", "atomhide.png", "Hide", "Hide atoms")],
            ("Cartoons", True): [
                ("shortcut:rb", "ribshow.png", "Show", "Show cartoons"),
                ("shortcut:hr", "ribhide.png", "Hide", "Hide cartoons")],
            ("Styles", False): [
                ("shortcut:st", "stick.png", "Stick", "Display atoms in stick style"),
                ("shortcut:sp", "sphere.png", "Sphere", "Display atoms in sphere style"),
                ("shortcut:bs", "ball.png", "Ball and stick", "Display atoms in ball and stick style")],
            ("Background", False): [
                ("shortcut:wb", "whitebg.png", "White", "White background"),
                ("shortcut:bk", "blackbg.png", "Black", "Black background")],
            ("Lighting", False): [
                ("shortcut:ls", "simplelight.png", "Simple", "Simple lighting"),
                ("shortcut:la", "softlight.png", "Soft", "Ambient lighting"),
                ("shortcut:lf", "fulllight.png", "Full", "Full lighting")],
        },
    ),
    "File": (
        None,
        {
            ("Session", False): [
                ("cmd:open browse", "open-in-app.png", "Open", "Open data file"),
                ("cmd:save browse", "content-save.png", "Save", "Save session file"),
                ("cmd:close session", "close-box.png", "Close", "Close current session"),
                ("cmd:exit", "exit.png", "Exit", "Exit application")],
            ("Image", False): [
                ("cmd:save image browse", "file-image.png", "Save", "Save image")],
        },
    ),
    "Molecule Display": (
        "help:user/tools/moldisplay.html",
        {
            ("Last action", True): [
                ("cmd:undo", "undo-variant.png", "Undo", "Undo last action"),
                ("cmd:redo", "redo-variant.png", "Redo", "Redo last action")],
            ("Atoms", True): [
                ("shortcut:da", "atomshow.png", "Show", "Show atoms"),
                ("shortcut:ha", "atomhide.png", "Hide", "Hide atoms")],
            ("Cartoons", True): [
                ("shortcut:rb", "ribshow.png", "Show", "Show cartoons"),
                ("shortcut:hr", "ribhide.png", "Hide", "Hide cartoons")],
            ("Surfaces", True): [
                ("shortcut:ms", "surfshow.png", "Show", "Show surfaces"),
                ("shortcut:hs", "surfhide.png", "Hide", "Hide surfaces")],
            ("Styles", False): [
                ("shortcut:st", "stick.png", "Stick", "Display atoms in stick style"),
                ("shortcut:sp", "sphere.png", "Sphere", "Display atoms in sphere style"),
                ("shortcut:bs", "ball.png", "Ball-and-stick", "Display atoms in ball and stick style")],
            ("Color Atoms", False): [
                ("shortcut:ce", "colorbyelement.png", "By element", "Color non-carbon atoms by element"),
                ("shortcut:cc", "colorbychain.png", "By chain", "Color chains")],
            ("Nucleotides", False): [
                ("cmd:nuc selAtoms atoms; style selAtoms & nucleic ringFill off", "nuc-atoms.png", "Plain", "Remove nucleotide abstraction"),
                ("cmd:nuc selAtoms atoms; style selAtoms & nucleic ringFill on", "nuc-fill.png", "Filled", "Fill nucleotide rings"),
                ("cmd:nuc selAtoms slab; style selAtoms & nucleic ringFill on", "nuc-slab.png", "Slab", "Show nucleotide bases as slabs and fill sugars"),
                ("cmd:nuc selAtoms tube", "nuc-tube.png", "Tube/Slab", "Show nucleotide bases as slabs and sugars as tubes"),
                ("cmd:nuc selAtoms stubs", "nuc-stubs.png", "Stubs", "Show nucleotides as stubs"),
                ("cmd:nuc selAtoms ladder", "nuc-ladder.png", "Ladder", "Show nucleotide h-bond ladders"),
                ("cmd:color selAtoms bynuc", "nuc-color.png", "Color", "Color by nucleotide")],

            ("Misc", False): [
                ("shortcut:hb", "hbonds.png", "Show hydrogen bonds", "Show hydrogen bonds")],
        },
    ),
    "Graphics": (
        "help:user/tools/graphics.html",
        {
            ("Last action", True): [
                ("cmd:undo", "undo-variant.png", "Undo", "Undo last action"),
                ("cmd:redo", "redo-variant.png", "Redo", "Redo last action")],
            ("Background", True): [
                ("shortcut:wb", "whitebg.png", "White", "White background"),
                ("shortcut:gb", "graybg.png", "Gray", "Gray background"),
                ("shortcut:bk", "blackbg.png", "Black", "Black background")],
            ("Lighting", False): [
                ("shortcut:ls", "simplelight.png", "Simple", "Simple lighting"),
                ("shortcut:sh", "shadow.png", "Single shadow", "Toggle shadows"),
                ("shortcut:la", "softlight.png", "Soft", "Ambient lighting"),
                ("shortcut:lf", "fulllight.png", "Full", "Full lighting"),
                ("shortcut:lF", "flat.png", "Flat", "Flat lighting"),
                ("shortcut:se", "silhouette.png", "Silhouettes", "Toggle silhouettes")],
            ("Camera", False): [
                ("shortcut:va", "viewall.png", "View all", "View all"),
                ("shortcut:dv", "orient.png", "Default orientation", "Default orientation")],
            ("Images", False): [
                ("shortcut:sx", "camera.png", "Save snapshot to desktop", "Save snapshot to desktop"),
                ("shortcut:vd", "video.png", "Record spin movie", "Record spin movie")],
        }
    ),
    "Density Map": (
        "help:user/tools/densitymaps.html",
        {
            ("Map", False): [
                ("shortcut:sM", "showmap.png", "Show", "Show map"),
                ("shortcut:hM", "hidemap.png", "Hide", "Hide map")],
            ("Style", False): [
                ("shortcut:fl", "mapsurf.png", "As surface", "Show map or surface in filled style"),
                ("shortcut:me", "mesh.png", "As mesh", "Show map or surface as mesh"),
                ("shortcut:gs", "mapimage.png", "As image", "Show map as grayscale"),
                ("shortcut:tt", "icecube.png", "Transparent surface", "Toggle surface transparency"),
                ("shortcut:ob", "outlinebox.png", "Outline box", "Toggle outline box")],
            ("Steps", False): [
                ("shortcut:s1", "step1.png", "Step 1", "Show map at step 1"),
                ("shortcut:s2", "step2.png", "Step 2", "Show map at step 2")],
            ("Fitting", False): [
                ("shortcut:fT", "fitmap.png", "Fit in map", "Fit map in map")],
            ("Rendering", False): [
                ("shortcut:sb", "diffmap.png", "Compute difference", "Subtract map from map"),
                ("shortcut:gf", "smooth.png", "Smooth", "Smooth map"),
                ("shortcut:pl", "plane.png", "One plane", "Show one plane"),
                ("shortcut:o3", "orthoplanes.png", "Orthogonal planes", "Show 3 orthogonal planes"),
                ("shortcut:pa", "fullvolume.png", "Full volume", "Show all planes")],
            ("Solid", False): [
                ("shortcut:zs", "xyzslice.png", "xyz slices", "Volume xyz slices"),
                ("shortcut:ps", "perpslice.png", "Perpendicular slices", "Volume perpendicular slices"),
                ("shortcut:aw", "airways.png", "Airways preset", "Airways preset"),
                ("shortcut:dc", "initialcurve.png", "Default curve", "Default volume curve")],
        }
    ),
    "Right Mouse": (
        "help:user/tools/mousemodes.html",
        {
            ("Models", False): [
                ("mouse:select", None, "Select", "Select models"),
                ("mouse:rotate", None, "Rotate", "Rotate models"),
                ("mouse:translate", None, "Translate", "Translate models"),
                ("mouse:zoom", None, "Zoom", "Zoom view"),
                ("mouse:rotate and select", None, "Rotate and select", "Select and rotate models"),
                ("mouse:translate selected", None, "Translate Selected", "Translate selected models"),
                ("mouse:rotate selected", None, "Rotate Selected", "Rotate selected models")],
            ("Clip", False): [
                ("mouse:clip", None, "Clip", "Activate clipping"),
                ("mouse:clip rotate", None, "Clip rotate", "Rotate clipping planes"),
                ("mouse:zone", None, "Zone", "Limit display to zone around clicked residues")],
            ("Label", False): [
                ("mouse:label", None, "Label", "Toggle atom or cartoon label"),
                ("mouse:move label", None, "Move label", "Reposition 2D label")],
            ("Misc", False): [
                ("mouse:pivot", None, "Pivot", "Set center of rotation at atom"),
                ("mouse:bond rotation", None, "Bond rotation", "Adjust torsion angle"),
                ("mouse:distance", None, "distance", "Toggle distance monitor between two atoms"),
                ("mouse:swapaa", None, "Swapaa", "Mutate and label residue")],
            ("Dynamics", False): [
                ("mouse:tug", None, "Tug", "Drag atom while applying dynamics"),
                ("mouse:minimize", None, "Minimize", "Jiggle residue and its neighbors")],
            ("Volumes", False): [
                ("mouse:place marker", None, "Place marker", None),
                ("mouse:contour level", None, "Contour level", "Adjust volume data threshold level"),
                ("mouse:windowing", None, "Windowing", "Adjust volume data thresholds collectively"),
                ("mouse:move planes", None, "Move planes", "Move plane or slab along its axis to show a different section"),
                ("mouse:crop volume", None, "Crop", "Crop volume data dragging any face of box outline"),
                ("mouse:play map series", None, "Play series", "Play map series")],
        }
    ),
}
