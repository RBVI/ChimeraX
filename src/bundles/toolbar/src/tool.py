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
        "show_section_labels": True,
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
        self.tool_window = MainToolWindow(self, close_destroys=False, hide_title_bar=True)
        self._build_ui()
        self.tool_window.fill_context_menu = self.fill_context_menu

    def _build_ui(self):
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        margins = layout.contentsMargins()
        margins.setTop(0)
        margins.setBottom(0)
        layout.setContentsMargins(margins)
        self.ttb = TabbedToolbar(
            self.tool_window.ui_area, show_section_titles=self.settings.show_section_labels,
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
        section_labels = QAction("Show section labels", menu)
        section_labels.setCheckable(True)
        section_labels.setChecked(self.settings.show_section_labels)
        section_labels.toggled.connect(lambda arg, f=self._set_section_labels: f(arg))
        menu.addAction(section_labels)

    def _set_button_labels(self, show_button_labels):
        self.settings.show_button_labels = show_button_labels
        self.ttb.set_show_button_titles(show_button_labels)

    def _set_section_labels(self, show_section_labels):
        self.settings.show_section_labels = show_section_labels
        self.ttb.set_show_section_titles(show_section_labels)

    def handle_scheme(self, cmd):
        # First check that the path is a real command
        if callable(cmd):
            cmd(self.session)
            return
        kind, value = cmd.split(':', maxsplit=1)
        if kind == "shortcut":
            from chimerax.shortcuts import shortcuts
            shortcuts.keyboard_shortcuts(self.session).run_shortcut(value)
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
            for (section, compact), shortcuts in info.items():
                if compact:
                    self.ttb.set_section_compact(tab, section, True)
                for item in shortcuts:
                    if len(item) == 4:
                        (what, icon_file, descrip, tooltip) = item
                        kw = {}
                    else:
                        (what, icon_file, descrip, tooltip, kw) = item
                    kind, value = what.split(':', 1) if isinstance(what, str) else (None, None)
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
#                    icon = QIcon(pm.scaledToHeight(128, Qt.SmoothTransformation))
                    icon = QIcon(pm)
                    if not tooltip:
                        tooltip = descrip
                    if kind == "mouse":
                        kw["vr_mode"] = what[6:]   # Allows VR to recognize mouse mode tool buttons
                    if descrip and not descrip[0].isupper():
                        descrip = descrip.capitalize()
                    self.ttb.add_button(
                        tab, section, descrip,
                        lambda e, what=what, self=self: self.handle_scheme(what),
                        icon, tooltip, **kw)
        self.ttb.show_tab('Home')

def _file_open(session):
    session.ui.main_window.file_open_cb(session)
def _file_save(session):
    session.ui.main_window.file_save_cb(session)

_Toolbars = {
    "Home": (
        None,
        {
            ("File", False): [
                (_file_open, "open-in-app.png", "Open", "Open data file"),
                (_file_save, "content-save.png", "Save", "Save session file"),
                #("cmd:close session", "close-box.png", "Close", "Close current session"),
                #("cmd:exit", "exit.png", "Exit", "Exit application"),
            ],
            ("Images", False): [
                ("shortcut:sx", "camera.png", "Snapshot", "Save snapshot to desktop"),
                ("shortcut:vd", "video.png", "Spin movie", "Record spin movie"),
            ],
            ("Atoms", True): [
                ("shortcut:da", "atomshow.png", "Show", "Show atoms"),
                ("shortcut:ha", "atomhide.png", "Hide", "Hide atoms"),
            ],
            ("Cartoons", True): [
                ("shortcut:rb", "ribshow.png", "Show", "Show cartoons"),
                ("shortcut:hr", "ribhide.png", "Hide", "Hide cartoons"),
            ],
            ("Styles", False): [
                ("shortcut:st", "stick.png", "Stick", "Display atoms in stick style"),
                ("shortcut:sp", "sphere.png", "Sphere", "Display atoms in sphere style"),
                ("shortcut:bs", "ball.png", "Ball && stick", "Display atoms in ball and stick style"),
            ],
            ("Background", False): [
                ("shortcut:wb", "whitebg.png", "White", "White background"),
                ("shortcut:bk", "blackbg.png", "Black", "Black background"),
            ],
            ("Lighting", False): [
                ("shortcut:ls", "simplelight.png", "Simple", "Simple lighting"),
                ("shortcut:la", "softlight.png", "Soft", "Ambient lighting"),
                ("shortcut:lf", "fulllight.png", "Full", "Full lighting"),
            ],
#            ("Undo", True): [
#                ("cmd:undo", "undo-variant.png", "Undo", "Undo last action"),
#                ("cmd:redo", "redo-variant.png", "Redo", "Redo last action"),
#            ],
        },
    ),
    "Molecule Display": (
        "help:user/tools/moldisplay.html",
        {
            ("Atoms", True): [
                ("shortcut:da", "atomshow.png", "Show", "Show atoms"),
                ("shortcut:ha", "atomhide.png", "Hide", "Hide atoms"),
            ],
            ("Cartoons", True): [
                ("shortcut:rb", "ribshow.png", "Show", "Show cartoons"),
                ("shortcut:hr", "ribhide.png", "Hide", "Hide cartoons"),
            ],
            ("Surfaces", True): [
                ("shortcut:ms", "surfshow.png", "Show", "Show surfaces"),
                ("shortcut:hs", "surfhide.png", "Hide", "Hide surfaces"),
            ],
            ("Styles", False): [
                ("shortcut:st", "stick.png", "Stick", "Display atoms in stick style"),
                ("shortcut:sp", "sphere.png", "Sphere", "Display atoms in sphere style"),
                ("shortcut:bs", "ball.png", "Ball && stick", "Display atoms in ball and stick style"),
                ("cmd:nucleotides selAtoms atoms; style nucleic & selAtoms stick", "nuc-atoms.png", "Plain", "Remove nucleotides styling", {'group': 'nuc'}),
                ("cmd:nucleotides selAtoms fill; style nucleic & selAtoms stick", "nuc-fill.png", "Filled", "Show nucleotides with filled rings", {'group': 'nuc'}),
                ("cmd:nucleotides selAtoms tube/slab shape box", "nuc-box.png", "Tube/\nSlab", "Show nucleotide bases as boxes and sugars as tubes", {'group': 'nuc'}),
                ("cmd:nucleotides selAtoms ladder", "nuc-ladder.png", "Ladder", "Show nucleotides as H-bond ladders", {'group': 'nuc'}),
            ],
            ("Coloring", False): [
                ("shortcut:ce", "colorbyelement.png", "heteroatom", "Color non-carbon atoms by element"),
                ("shortcut:cc", "colorbychain.png", "chain", "Color by chain"),
                ("shortcut:rB", "rainbow.png", "rainbow", 'Rainbow color N to C-terminus'),
                ("shortcut:bf", "bfactor.png", "b-factor", 'Color by b-factor'),
                ("shortcut:hp", "hydrophobicity.png", "hydrophobic", 'Color surface by hydrophobicity'),
                ("cmd:color selAtoms bynuc", "nuc-color.png", "nucleotide", "Color by nucleotide"),
            ],
            ("Analysis", False): [
                ("shortcut:hb", "hbondsflat.png", "H-bonds", "Show hydrogen bonds"),
                ("shortcut:HB", "hbondsflathide.png", "Hide H-bonds", "Hide hydrogen bonds"),
                ("shortcut:sq", "sequence.png", "Sequence", "Show polymer sequence"),
                ("shortcut:if", "interfaces.png", "Interfaces", "Show chain contacts diagram"),
            ],
        },
    ),
    "Nucleotides": (
        None,
        {
            ("Styles", False): [
                ("cmd:nucleotides selAtoms atoms; style nucleic & selAtoms stick", "nuc-atoms.png", "Plain", "Remove nucleotides styling"),
                ("cmd:nucleotides selAtoms fill; style nucleic & selAtoms stick", "nuc-fill.png", "Filled", "Show nucleotides with filled rings"),
                ("cmd:nucleotides selAtoms slab; style nucleic & selAtoms stick", "nuc-slab.png", "Slab", "Show nucleotide bases as slabs and fill sugars"),
                ("cmd:nucleotides selAtoms tube/slab shape box", "nuc-box.png", "Tube/\nSlab", "Show nucleotide bases as boxes and sugars as tubes"),
                ("cmd:nucleotides selAtoms tube/slab shape ellipsoid", "nuc-elli.png", "Tube/\nEllipsoid", "Show nucleotide bases as ellipsoids and sugars as tubes"),
                ("cmd:nucleotides selAtoms tube/slab shape muffler", "nuc-muff.png", "Tube/\nMuffler", "Show nucleotide bases as mufflers and sugars as tubes"),
                ("cmd:nucleotides selAtoms ladder", "nuc-ladder.png", "Ladder", "Show nucleotides as H-bond ladders"),
                ("cmd:nucleotides selAtoms stubs", "nuc-stubs.png", "Stubs", "Show nucleotides as stubs"),
            ],
            ("Coloring", False): [
                ("cmd:color selAtoms bynuc", "nuc-color.png", "nucleotide", "Color by nucleotide"),
            ],
        },
    ),
    "Graphics": (
        "help:user/tools/graphics.html",
        {
            ("Background", True): [
                ("shortcut:wb", "whitebg.png", "White", "White background"),
                ("shortcut:gb", "graybg.png", "Gray", "Gray background"),
                ("shortcut:bk", "blackbg.png", "Black", "Black background"),
            ],
            ("Lighting & Effects", False): [
                ("shortcut:ls", "simplelight.png", "Simple", "Simple lighting"),
                ("shortcut:la", "softlight.png", "Soft", "Ambient lighting"),
                ("shortcut:lf", "fulllight.png", "Full", "Full lighting"),
                ("shortcut:lF", "flat.png", "Flat", "Flat lighting"),
                ("shortcut:sh", "shadow.png", "Shadow", "Toggle shadows"),
                ("shortcut:se", "silhouette.png", "Silhouettes", "Toggle silhouettes"),
            ],
            ("Camera", False): [
                ("shortcut:vs", "viewsel.png", "View selected", "View selected"),
                ("shortcut:va", "viewall.png", "View all", "View all"),
                ("shortcut:dv", "orient.png", "Orient", "Default orientation"),
            ],
        }
    ),
    "Map": (
        "help:user/tools/densitymaps.html",
        {
            ("Map", False): [
                ("shortcut:sM", "showmap.png", "Show", "Show map"),
                ("shortcut:hM", "hidemap.png", "Hide", "Hide map"),
            ],
            ("Style", False): [
                ("shortcut:fl", "mapsurf.png", "surface", "Show map or surface in filled style"),
                ("shortcut:me", "mesh.png", "mesh", "Show map or surface as mesh"),
                ("shortcut:gs", "mapimage.png", "image", "Show map as grayscale"),
                ("shortcut:tt", "icecube.png", "Transparent surface", "Toggle surface transparency"),
                ("shortcut:ob", "outlinebox.png", "Outline box", "Toggle outline box"),
            ],
            ("Steps", False): [
                ("shortcut:s1", "step1.png", "Step 1", "Show map at step 1"),
                ("shortcut:s2", "step2.png", "Step 2", "Show map at step 2"),
            ],
            ("Subregions", False): [
                ("shortcut:pl", "plane.png", "Z-plane", "Show one plane"),
                ("shortcut:o3", "orthoplanes.png", "Orthoplanes", "Show 3 orthogonal planes"),
                ("shortcut:pa", "fullvolume.png", "Full", "Show all planes"),
            ],
            ("Calculations", False): [
                ("shortcut:fT", "fitmap.png", "Fit", "Fit map in map"),
                ("shortcut:sb", "diffmap.png", "Subtract", "Subtract map from map"),
                ("shortcut:gf", "smooth.png", "Smooth", "Smooth map"),
            ],
            ("Image Display", False): [
                ("shortcut:zs", "xyzslice.png", "XYZ slices", "Volume XYZ slices"),
                ("shortcut:ps", "perpslice.png", "Perpendicular slices", "Volume perpendicular slices"),
                ("shortcut:aw", "airways.png", "Airways preset", "Airways preset"),
                ("shortcut:as", "ear.png", "skin preset", "skin preset"),
                ("shortcut:dc", "initialcurve.png", "Default thresholds", "Default volume curve"),
            ],
        }
    ),
    "Markers": (
        "help:user/tools/markerplacement.html",
        {
            ("Place markers", False): [
                ("mouse:mark maximum", None, "Maximum", "Mark maximum"),
                ("mouse:mark plane", None, "Plane", "Mark volume plane"),
                ("mouse:mark surface", None, "Surface", "Mark surface"),
                ("mouse:mark center", None, "Center", "Mark center of connected surface"),
                ("mouse:mark point", None, "Point", "Mark 3d point"),
            ],
            ("Adjust markers", False): [
                ("mouse:link markers", None, "Link", "Link consecutively clicked markers"),
                ("mouse:move markers", None, "Move", "Move markers"),
                ("mouse:resize markers", None, "Resize", "Resize markers or links"),
                ("mouse:delete markers", None, "Delete", "Delete markers or links"),
            ],
        }
    ),
    "Right Mouse": (
        "help:user/tools/mousemodes.html",
        {
            ("Movement", False): [
                ("mouse:select", None, "Select", "Select models"),
                ("mouse:rotate", None, "Rotate", "Rotate models"),
                ("mouse:translate", None, "Translate", "Translate models"),
                ("mouse:zoom", None, "Zoom", "Zoom view"),
                ("mouse:translate selected models", None, "Translate Selected", "Translate selected models"),
                ("mouse:rotate selected models", None, "Rotate Selected", "Rotate selected models"),
                ("mouse:pivot", None, "Pivot", "Set center of rotation at atom"),
            ],
            ("Annotation", False): [
                ("mouse:distance", None, "distance", "Toggle distance monitor between two atoms"),
                ("mouse:label", None, "Label", "Toggle atom or cartoon label"),
                ("mouse:move label", None, "Move label", "Reposition 2D label"),
            ],
            ("Clipping", False): [
                ("mouse:clip", None, "Clip", "Activate clipping"),
                ("mouse:clip rotate", None, "Clip rotate", "Rotate clipping planes"),
                ("mouse:zone", None, "Zone", "Limit display to zone around clicked residues"),
            ],
            ("Map", False): [
                ("mouse:contour level", None, "Contour level", "Adjust volume data threshold level"),
                ("mouse:move planes", None, "Move planes", "Move plane or slab along its axis to show a different section"),
                ("mouse:crop volume", None, "Crop", "Crop volume data dragging any face of box outline"),
                ("mouse:pick blobs", None, "Blob", "Measure and color connected parts of surface"),
                ("mouse:map eraser", None, "Erase", "Erase parts of a density map setting values in a sphere to zero"),
                ("mouse:play map series", None, "Play series", "Play map series"),
                ("mouse:windowing", None, "Windowing", "Adjust volume data thresholds collectively"),
            ],
            ("Structure Modification", False): [
                ("mouse:bond rotation", None, "Bond rotation", "Adjust torsion angle"),
                ("mouse:swapaa", None, "Swapaa", "Mutate and label residue"),
                ("mouse:tug", None, "Tug", "Drag atom while applying dynamics"),
                ("mouse:minimize", None, "Minimize", "Jiggle residue and its neighbors"),
            ],
        }
    ),
}
