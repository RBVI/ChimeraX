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
            if ' ' in value:
                value = '"%s"' % value
            run(self.session, f'ui mousemode {button_to_bind} {value}')
        elif kind == "cmd":
            from chimerax.core.commands import run
            run(self.session, f'{value}')
        else:
            from chimerax.core.errors import UserError
            raise UserError("unknown toolbar command: %s" % cmd)

    def _add_mouse_modes(self):
        # legacy support
        import os
        import chimerax.shortcuts
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

    def _build_buttons(self):
        # add buttons from toolbar manager
        from PyQt5.QtGui import QPixmap, QIcon
        toolbar = self.session.toolbar._toolbar
        for tab in _layout(toolbar, "tabs"):
            if tab.startswith("_") or tab not in toolbar:
                continue
            tab_info = toolbar[tab]
            for section in _layout(tab_info, "%s sections" % tab):
                if section.startswith("_") or section not in tab_info:
                    continue
                section_info = tab_info[section]
                compact = "__compact__" in section_info
                if compact:
                    self.ttb.set_section_compact(tab, section, True)
                for display_name in _layout(section_info, "%s %s buttons" % (tab, section)):
                    if display_name.startswith("_") or display_name not in section_info:
                        continue
                    args = section_info[display_name]
                    (name, bundle_info, icon_path, description, kw) = args
                    if "hidden" in kw:
                        continue
                    if description and not description[0].isupper():
                        description = description.capitalize()
                    pm = QPixmap(icon_path)
                    icon = QIcon(pm)

                    def callback(event, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                        bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
                    # TODO: vr_mode
                    self.ttb.add_button(
                            tab, section, display_name, callback,
                            icon, description, **kw)
        self._add_mouse_modes()
        self.ttb.show_tab('Home')


def _layout(d, what):
    # Home is always first
    if "__layout__" not in d:
        keys = list(d.keys())
        try:
            home = keys.index("Home")
        except ValueError:
            keys.insert(0, "Home")
        else:
            if home != 0:
                keys = ["Home"] + keys[0:home] + keys[home + 1:]
        return keys
    import copy
    layout = copy.deepcopy(d["__layout__"])
    for k in d:
        if k == "Home":
            continue
        if k not in layout:
            layout[k] = ["Home"]
        else:
            layout[k].add("Home")
    if "Home" in layout and layout["Home"]:
        raise RuntimeError("%s: 'Home' must be first" % what)
    layout["Home"] = []
    from chimerax.core import order_dag
    ordered = []
    try:
        for n in order_dag.order_dag(layout):
            ordered.append(n)
    except order_dag.OrderDAGError as e:
        raise RuntimeError("%s: %s" % (what, e))
    return ordered


def _file_open(session):
    session.ui.main_window.file_open_cb(session)


def _file_recent(session):
    mw = session.ui.main_window
    mw.rapid_access_shown = not mw.rapid_access_shown


def _file_save(session):
    session.ui.main_window.file_save_cb(session)


# TODO: old style toolbars until mouse mode support is added
_Toolbars = {
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


_providers = {
    "Open": _file_open,
    "Recent": _file_recent,
    "Save": _file_save,
    "Close": "close session",
    "Exit": "exit",
    "Undo": "undo",
    "Redo": "redo",
    "sideview": "tool show 'Side View'"
}

def run_provider(session, name):
    what = _providers[name]
    if not isinstance(what, str):
        what(session)
    else:
        from chimerax.core.commands import run
        run(session, what)

