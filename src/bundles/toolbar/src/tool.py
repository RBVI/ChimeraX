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
        session.triggers.add_handler('set right mouse', self._set_right_mouse_button)

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

    def _build_buttons(self):
        # add buttons from toolbar manager
        from PyQt5.QtGui import QIcon
        from .manager import fake_mouse_mode_bundle_info
        self.right_mouse_buttons = {}
        self.current_right_mouse_button = None
        toolbar = self.session.toolbar._toolbar
        for tab in _layout(toolbar, "tabs"):
            if tab.startswith("__") or tab not in toolbar:
                continue
            tab_info = toolbar[tab]
            for section in _layout(tab_info, "%s sections" % tab):
                if section.startswith("__") or section not in tab_info:
                    continue
                section_info = tab_info[section]
                has_buttons = False
                for display_name in _layout(section_info, "%s %s buttons" % (tab, section)):
                    if display_name.startswith("__") or display_name not in section_info:
                        continue
                    args = section_info[display_name]
                    (name, bundle_info, icon_path, description, kw) = args
                    if "hidden" in kw:
                        continue
                    has_buttons = True
                    if description and not description[0].isupper():
                        description = description.capitalize()
                    if bundle_info == fake_mouse_mode_bundle_info:
                        kw["vr_mode"] = name  # Allows VR to recognize mouse mode buttons
                        rmbs = self.right_mouse_buttons.setdefault(name, [])
                        if icon_path is None:
                            m = self.session.ui.mouse_modes.named_mode(name)
                            if m is not None:
                                icon_path = m.icon_path
                        rmbs.append((tab, section, display_name, icon_path))
                    if icon_path is None:
                        icon = None
                    else:
                        icon = QIcon(icon_path)

                    def callback(event, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                        bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
                    # TODO: vr_mode
                    self.ttb.add_button(
                            tab, section, display_name, callback,
                            icon, description, **kw)
                if has_buttons:
                    compact = "__compact__" in section_info
                    if compact:
                        self.ttb.set_section_compact(tab, section, True)
        self.ttb.show_tab('Home')
        self._set_right_mouse_button('init', self.session.ui.mouse_modes.mode("right", exact=True))

    def _set_right_mouse_button(self, trigger_name, mode):
        # TODO: highlight current right mouse button
        return

        name = mode.name if mode is not None else None
        if name == self.current_right_mouse_button:
            return

        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QIcon, QPixmap, QColor
        has_button = name in self.right_mouse_buttons
        if self.current_right_mouse_button is not None:
            # remove highlighting
            icon = None
            for info in self.right_mouse_buttons[self.current_right_mouse_button]:
                tab_title, section_title, button_title, icon_path = info
                a = self.ttb.get_qt_button_action(tab_title, section_title, button_title)
                if a is None:
                    continue
                if icon is None and icon_path is not None:
                    # all icon_paths should be the same
                    icon = QIcon(icon_path)
                if icon is not None:
                    a.setIcon(icon)
        if not has_button:
            self.current_right_mouse_button = None
            return
        self.current_right_mouse_button = name
        # highlight button(s)
        icon = None
        for info in self.right_mouse_buttons[name]:
            tab_title, section_title, button_title, icon_path = info
            a = self.ttb.get_qt_button_action(tab_title, section_title, button_title)
            if a is None:
                continue
            if icon is None and icon_path is not None:
                # all icon_paths should be the same
                pixmap = QPixmap(icon_path)
                mask = pixmap.createMaskFromColor(QColor('transparent'), Qt.MaskOutColor)
                pixmap.fill(QColor('light green'))
                pixmap.setMask(mask)
                icon = QIcon(pixmap)
            if icon is not None:
                a.setIcon(icon)


def _layout(d, what):
    # Home is always first
    if "__layout__" not in d:
        keys = [k for k in d if not k.startswith("__")]
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
        if k == "Home" or k.startswith("__"):
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


_providers = {
    "Open": _file_open,
    "Recent": _file_recent,
    "Save": _file_save,
    "Close": "close session",
    "Exit": "exit",
    "Undo": "undo",
    "Redo": "redo",
    "Side view": "tool show 'Side View'"
}


def run_provider(session, name):
    what = _providers[name]
    if not isinstance(what, str):
        what(session)
    else:
        from chimerax.core.commands import run
        run(session, what)
