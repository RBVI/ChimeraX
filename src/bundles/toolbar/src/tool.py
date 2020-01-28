# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2019 Regents of the University of California.
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
from copy import deepcopy
from PyQt5.QtWidgets import QFrame

defaults = {
    "home_tab": [
        # All buttons are links to existing buttons
        ("File", [
            "ChimeraX-Toolbar:Open",  # open
            "ChimeraX-Toolbar:Recent",  # recent
            "ChimeraX-Toolbar:Save",  # save
        ]),
        ("Images", [
            "ChimeraX-Shortcuts:sx",  # snapshot
            "ChimeraX-Shortcuts:vd",  # spin movie
        ]),
        ("Atoms", [
            "ChimeraX-Shortcuts:da",  # show
            "ChimeraX-Shortcuts:ha",  # hide
        ]),
        ("Cartoons", [
            "ChimeraX-Shortcuts:rb",  # show
            "ChimeraX-Shortcuts:hr",  # hide
        ]),
        ("Styles", [
            "ChimeraX-Shortcuts:st",  # stick
            "ChimeraX-Shortcuts:sp",  # sphere
            "ChimeraX-Shortcuts:bs",  # ball
        ]),
        ("Background", [
            "ChimeraX-Shortcuts:wb",  # white
            "ChimeraX-Shortcuts:bk",  # black
        ]),
        ("Lighting", [
            "ChimeraX-Shortcuts:ls",  # simple
            "ChimeraX-Shortcuts:la",  # soft
            "ChimeraX-Shortcuts:lf",  # full
        ]),
    ],
}

_settings = None


class _ToolbarSettings(Settings):
    AUTO_SAVE = {
        "show_button_labels": True,
        "show_section_labels": True,
    }
    EXPLICIT_SAVE = deepcopy(defaults)


class ToolbarTool(ToolInstance):

    SESSION_ENDURING = True
    SESSION_SAVE = False
    PLACEMENT = "top"
    CUSTOM_SCHEME = "toolbar"
    help = "help:user/tools/Toolbar.html"  # Let ChimeraX know about our help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "Toolbar"
        global _settings
        if _settings is None:
            _settings = _ToolbarSettings(self.session, "Toolbar")
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
            self.tool_window.ui_area, show_section_titles=_settings.show_section_labels,
            show_button_titles=_settings.show_button_labels)
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
        button_labels.setChecked(_settings.show_button_labels)
        button_labels.toggled.connect(lambda arg, f=self._set_button_labels: f(arg))
        menu.addAction(button_labels)
        section_labels = QAction("Show section labels", menu)
        section_labels.setCheckable(True)
        section_labels.setChecked(_settings.show_section_labels)
        section_labels.toggled.connect(lambda arg, f=self._set_section_labels: f(arg))
        menu.addAction(section_labels)

    def _set_button_labels(self, show_button_labels):
        _settings.show_button_labels = show_button_labels
        self.ttb.set_show_button_titles(show_button_labels)

    def _set_section_labels(self, show_section_labels):
        _settings.show_section_labels = show_section_labels
        self.ttb.set_show_section_titles(show_section_labels)

    def _build_buttons(self):
        # add buttons from toolbar manager
        from PyQt5.QtGui import QIcon
        from .manager import fake_mouse_mode_bundle_info
        self.right_mouse_buttons = {}
        self.current_right_mouse_button = None

        # Build Home tab from settings
        last_section = None
        for (section, compact, display_name, icon_path, description, link, bundle_info, name, kw) in _home_layout(self.session, _settings.home_tab):
            if section != last_section:
                last_section = section
                if compact:
                    self.ttb.set_section_compact("Home", section, True)
            if icon_path is None:
                icon = None
            else:
                icon = QIcon(icon_path)

            def callback(event, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
            self.ttb.add_button(
                    "Home", section, display_name, callback,
                    icon, description, **kw)

        # Build other tabs from toolbar manager
        toolbar = self.session.toolbar._toolbar
        last_tab = None
        last_section = None
        for (tab, section, compact, display_name, icon_path, description, bundle_info, name, kw) in _other_layout(self.session, toolbar):
            if tab != last_tab:
                last_tab = tab
                last_section = None
            if section != last_section:
                last_section = section
                if compact:
                    self.ttb.set_section_compact(tab, section, True)
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
            self.ttb.add_button(
                    tab, section, display_name, callback,
                    icon, description, **kw)
        self.ttb.show_tab('Home')
        self._set_right_mouse_button('init', self.session.ui.mouse_modes.mode("right", exact=True))

    def _set_right_mouse_button(self, trigger_name, mode):
        # highlight current right mouse button
        name = mode.name if mode is not None else None
        if name == self.current_right_mouse_button:
            return

        set_sections = set()
        has_button = name in self.right_mouse_buttons
        if has_button:
            for info in self.right_mouse_buttons[name]:
                tab_title, section_title, _, _ = info
                set_sections.add((tab_title, section_title))

        if self.current_right_mouse_button is not None:
            # remove highlighting
            for info in self.right_mouse_buttons[self.current_right_mouse_button]:
                tab_title, section_title, button_title, icon_path = info
                redo = (tab_title, section_title) not in set_sections
                self.ttb.remove_button_highlight(tab_title, section_title, button_title, redo=redo)
        if not has_button:
            return
        # highlight button(s)
        self.current_right_mouse_button = name
        for info in self.right_mouse_buttons[name]:
            tab_title, section_title, button_title, icon_path = info
            self.ttb.add_button_highlight(tab_title, section_title, button_title)


def _home_layout(session, home_tab):
    # interact through buttons in home tab
    # All buttons were vetted, so silently skip missing ones
    for section_title, buttons in _settings.home_tab:
        compact = False
        if type(section_title) is tuple:
            section_title, compact = section_title
        for link in buttons:
            kw = {}
            display_name = None
            if type(link) is tuple:
                link, display_name = link
            try:
                bundle_name, name = link.split(':', maxsplit=1)
            except ValueError:
                continue
            bi = session.toolshed.find_bundle(bundle_name, session.logger, installed=True)
            if not bi:
                continue
            pi = bi.providers.get(name, None)
            if not pi:
                continue
            pi_manager, pi_kw = pi
            if display_name is None:
                display_name = pi_kw.get("display_name", None)
                if display_name is None:
                    display_name = name
            try:
                icon_path = pi_kw["icon"]
                description = pi_kw["description"]
            except KeyError:
                continue
            if description and not description[0].isupper():
                description = description.capitalize()
            if icon_path is not None:
                icon_path = bi.get_path('icons/%s' % icon_path)
            yield (section_title, compact, display_name, icon_path, description, link, bi, name, kw)


def _other_layout(session, toolbar, hide_hidden=True):
    for tab in _layout(toolbar, "tabs"):
        if tab.startswith("__") or tab not in toolbar:
            continue
        tab_info = toolbar[tab]
        for section in _layout(tab_info, "%s sections" % tab):
            if section.startswith("__") or section not in tab_info:
                continue
            section_info = tab_info[section]
            for display_name in _layout(section_info, "%s %s buttons" % (tab, section)):
                if display_name.startswith("__") or display_name not in section_info:
                    continue
                args = section_info[display_name]
                (name, bundle_info, icon_path, description, kw) = args
                if hide_hidden and "hidden" in kw:
                    continue
                if description and not description[0].isupper():
                    description = description.capitalize()
                compact = "__compact__" in section_info
                yield (tab, section, compact, display_name, icon_path, description, bundle_info, name, kw)


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


class CustomizeTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False        # No session saving for now
    PLACEMENT = "top"
    CUSTOM_SCHEME = "toolbar"
    help = "help:user/tools/Toolbar.html#customize"  # Let ChimeraX know about our help page

    TAB_TYPE = 1
    SECTION_TYPE = 2
    BUTTON_TYPE = 3
    GROUP_TYPE = 4

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        tw.title = "Customize Toolbar Home Tab"
        self._build_ui()
        tw.manage(placement=None)

    def _build_ui(self):
        from PyQt5.QtWidgets import (
            QWidget, QLabel, QPushButton,
            QTreeWidget, QTreeWidgetItem,
            QGridLayout, QHBoxLayout,
        )
        from PyQt5.QtGui import QIcon
        from PyQt5.QtCore import Qt
        # widget layout:
        parent = self.tool_window.ui_area
        layout = QGridLayout()
        parent.setLayout(layout)
        self.instructions = QLabel(parent)
        layout.addWidget(self.instructions, 1, 1, 1, 2)
        self.home = QTreeWidget(parent)
        layout.addWidget(self.home, 2, 1)
        self.home.setColumnCount(1)
        self.other = QTreeWidget(parent)
        layout.addWidget(self.other, 2, 2)
        self.other.setColumnCount(1)
        line = QHLine(parent)
        layout.addWidget(line, 3, 1, 1, 2)
        bottom = QWidget(parent)
        layout.addWidget(bottom, 4, 1, 1, 2)
        layout = QHBoxLayout()
        bottom.setLayout(layout)
        # TODO: right-justify bottom buttons
        save = QPushButton("Save", bottom)
        save.setToolTip("Save current Home tab configuration")
        layout.addWidget(save)
        revert = QPushButton("Revert", bottom)
        revert.setToolTip("Revert to previously saved Home tab configuration")
        layout.addWidget(revert)
        reset = QPushButton("Reset", bottom)
        reset.setToolTip("Reset Home tab to default configuration")
        layout.addWidget(reset)
        close = QPushButton("Close", bottom)
        layout.addWidget(close)

        # widget contents/customization:
        self.instructions.setText("""
        <h1>Customize Toolbar Home Tab</h1>

        Instructional text on how to use inteface.
        """)

        # the following is very similar to code for toolbar layout
        self.home.setHeaderLabels(["Home Tab"])
        last_section = None
        section_item = None
        for (section, compact, display_name, icon_path, description, link, bi, name, kw) in _home_layout(self.session, _settings.home_tab):
            if section != last_section:
                last_section = section
                section_item = QTreeWidgetItem(self.home, [section], self.SECTION_TYPE)
                section_item.setFlags(Qt.ItemIsDropEnabled | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.home.expandItem(section_item)
            item = QTreeWidgetItem(section_item, [f"{display_name} ({link})"], self.BUTTON_TYPE)
            item.setFlags(Qt.ItemIsDropEnabled | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            if 0:
                # TODO
                if icon_path is None:
                    icon = None
                else:
                    icon = QIcon(icon_path)
                item.setIcon(1, icon)
            item.setToolTip(1, description)

        self.other.setHeaderLabels([""])
        toolbar = self.session.toolbar._toolbar
        last_tab = None
        last_section = None
        tab_item = None
        section_item = None
        for (tab, section, compact, display_name, icon_path, description, bundle_info, name, kw) in _other_layout(self.session, toolbar):
            if tab != last_tab:
                last_tab = tab
                last_section = None
                tab_item = QTreeWidgetItem(self.other, [tab], self.TAB_TYPE)
                tab_item.setFlags(Qt.ItemIsEnabled)
                self.other.expandItem(tab_item)
            if section != last_section:
                last_section = section
                section_item = QTreeWidgetItem(tab_item, [section], self.SECTION_TYPE)
                section_item.setFlags(Qt.ItemIsDropEnabled | Qt.ItemIsEnabled)
                self.other.expandItem(section_item)
            item = QTreeWidgetItem(section_item, [f"{display_name}"], self.BUTTON_TYPE)
            item.setFlags(Qt.ItemIsDragEnabled | Qt.ItemIsEnabled)
            item.setToolTip(1, description)

# Adapted QHLine from
# https://stackoverflow.com/questions/5671354/how-to-programmatically-make-a-horizontal-line-in-qt


class QHLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
