# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2019 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
from chimerax.core.settings import Settings
from copy import deepcopy
from Qt.QtCore import Qt

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
        ("Selection", [
            "ChimeraX-SelInspector:selection inspector",  # inspect selection
        ]),
    ],
}

_settings = None
_tb = None


def get_toolbar_singleton(session, create=True):
    global _tb
    if create and _tb is None:
        _tb = ToolbarTool(session, "Toolbar")
    return _tb


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
    help = "help:user/tools/toolbar.html"  # Let ChimeraX know about our help page

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
        session.triggers.add_handler('set mouse mode', self._set_right_mouse_button)
        session.ui.triggers.add_handler('color scheme changed', self._color_scheme_changed)

    def _build_ui(self):
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        from Qt.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        margins = layout.contentsMargins()
        margins.setTop(0)
        margins.setBottom(0)
        layout.setContentsMargins(margins)
        self.ttb = TabbedToolbar(
            self.tool_window.ui_area, show_section_titles=_settings.show_section_labels,
            show_button_titles=_settings.show_button_labels, lazy_tabs=True)
        layout.addWidget(self.ttb)
        self._build_tabs()
        self.tool_window.ui_area.setLayout(layout)
        self.tool_window.manage(self.PLACEMENT, allowed_areas=Qt.AllDockWidgetAreas)

    def _color_scheme_changed(self, *_):
        from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar
        self.ttb.clear_all()
        if 1:
            # don't know why, but need to use new toolbar
            layout = self.tool_window.ui_area.layout()
            layout.removeWidget(self.ttb)
            self.ttb.setParent(None)
            self.ttb = TabbedToolbar(
                self.tool_window.ui_area, show_section_titles=_settings.show_section_labels,
                show_button_titles=_settings.show_button_labels, lazy_tabs=True)
            layout.addWidget(self.ttb)
        self._build_tabs()

    def fill_context_menu(self, menu, x, y):
        # avoid having actions destroyed when this routine returns
        # by stowing a reference in the menu itself
        from Qt.QtGui import QAction
        button_labels = QAction("Show button labels", menu)
        button_labels.setCheckable(True)
        button_labels.setChecked(_settings.show_button_labels)
        button_labels.toggled.connect(lambda arg, *, f=self._set_button_labels: f(arg))
        menu.addAction(button_labels)
        section_labels = QAction("Show section labels", menu)
        section_labels.setCheckable(True)
        section_labels.setChecked(_settings.show_section_labels)
        section_labels.toggled.connect(lambda arg, *, f=self._set_section_labels: f(arg))
        menu.addAction(section_labels)
        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(lambda arg: self.show_settings())
        menu.addAction(settings_action)

    def show_settings(self):
        if not hasattr(self, "settings_tool"):
            from .settings_ui import ToolbarSettingsTool
            self.settings_tool = ToolbarSettingsTool(
                self.session, self,
                self.tool_window.create_child_window("Toolbar Settings", close_destroys=False))
            self.settings_tool.tool_window.manage(None)
        self.settings_tool.tool_window.shown = True

    def _set_button_labels(self, show_button_labels):
        _settings.show_button_labels = show_button_labels
        self.ttb.set_show_button_titles(show_button_labels)

    def _set_section_labels(self, show_section_labels):
        _settings.show_section_labels = show_section_labels
        self.ttb.set_show_section_titles(show_section_labels)

    def build_home_tab(self):
        # (re)Build Home tab from settings
        from Qt.QtGui import QIcon
        self.ttb.clear_tab("Home")
        last_section = None
        for (section, compact, display_name, icon_info, description, link, bundle_info, name, kw) in _home_layout(self.session, _settings.home_tab):
            theme_icon, icon_path, dark_icon_path = icon_info
            if section != last_section:
                last_section = section
                if compact:
                    self.ttb.set_section_compact("Home", section, True)
            if theme_icon is not None:
                icon = QIcon.fromTheme(theme_icon)
            elif dark_icon_path is not None and self.session.ui.dark_mode():
                icon = QIcon(dark_icon_path)
            elif icon_path is not None:
                icon = QIcon(icon_path)
            else:
                icon = None

            def callback(*, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
            self.ttb.add_button(
                    "Home", section, display_name, callback,
                    icon, description, **kw)

    def _build_tabs(self, *_):
        # add buttons from toolbar manager
        from Qt.QtGui import QIcon
        from .manager import fake_mouse_mode_bundle_info
        self.right_mouse_buttons = {}
        self.current_right_mouse_button = None

        self.build_home_tab()

        # Build other tabs from toolbar manager
        toolbar = self.session.toolbar._toolbar
        last_tab = None
        last_section = None
        for tab, section, compact, display_name, icon_info, description, bundle_info, name, kw in _other_layout(self.session, toolbar):
            theme_icon, icon_path, dark_icon_path = icon_info
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
            if theme_icon is not None:
                icon = QIcon.fromTheme(theme_icon)
            elif dark_icon_path is not None and self.session.ui.dark_mode():
                icon = QIcon(dark_icon_path)
            elif icon_path is not None:
                icon = QIcon(icon_path)
            else:
                icon = None

            def callback(*, session=self.session, name=name, bundle_info=bundle_info, display_name=display_name):
                bundle_info.run_provider(session, name, session.toolbar, display_name=display_name)
            self.ttb.add_button(
                    tab, section, display_name, callback,
                    icon, description, **kw)
        self.ttb.show_tab('Home')
        self._set_right_mouse_button('init', self.session.ui.mouse_modes.mode("right", exact=True))

    def _set_right_mouse_button(self, trigger_name, data):
        # highlight current right mouse button
        if trigger_name == 'init':
            mode = data
        else:
            button, modifiers, mode = data
            if button != "right" or modifiers:
                return
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

    def set_enabled(self, enabled, tab_title, section_title, button_title):
        self.ttb.set_enabled(enabled, tab_title, section_title, button_title)

    def show_group_button(self, tab_title, section_title, button_title):
        self.ttb.show_group_button(tab_title, section_title, button_title)


def _home_layout(session, home_tab):
    # interact through buttons in home tab
    # All buttons were vetted, so silently skip missing ones
    from Qt.QtGui import QIcon
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
            pi = bi.providers.get('toolbar/' + name, None)
            if not pi:
                continue
            pi_kw = pi
            if display_name is None:
                display_name = pi_kw.get("display_name", None)
                if display_name is None:
                    display_name = name
            theme_icon = pi_kw.get('themeIcon', None)
            icon_path = pi_kw.get("icon", None)
            dark_icon_path = pi_kw.get("darkIcon", None)
            description = pi_kw.get("description", None)
            if description and not description[0].isupper():
                description = description.capitalize()
            if theme_icon is not None:
                theme_icon = getattr(QIcon.ThemeIcon, theme_icon, None)
            if icon_path is not None:
                icon_path = bi.get_path('icons/%s' % icon_path)
            if dark_icon_path is not None:
                dark_icon_path = bi.get_path('icons/%s' % dark_icon_path)
            yield (section_title, compact, display_name, (theme_icon, icon_path, dark_icon_path), description, link, bi, name, kw)


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
                (name, bundle_info, icon_info, description, kw) = args
                if hide_hidden and "hidden" in kw:
                    continue
                if description and not description[0].isupper():
                    description = description.capitalize()
                compact = "__compact__" in section_info
                yield (tab, section, compact, display_name, icon_info, description, bundle_info, name, kw)


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
    from chimerax.core import is_daily_build
    if is_daily_build():
        import sys
        for key, values in layout.items():
            for value in values:
                if not isinstance(value, str):
                    continue
                if value not in layout:
                    print(f"developer warning: toolbar '{key}' depends on non-existent '{value}'", file=sys.__stderr__)
    from chimerax.core import order_dag
    ordered = []
    try:
        for n in order_dag.order_dag(layout):
            ordered.append(n)
    except order_dag.OrderDAGError as e:
        raise RuntimeError("%s: %s" % (what, e))
    return ordered


# Preserve imports from toolbar.tool while keeping editor widgets off startup.
_settings_ui_exports = (
    'ToolbarSettingsTool', 'QHLine', '_HomeTab', 'LINK_ROLE', 'ITEM_TYPE_ROLE',
    'TAB_TYPE', 'SECTION_TYPE', 'BUTTON_TYPE', 'GROUP_TYPE', 'BUTTON_FLAGS',
    'SECTION_FLAGS', 'QFrame', 'QTreeWidget', 'QTreeWidgetItem', 'Signal',
)


def __getattr__(name):
    if name not in _settings_ui_exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import settings_ui
    value = getattr(settings_ui, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_settings_ui_exports))


__all__ = [name for name in globals() if not name.startswith('_')]
__all__ += [name for name in _settings_ui_exports if not name.startswith('_')]
