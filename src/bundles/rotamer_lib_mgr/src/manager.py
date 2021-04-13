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

from chimerax.core.errors import LimitationError

class NoRotamerLibraryError(ValueError):
    pass

from chimerax.core.settings import Settings
class _RotamerManagerSettings(Settings):
    AUTO_SAVE = {
        'gui_lib_name': "Dunbrack"
    }

from chimerax.core.toolshed import ProviderManager
class RotamerLibManager(ProviderManager):
    """Manager for rotmer libraries"""

    def __init__(self, session, name):
        self.session = session
        self.rot_libs = None
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("rotamer libs changed")
        self._library_info = {}
        self.settings = _RotamerManagerSettings(session, "rotamer lib manager")
        self._uninstalled_suffix = " [not installed]"
        super().__init__(name)

    def library(self, name):
        try:
            lib_info = self._library_info[name]
        except KeyError:
            raise NoRotamerLibraryError("No rotamer library named %s" % name)
        from . import RotamerLibrary
        if not isinstance(lib_info, RotamerLibrary):
            self._library_info[name] = lib_info = lib_info.run_provider(self.session, name, self)
        return lib_info

    def library_names(self, *, installed_only=False):
        if not installed_only:
            return list(self._library_info.keys())
        from . import RotamerLibrary
        lib_names = []
        for name, info in self._library_info.items():
            if isinstance(info, RotamerLibrary) or info.installed:
                lib_names.append(name)
        return lib_names

    def library_name_menu(self, *, initial_lib=None, installed_only=False, callback=None):
        from Qt.QtWidgets import QPushButton, QMenu
        menu_button = QPushButton()
        if initial_lib is None:
            lib_name = self.settings.gui_lib_name
        else:
            lib_name = initial_lib
        if lib_name not in self.library_names(installed_only=installed_only):
            lib_name = self.default_command_library_name
        menu_button.setText(lib_name)
        menu = QMenu()
        menu_button.setMenu(menu)
        menu.aboutToShow.connect(lambda *, menu=menu, installed=installed_only:
            self._menu_show_cb(menu, installed))
        menu.triggered.connect(lambda action, button=menu_button, cb=callback:
            self._menu_choose_cb(action, button, cb))
        return menu_button

    def library_name_option(self, *, installed_only=False):
        from chimerax.ui.options import Option
        class RotLibOption(Option):
            def _make_widget(self, *, mgr=self, installed_only=installed_only, **kw):
                self.widget = mgr.library_name_menu(initial_lib=self.default, installed_only=installed_only,
                    callback=self.make_callback)

            def get_value(self):
                return self.widget.text()

            def set_value(self, val):
                self.widget.setText(val)

            value = property(get_value, set_value)

            def set_multiple(self):
                self.widget.setText(self.multiple_value)

        return RotLibOption

    @property
    def default_command_library_name(self):
        available_libs = self.library_names()
        for lib_name in available_libs:
            if "Dunbrack" in lib_name:
                lib = lib_name
                break
        else:
            if available_libs:
                lib = list(available_libs)[0]
            else:
                raise LimitationError("No rotamer libraries installed")
        return lib

    def add_provider(self, bundle_info, name, **kw):
        self._library_info[name] = bundle_info

    def end_providers(self):
        self.triggers.activate_trigger("rotamer libs changed", self)

    def _menu_choose_cb(self, action, button, callback):
        menu_text = action.text()
        if menu_text.endswith(self._uninstalled_suffix):
            lib_name = menu_text[:-len(self._uninstalled_suffix)]
        else:
            lib_name = menu_text
        button.setText(lib_name)
        self.settings.gui_lib_name = lib_name
        if callback:
            callback()

    def _menu_show_cb(self, menu, installed_only):
        menu.clear()
        names = self.library_names(installed_only=installed_only)
        if not names:
            raise LimitationError("No rotamer libraries %s!"
                % ("installed" if installed_only else "available"))
        names.sort()
        installed = set(self.library_names(installed_only=True))
        for name in names:
            if name in installed:
                menu.addAction(name)
            else:
                menu.addAction(name + self._uninstalled_suffix)

