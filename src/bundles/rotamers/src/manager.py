# vim: set expandtab shiftwidth=4 softtabstop=4:

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
        self._ui_names = {}
        self._descriptions = {}
        self.settings = _RotamerManagerSettings(session, "rotamer lib manager")
        self._uninstalled_suffix = " [not installed]"
        super().__init__(name)

    def library(self, name):
        lib_name = self._name_to_lib_name(name)
        lib_info = self._library_info[lib_name]
        from . import RotamerLibrary
        if not isinstance(lib_info, RotamerLibrary):
            self._library_info[lib_name] = lib_info = lib_info.run_provider(self.session, lib_name, self)
        return lib_info

    def library_names(self, *, installed_only=False, for_display=False):
        if not installed_only:
            if for_display:
                return list(self._ui_names.values())
            return list(self._library_info.keys())
        from . import RotamerLibrary
        lib_names = []
        for name, info in self._library_info.items():
            if isinstance(info, RotamerLibrary) or info.installed:
                if for_display:
                    lib_names.append(self._ui_names[name])
                else:
                    lib_names.append(name)
        return lib_names

    def library_name_menu(self, *, initial_lib=None, installed_only=False, callback=None):
        from Qt.QtWidgets import QPushButton, QMenu
        class RotLibMenuButton(QPushButton):
            def __init__(self, session, *args, **kw):
                self.session = session
                super().__init__(*args, **kw)

            @property
            def lib_name(self):
                return self.session.rotamers._menu_text_to_lib_name(self.text())

        menu_button = RotLibMenuButton(self.session)
        if initial_lib is None:
            lib_name = self.settings.gui_lib_name
        else:
            lib_name = initial_lib
        if lib_name not in self.library_names(installed_only=installed_only, for_display=True):
            lib_name = self._ui_names[self.default_command_library_name]
        menu_button.setText(lib_name)
        menu = QMenu(menu_button)
        menu_button.setMenu(menu)
        menu.aboutToShow.connect(lambda *, menu=menu, installed=installed_only:
            self._menu_show_cb(menu, installed))
        menu.triggered.connect(lambda action, *, button=menu_button, cb=callback:
            self._menu_choose_cb(action, button, cb))
        return menu_button

    def library_name_option(self, *, installed_only=False):
        from chimerax.ui.options import Option
        class RotLibOption(Option):
            def _make_widget(self, *, mgr=self, installed_only=installed_only, **kw):
                self.widget = mgr.library_name_menu(initial_lib=self.default, installed_only=installed_only,
                    callback=self.make_callback)

            def get_value(self):
                return self.widget.lib_name

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

    def description(self, provider_name):
        return self._descriptions[provider_name]

    def ui_name(self, provider_name):
        return self._ui_names[provider_name]

    def add_provider(self, bundle_info, name, *, ui_name=None, description=None):
        self._library_info[name] = bundle_info
        self._ui_names[name] = name if ui_name is None else ui_name
        self._descriptions[name] = name if description is None else description

    def end_providers(self):
        self.triggers.activate_trigger("rotamer libs changed", self)

    def _menu_choose_cb(self, action, button, callback):
        ui_lib_name = self.ui_name(self._menu_text_to_lib_name(action.text()))
        button.setText(ui_lib_name)
        self.settings.gui_lib_name = ui_lib_name
        if callback:
            callback()

    def _menu_show_cb(self, menu, installed_only):
        menu.clear()
        names = self.library_names(installed_only=installed_only, for_display=True)
        if not names:
            raise LimitationError("No rotamer libraries %s!"
                % ("installed" if installed_only else "available"))
        names.sort()
        installed = set(self.library_names(installed_only=True, for_display=True))
        for name in names:
            if name in installed:
                menu.addAction(name)
            else:
                menu.addAction(name + self._uninstalled_suffix)

    def _menu_text_to_lib_name(self, menu_text):
        if menu_text.endswith(self._uninstalled_suffix):
            ui_name = menu_text[:-len(self._uninstalled_suffix)]
        else:
            ui_name = menu_text
        return self._name_to_lib_name(ui_name)

    def _name_to_lib_name(self, name):
        if name in self._library_info:
            return name
        # might be display name, check...
        for lib_name, ui_name in self._ui_names.items():
            if name == ui_name:
                return lib_name
        raise NoRotamerLibraryError("No rotamer library named %s" % name)
