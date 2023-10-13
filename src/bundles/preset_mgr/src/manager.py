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

from chimerax.core.toolshed import ProviderManager
class PresetsManager(ProviderManager):
    """Manager for presets"""

    def __init__(self, session, name):
        self.session = session
        from . import settings
        settings.settings = settings._PresetsSettings(session, "presets")
        settings.settings.triggers.add_handler("setting changed", self._new_custom_folder_cb)
        self._presets = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("presets changed")
        self._new_custom_folder_cb()
        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready',
                lambda *arg, ses=session: settings.register_settings_options(session))
        super().__init__(name)

    @property
    def presets_by_category(self):
        return {cat:[name for name in info.keys()] for cat,info in self._presets.items()}

    def preset_function(self, category, preset_name):
        return self._presets[category][preset_name]

    def remove_presets(self, category, preset_names):
        for name in preset_names:
            del self._presets[category][name]
        self.triggers.activate_trigger("presets changed", self)

    def add_presets(self, category, preset_info):
        """'preset_info' should be a dictionary of preset-name -> callback-function/command-string"""
        self._add_presets(category, preset_info)
        self.triggers.activate_trigger("presets changed", self)

    def add_provider(self, bundle_info, name,
                     order=None, category="General", **kw):
        if not bundle_info.installed:
            return
        from chimerax.core.utils import CustomSortString
        if order is None:
            cname = name
        else:
            cname = CustomSortString(name, sort_val=int(order))
        def cb(name=name, mgr=self, bi=bundle_info):
            bi.run_provider(self.session, name, self)
        try:
            self._presets[category][cname] = cb
        except KeyError:
            self._presets[category] = {cname:cb}

    def end_providers(self):
        self.triggers.activate_trigger("presets changed", self)

    def execute(self, preset):
        """Presets should call this method to execute their preset so that appropriate information
        about the preset can be logged.  The 'preset' argument is either a command string or a
        callable Python function that takes no arguments.

        A command string can have embedded ';' characters to separate commands.  It can also
        have embedded newlines to separate commands, in which case the newline-separated commands
        will be executed with separate calls to chimera.core.commands.run(), whereas ';' separated
        commands will use a single run() call.
        """
        if callable(preset):
            preset()
            self.session.logger.info("Preset implemented in Python; no expansion to individual ChimeraX"
                " commands available.")
        else:
            from chimerax.core.commands import run
            num_lines = 0
            with self.session.undo.aggregate("preset"):
                for line in preset.splitlines():
                    run(self.session, line, log=False)
                    num_lines += 1
            if num_lines == 1:
                parts = [p.strip() for p in preset.split(';')]
                display_lines = '\n'.join(parts)
            else:
                display_lines = preset
            self.session.logger.info('Preset expands to these ChimeraX commands: '
                '<div style="padding-left:4em;padding-top:0px;margin-top:0px">'
                '<pre style="margin:0;padding:0">%s</pre></div>' % display_lines, is_html=True)

    def _add_presets(self, category, preset_info):
        self._presets.setdefault(category, {}).update({
            name: lambda p=preset: self.execute(p)
            for name, preset in preset_info.items()
        })

    def _gather_presets(self, folder):
        import os, os.path
        preset_info = {}
        subfolders = []
        for entry in os.listdir(folder):
            entry_path = os.path.join(folder, entry)
            if os.path.isdir(entry_path):
                subfolders.append(entry)
                continue
            if entry.endswith(".cxc"):
                f = open(entry_path, "r")
                preset_info[entry[:-4].replace('_', ' ')] = f.read()
                f.close()
            elif entry.endswith(".py"):
                from chimerax.core.commands import run, FileNameArg
                preset_info[entry[:-3].replace('_', ' ')] = lambda p=FileNameArg.unparse(entry_path), \
                    run=run, ses=self.session: run(ses, "open " + p, log=False)
        return preset_info, subfolders

    def _new_custom_folder_cb(self, *args):
        from .settings import settings
        if not settings.folder:
            return
        import os.path
        if not os.path.exists(settings.folder):
            self.session.logger.warning("Custom presets folder '%s' does not exist" % settings.folder)
            return
        presets_added = False
        preset_info, subfolders = self._gather_presets(settings.folder)
        if preset_info:
            self._add_presets("Custom", preset_info)
            presets_added = True
        for subfolder in subfolders:
            subpath = os.path.join(settings.folder, subfolder)
            preset_info, subsubfolders = self._gather_presets(subpath)
            if preset_info:
                self._add_presets(subfolder.replace('_', ' '), preset_info)
                presets_added = True
            else:
                self.session.logger.warning("No presets found in custom preset folder %s" % subpath)
        if args:
            # actual trigger callback, rather than startup call
            if presets_added:
                self.triggers.activate_trigger("presets changed", self)
        if not presets_added:
            self.session.logger.warning("No presets found in custom preset folder %s" % settings.folder)
