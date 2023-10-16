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

from chimerax.core.toolshed import BundleAPI

class _SessionAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            if name == "ChimeraX session":
                class Info(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from chimerax.core.session import open as cxs_open
                        return cxs_open(session, data, **kw)

                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg
                        return { 'resize_window': BoolArg, 'combine': BoolArg }

            elif name == "ChimeraX commands":
                class Info(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from chimerax.core.scripting import open_command_script
                        return open_command_script(session, data, file_name, **kw)

                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg, StringArg, RepeatOf
                        return {
                            'log': BoolArg,
                            'for_each_file': RepeatOf(StringArg),
                        }

            elif name == "Python":
                class Info(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from chimerax.core.scripting import open_python_script
                        return open_python_script(session, data, file_name)

            elif name == "Compiled Python":
                class Info(OpenerInfo):
                    def open(self, session, data, file_name, **kw):
                        from chimerax.core.scripting import open_compiled_python_script
                        return open_compiled_python_script(session, data, file_name)

            else: # web fetch
                from chimerax.open_command import FetcherInfo
                class Info(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache,
                            _protocol=name, **kw):
                        from .web_fetch import fetch_web
                        return fetch_web(session, _protocol + ':' + ident,
                            ignore_cache=ignore_cache, **kw)

                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import BoolArg, EnumOf
                        return {
                            'new_tab': BoolArg,
                            'data_format': EnumOf([fmt.nicknames[0]
                                for fmt in session.open_command.open_data_formats]),
                        }
                    # let what gets fetched handle file history insertion
                    in_file_history = False
        else:
            from chimerax.save_command import SaverInfo
            if name == "ChimeraX session":
                class Info(SaverInfo):
                    def save(self, session, path, **kw):
                        from chimerax.core.session import save as cxs_save
                        return cxs_save(session, path, **kw)

                    @property
                    def save_args(self):
                        from chimerax.core.commands import BoolArg, IntArg, EnumOf
                        return {
                            'include_maps': BoolArg,
                            'compress': EnumOf(('lz4', 'gzip', 'none')),
                            'version': IntArg,
                        }

                    @property
                    def hidden_args(self):
                        return ['version']

                    def save_args_widget(self, session):
                        from .gui import SaveOptionsWidget
                        return SaveOptionsWidget(session)

                    def save_args_string_from_widget(self, widget):
                        return widget.options_string()
            else: # X3D
                class Info(SaverInfo):
                    def save(self, session, path, **kw):
                        from chimerax.core.session import save_x3d
                        return save_x3d(session, path, **kw)

                    @property
                    def save_args(self):
                        from chimerax.core.commands import BoolArg
                        return {
                            'transparent_background': BoolArg,
                        }


        return Info()

bundle_api = _SessionAPI()
