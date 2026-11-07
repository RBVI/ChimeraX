# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

class _MarkersAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        """Register marker mouse modes."""
        if session.ui.is_gui:
            from . import mouse
            mouse.register_mousemode(session)

    @staticmethod
    def start_tool(session, tool_name):
        from .markergui import marker_panel
        p = marker_panel(session, tool_name)
        return p

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_marker_command(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'MarkerSet':
            from .markers import MarkerSet
            return MarkerSet
        return None

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class MarkerInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from . import cmmfiles
                    return cmmfiles.read_cmm(session, data)
        else:
            from chimerax.save_command import SaverInfo
            class MarkerInfo(SaverInfo):
                def save(self, session, path, *, models=None, **kw):
                    from . import cmmfiles
                    return cmmfiles.write_cmm(session, path, models)

                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg
                    return { 'models': ModelsArg }
        return MarkerInfo()

bundle_api = _MarkersAPI()

_lazy_exports = dict.fromkeys(('MarkerSet', 'create_link', 'selected_markers', 'selected_links'), 'markers')
_lazy_exports['mark_map_center'] = 'mouse'
_lazy_modules = set(_lazy_exports.values())


def __getattr__(name):
    from importlib import import_module
    if name in _lazy_modules:
        value = import_module('.' + name, __name__)
    elif name in _lazy_exports:
        value = getattr(import_module('.' + _lazy_exports[name], __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_lazy_exports) | _lazy_modules)


__all__ = [name for name in globals() if not name.startswith('_')]
__all__ += [*_lazy_exports, *_lazy_modules]
