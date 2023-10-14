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

from chimerax.core import toolshed
_new_bundle_handler = None

help_url_paths = []     # help directories in URL path form


def _update_cache(trigger_name=None, bundle_info=None):
    global help_url_paths

    import os
    from chimerax import app_dirs
    cached_index = os.path.join(app_dirs.user_cache_dir, 'docs', 'user', 'index.html')
    try:
        os.remove(cached_index)
    except OSError:
        pass

    def cvt_path(path):
        from urllib.request import pathname2url
        help_path = pathname2url(path)
        if help_path.startswith('///'):
            help_path = help_path[2:]
        if not help_path.endswith('/'):
            help_path += '/'
        return help_path

    help_directories = toolshed.get_help_directories()
    help_url_paths = [cvt_path(hd) for hd in help_directories]


class _MyAPI(toolshed.BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        global _new_bundle_handler
        ts = toolshed.get_toolshed()
        _new_bundle_handler = ts.triggers.add_handler(
            toolshed.TOOLSHED_BUNDLE_INSTALLED, _update_cache)
        # ? = ts.triggers.add_handler(
        #    toolshed.TOOLSHED_BUNDLE_UNINSTALLED, _update_cache)
        _update_cache()

    @staticmethod
    def finish(session, bundle_info):
        global _new_bundle_handler
        ts = toolshed.get_toolshed()
        if _new_bundle_handler is not None:
            ts.triggers.remove_handler(_new_bundle_handler)
            _new_bundle_handler = None

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when command is referenced
        from . import cmd
        from chimerax.core.commands import register
        register(command_name, cmd.help_desc, cmd.help, logger=logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'HelpUI':
            from . import tool
            return tool.HelpUI
        return None

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if name == "HTML":
            from chimerax.open_command import OpenerInfo

            class HelpViewerInfo(OpenerInfo):

                def open(self, session, path, file_name, *, new_tab=False):
                    import os
                    base, ext = os.path.splitext(path)
                    ext, *fragment = ext.split('#')
                    if not fragment:
                        fragment = ''
                    else:
                        fragment = fragment[0]
                        path = path[:-(len(fragment) + 1)]
                    path = os.path.abspath(path)
                    from urllib.parse import urlunparse
                    from urllib.request import pathname2url
                    url = urlunparse(('file', '', pathname2url(path), '', '', fragment))
                    show_url(session, url, new_tab=new_tab)
                    return [], "Opened %s" % file_name

                @property
                def open_args(self):
                    from chimerax.core.commands import BoolArg
                    return {'new_tab': BoolArg}

                in_file_history = False
        else:  # help: / http: / https:
            from chimerax.open_command import FetcherInfo

            class HelpViewerInfo(FetcherInfo):

                def fetch(self, session, ident, format_name, ignore_cache,
                          _protocol=name, **kw):
                    url = _protocol + ':' + ident
                    show_url(session, url, **kw)
                    return [], "Opened %s" % url

                @property
                def fetch_args(self):
                    from chimerax.core.commands import BoolArg
                    return {'new_tab': BoolArg}

                in_file_history = False

        return HelpViewerInfo()


def show_url(session, url, *, new_tab=False, html=None):
    if session.ui.is_gui:
        from .tool import HelpUI
        help_viewer = HelpUI.get_viewer(session)
        help_viewer.show(url, new_tab=new_tab, html=html)
    else:
        import webbrowser
        if new_tab:
            webbrowser.open_new_tab(url)
        else:
            webbrowser.open(url)


def reload_toolshed_tabs(session):
    if not session.ui.is_gui:
        return
    from .tool import HelpUI
    help_viewer = HelpUI.get_viewer(session, create=False)
    if not help_viewer:
        return
    help_viewer.reload_toolshed_tabs()


bundle_api = _MyAPI()
