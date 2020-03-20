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

from chimerax.core.toolshed import BundleAPI

class _SessionAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open import OpenerInfo
            class Info(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from chimerax.core.session import open as cxs_open
                    return cxs_open(session, data, **kw)

                @property
                def open_args(self):
                    from chimerax.core.commands import BoolArg
                    return { 'resize_window': BoolArg }
        else:
            from chimerax.save import SaverInfo
            if name == "session":
                class Info(SaverInfo):
                    def save(self, session, path, **kw):
                        from chimerax.core.session import save as cxs_save
                        return cxs_save(session, path, **kw)

                    @property
                    def save_args(self):
                        from chimerax.core.commands import BoolArg, IntArg
                        return {
                            'include_maps': BoolArg,
                            'uncompressed': BoolArg,
                            'version': IntArg,
                        }

                    @property
                    def hidden_args(self):
                        return ['uncompressed', 'version']

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
