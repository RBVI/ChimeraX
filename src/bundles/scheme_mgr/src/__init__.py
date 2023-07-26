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


class _SchemesBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "SchemesManager":
            from . import manager
            return manager.SchemesManager

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Initialize schemes manager"""
        if name == "url_schemes":
            from .manager import SchemesManager
            session.url_schemes = SchemesManager(session, name)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        """Run scheme provider (which does nothing)"""
        return


bundle_api = _SchemesBundleAPI()
