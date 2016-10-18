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

class _AlignmentsBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "AlignmentsManager":
            from . import manager
            return manager.AlignmentsManager

    @staticmethod
    def initialize(session, bundle_info):
        """Install alignments manager into existing session"""
        from . import settings
        settings.init(session)

        from .manager import AlignmentsManager
        session.alignments = AlignmentsManager(session, bundle_info)

    @staticmethod
    def finish(session, bundle_info):
        """De-install alignments manager from existing session"""
        del session.alignments

    @staticmethod
    def open_file(*args, **kw):
        from .parse import open_file
        return open_file(*args, **kw)

bundle_api = _AlignmentsBundleAPI()
