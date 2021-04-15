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

class _InspectionBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "ItemsInspection":
            from . import manager
            return manager.ItemsInspection

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        if name == "items inspection":
            from .manager import ItemsInspection
            session.items_inspection = ItemsInspection(session)
            return session.items_inspection
        raise ValueError("No manager named '%s' in %s module" % (name, __module__))

    @staticmethod
    def finish(session, bundle_info):
        del session.items_inspection

bundle_api = _InspectionBundleAPI()
