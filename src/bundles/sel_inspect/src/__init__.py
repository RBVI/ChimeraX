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

class _PresetsBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "SelectionInspector":
            from . import manager
            return manager.SelectionInspector

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        if name == "selection inspector":
            from .manager import SelectionInspection
            session.selection_inspection = SelectionInspection(session)
            return session.selection_inspection
        raise ValueError("No manager named '%s' in %s module" % (name, __module__))

    @staticmethod
    def run_provider(session, bundle_info, name, mgr, **kw):
        #TODO
        pass
        #from .builtin import run_preset
        #run_preset(session, name, mgr, **kw)

    @staticmethod
    def finish(session, bundle_info):
        del session.selection_inspector

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import SelectionInspector
        return SelectionInspector(session, tool_name)

bundle_api = _PresetsBundleAPI()
