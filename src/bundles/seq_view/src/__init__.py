# vim: set expandtab ts=4 sw=4:

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

class _SeqViewerBundleAPI(BundleAPI):

    @staticmethod
    def finish(session, bundle_info):
        """De-register sequence viewer from alignments manager"""
        session.alignments.deregister_viewer(bundle_info.tools[0].name)

    @staticmethod
    def get_class(class_name):
        if class_name == "SequenceViewer":
            from .tool import SequenceViewer
            return SequenceViewer

    @staticmethod
    def initialize(session, bundle_info):
        """Register sequence viewer with alignments manager"""
        session.alignments.register_viewer(bundle_info.tools[0].name, _show_alignment,
            synonyms=["sv"])

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import _start_seq_viewer
        return _start_seq_viewer(session, tool_name)


bundle_api = _SeqViewerBundleAPI()

def _show_alignment(session, tool_name, alignment):
    from .tool import _start_seq_viewer
    return _start_seq_viewer(session, tool_name, alignment)
