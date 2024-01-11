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

# subcommand name is in bundle_info.xml, but used in various .py files also
subcommand_name = "grid"
class _ProfileGridsBundleAPI(BundleAPI):

    """
    @staticmethod
    def get_class(class_name):
        if class_name == "SequenceViewer":
            from .tool import SequenceViewer
            return SequenceViewer
        # so that old sessions _might_ work
        if class_name == "Consensus":
            from chimerax.alignment_headers import Consensus
            return Consensus
        if class_name == "Conservation":
            from chimerax.alignment_headers import Conservation
            return Conservation
    """

    @staticmethod
    def run_provider(session, name, manager, *, alignment=None):
        """Register sequence viewer with alignments manager"""
        from .tool import ProfileGridsTool
        return ProfileGridsTool(session, "Sequence Viewer", alignment)

bundle_api = _ProfileGridsBundleAPI()
