# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

# subcommand name is in bundle_info.xml, but used in various .py files also
subcommand_name = "viewer"
class _SeqViewerBundleAPI(BundleAPI):

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

    @staticmethod
    def run_provider(session, name, manager, *, alignment=None):
        """Register sequence viewer with alignments manager"""
        from .tool import _start_seq_viewer
        return _start_seq_viewer(session, "Sequence Viewer", alignment)


bundle_api = _SeqViewerBundleAPI()
