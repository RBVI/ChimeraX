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

from .header_sequence import HeaderSequence, FixedHeaderSequence, \
    DynamicHeaderSequence, DynamicStructureHeaderSequence, register_header, registered_headers
from .consensus import Consensus
from .conservation import Conservation

from chimerax.core.toolshed import BundleAPI

class _AlignmentHdrsAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "Consensus":
            return Consensus
        if class_name == "Conservation":
            return Conservation


bundle_api = _AlignmentHdrsAPI()
