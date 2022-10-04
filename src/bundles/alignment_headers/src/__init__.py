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
__version__ = "3.2.1"

import os

def get_bin() -> str:
    return os.path.join(os.path.dirname(__file__), "bin")

from .header_sequence import HeaderSequence, FixedHeaderSequence, DynamicHeaderSequence, \
    DynamicStructureHeaderSequence

from chimerax.core.toolshed import BundleAPI

class _AlignmentHdrsAPI(BundleAPI):

    @classmethod
    def get_class(cls, class_name):
        import importlib
        hdr_mod = importlib.import_module(".%s" % class_name.lower(), cls.__module__)
        return getattr(hdr_mod, class_name)

    @classmethod
    def run_provider(cls, session, name, mgr, **kw):
        return cls.get_class(name)

bundle_api = _AlignmentHdrsAPI()
