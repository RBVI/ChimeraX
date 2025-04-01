# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
__version__ = "3.6.1"

import os

def get_bin() -> str:
    return os.path.join(os.path.dirname(__file__), "bin")

from .header_sequence import HeaderSequence, FixedHeaderSequence, DynamicHeaderSequence, \
    DynamicStructureHeaderSequence, position_color_to_qcolor

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
