# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
__version__ = "3.4.1"

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
