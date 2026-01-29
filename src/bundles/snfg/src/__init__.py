# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California. All rights reserved.
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

"""
ChimeraX SNFG (Symbol Nomenclature for Glycans) bundle.

Displays carbohydrate residues as colored 3D geometric shapes following
the SNFG standard (Varki et al., 2015).
"""

__version__ = "1.0"

from chimerax.core.toolshed import BundleAPI


class _SNFGAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        from .snfg import register_command
        register_command(logger)

    @staticmethod
    def get_class(class_name):
        if class_name == 'SNFGModel':
            from .snfg import SNFGModel
            return SNFGModel
        elif class_name == 'SNFGDrawing':
            from .snfg import SNFGDrawing
            return SNFGDrawing
        return None


bundle_api = _SNFGAPI()
