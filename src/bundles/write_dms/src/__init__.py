# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.toolshed import BundleAPI

class _DMSBundleAPI(BundleAPI):

    from chimerax.atomic import AtomsArg

    @staticmethod
    def run_provider(session, name, mgr):
        from chimerax.save_command import SaverInfo
        class Info(SaverInfo):
            def save(self, session, path, **kw):
                from .io import write_dms
                write_dms(session, path, status=session.logger.status, **kw)

            @property
            def save_args(self):
                from chimerax.core.commands import BoolArg, SurfaceArg
                from chimerax.atomic import AtomsArg, MolecularSurface, AnnotationError
                class MolecularSurfaceArg(SurfaceArg):
                    @classmethod
                    def parse(cls, text, session):
                        surf, text, rest = super().parse(text, session)
                        if not isinstance(surf, MolecularSurface):
                            raise AnnotationError("Must specify a molecular surface")
                        return surf, text, rest
                return {
                    'displayed_only': BoolArg,
                    'save_normals': BoolArg,
                    'surface': MolecularSurfaceArg,
                }

        return Info()

bundle_api = _DMSBundleAPI()
