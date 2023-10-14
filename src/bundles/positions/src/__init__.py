# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.toolshed import BundleAPI


class _PositionsBundleAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class OpenPositionsInfo(OpenerInfo):
                def open(self, session, data, file_name,
                         models=None, match_names=False, child_models=False, **kw):
                    from . import positions
                    return positions.read_positions(session, data, file_name,
                                                    models = models, match_names = match_names,
                                                    child_models = child_models)
                @property
                def open_args(self):
                    from chimerax.core.commands import ModelsArg, BoolArg
                    return { 'models': ModelsArg,
                             'match_names': BoolArg,
                             'child_models': BoolArg, }
            return OpenPositionsInfo()

        elif mgr == session.save_command:
            from chimerax.save_command import SaverInfo
            class SavePositionsInfo(SaverInfo):
                def save(self, session, path, *, models=None, child_models=False, **kw):
                    from . import positions
                    positions.save_positions(session, path, models=models,
                                             child_models=child_models)
                @property
                def save_args(self):
                    from chimerax.core.commands import ModelsArg, BoolArg
                    return { 'models': ModelsArg,
                             'child_models': BoolArg, }

            return SavePositionsInfo()

bundle_api = _PositionsBundleAPI()
