# vim: set expandtab shiftwidth=4 softtabstop=4:

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
from .io import write_mol2

class _Mol2BundleAPI(BundleAPI):

    from chimerax.atomic import AtomsArg

    @staticmethod
    def run_provider(session, name, mgr):
        from chimerax.save_command import SaverInfo
        class Info(SaverInfo):
            def save(self, session, path, **kw):
                from .io import write_mol2
                write_mol2(session, path, status=session.logger.status, **kw)

            @property
            def save_args(self):
                from chimerax.core.commands import BoolArg, ModelsArg, ModelArg
                from chimerax.atomic import AtomsArg
                return {
                    'anchor': AtomsArg,
                    'atoms': AtomsArg,
                    'combine_models': BoolArg,
                    'gaff_type': BoolArg,
                    'models': ModelsArg,
                    'rel_model': ModelArg,
                    'res_num': BoolArg,
                    'skip_atoms': AtomsArg,
                    'sybyl_hyd_naming': BoolArg,
                }

        return Info()

bundle_api = _Mol2BundleAPI()
