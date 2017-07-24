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

class _Mol2BundleAPI(BundleAPI):

    @staticmethod
    def save_file(session, path, models=None, atoms=None, anchor=None, rel_model=None,
            sybyl_hyd_naming=True, combine_models=False, skip_atoms=None, res_num=False,
            gaff_type=False):
        from .io import write_mol2
        return write_mol2(session, path, models=models, atoms=atoms,
            status=session.logger.status, anchor=anchor, rel_model=rel_model,
            sybyl_hyd_naming=sybyl_hyd_naming, combine_models=combine_models,
            skip_atoms=skip_atoms, res_num=res_num, gaff_type=gaff_type)

bundle_api = _Mol2BundleAPI()
