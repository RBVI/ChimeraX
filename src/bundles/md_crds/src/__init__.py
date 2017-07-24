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

class _MDCrdsBundleAPI(BundleAPI):

    @staticmethod
    def open_file(session, path, format_name, structure_model=None, replace=True):
        if structure_model is None:
            from chimerax.core.errors import UserError
            raise UserError("Must specify a structure model to read the coordinates into")
        from .read_coords import read_coords
        num_coords = read_coords(session, path, structure_model, format_name, replace=replace)
        if replace:
            return [], "Replaced existing frames of %s with  %d new frames" % (structure_model,
                num_coords)
        return [], "Added %d frames to %s" % (num_coords, model)

    @staticmethod
    def save_file(session, path, format_name, models=None):
        from chimerax.core import atomic
        if models is None:
            models = atomic.all_structures(session)
        else:
            models = [m for m in models if isinstance(m, atomic.Structure)]
        if len(models) == 0:
            from chimerax.core.errors import UserError
            raise UserError("Must specify models to write DCD coordinates")
        # Check that all models have same number of atoms.
        nas = set(m.num_atoms for m in models)
        if len(nas) != 1:
            from chimerax.core.errors import UserError
            raise UserError("Models have different number of atoms")
        from .write_coords import write_coords
        write_coords(session, path, format_name, models)

bundle_api = _MDCrdsBundleAPI()
