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
    def open_file(session, stream, fname, format_name=None, model=None, filespec=None, replace=True):
        if model is None:
            from chimerax.core.errors import UserError
            raise UserError("Must specify a model to read the coordinates into")
        from .read_coords import read_coords
        num_coords = read_coords(session, filespec, model, format_name, replace=replace)
        if replace:
            return [], "Replaced existing frames of %s with  %d new frames" % (model, num_coords)
        return [], "Added %d frames to %s" % (num_coords, model)

bundle_api = _MDCrdsBundleAPI()
