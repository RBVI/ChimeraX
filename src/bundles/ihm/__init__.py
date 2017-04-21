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

class _IHMAPI(BundleAPI):

    @staticmethod
    def open_file(session, f, name, filespec=None, format_name=None, **kw):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        if format_name == 'IHM':
            from . import ihm
            return ihm.read_ihm(session, filespec, name, **kw)
        elif format_name == 'Binary Coordinates':
            if 'model' not in kw:
                from chimerax.core.errors import UserError
                raise UserError('Must specify model option to open command to load binary coordinates')
            from . import coordsets
            coordsets.read_coordinate_sets(filespec, kw['model'])
            return [], 'Read coordinate set %s' % name
        raise ValueError('Attempt to open unrecognized format "%s"' % format_name)

    @staticmethod
    def save_file(session, name, _, models=None):
        # 'save_file' is called by session code to save a file
        from . import savecoords
        return savecoords.save_binary_coordinates(session, name, models)

bundle_api = _IHMAPI()
