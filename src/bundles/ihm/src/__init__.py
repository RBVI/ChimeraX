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
    def open_file(session, path, file_name, format_name, ensembles=False, model=None):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        if format_name == 'ihm':
            from . import ihm
            return ihm.read_ihm(session, path, file_name, load_ensembles = ensembles)
        raise ValueError('Attempt to open unrecognized format "%s"' % format_name)

    @staticmethod
    def save_file(session, name, _, models=None):
        # 'save_file' is called by session code to save a file
        from . import savecoords
        return savecoords.save_binary_coordinates(session, name, models)

    @staticmethod
    def fetch_from_database(session, identifier, ignore_cache=False, database_name=None, format_name=None, **kw):
        # 'fetch_from_database' is called by session code to fetch data with give identifier
        from .fetch_ihm import fetch_ihm
        return fetch_ihm(session, identifier, ignore_cache=ignore_cache)

bundle_api = _IHMAPI()
