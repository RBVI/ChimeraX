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

class _MyAPI(BundleAPI):

    # @staticmethod
    # def get_class(class_name):
    #     # 'get_class' is called by session code to get class saved in a session
    #     return None

    @staticmethod
    def fetch_from_database(session, identifier, ignore_cache=False, database_name=None, format_name=None, **kw):
        # 'fetch_from_database' is called by session code to fetch data with give identifier
        # returns (list of models, status message)
        from . import mmtf
        return mmtf.fetch_mmtf(session, identifier, ignore_cache, **kw)

    @staticmethod
    def open_file(session, stream, file_name):
        # 'open_file' is called by session code to open a file
        # returns (list of models, status message)
        from . import mmtf
        return mmtf.open_mmtf(session, stream, file_name)

    # @staticmethod
    # def save_file(session, name, _, models=None):
    #     # 'save_file' is called by session code to save a file
    #     from . import mmtf
    #     return mmtf.write_mmtf(session, name, models)

bundle_api = _MyAPI()
