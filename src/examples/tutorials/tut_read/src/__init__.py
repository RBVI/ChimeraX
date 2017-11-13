# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for opening files,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1

    # Override method for opening file
    @staticmethod
    def open_file(session, stream, format_name):
        # 'open_file' is called by session code to open a file;
        # returns (list of models, status message).
        #
        # The first argument must be named 'session'.
        # The second argument must be named either 'path' or 'stream'.
        # A 'path' argument will be bound to the path to the input file, 
        # which may be a temporary file; a 'stream' argument will be
        # bound to an file-like object in text or binary mode, depending
        # on the DataFormat ChimeraX classifier in bundle_info.xml.
        # If you want both, use 'stream' as the second argument and
        # add a 'file_name' argument, which will be bound to the
        # last (file) component of the path.
        from .io import open_xyz
        return open_xyz(session, stream)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
