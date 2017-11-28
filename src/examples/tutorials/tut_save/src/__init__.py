# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for opening and saving files,
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
        #
        # Return value for this method should be a 2-tuple of
        # a list of structures and a status message string to
        # display to the user.
        from .io import open_xyz
        return open_xyz(session, stream)

    # Override method for saving file
    @staticmethod
    def save_file(session, path, models=None):
        # 'save_file' is called by session code to save a file.
        # There is no return value.
        #
        # The first argument must be named 'session'.
        # The second argument must be named 'path'.
        # A 'path' argument will be bound to a string
        # or file-like object that can be converted into
        # a file-like object using chimerax.core.io.open_filename
        # (see src/io.py).
        from .io import save_xyz
        save_xyz(session, path, models)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
