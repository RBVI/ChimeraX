# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for opening files,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1

    # Implement provider method for opening file
    @staticmethod
    def run_provider(session, name, mgr):
        # 'run_provider' is called by a manager to invoke the 
        # functionality of the provider.  Since the "data formats"
        # manager never calls run_provider (all the info it needs
        # is in the Provider tag), we know that only the "open
        # command" manager will call this function, and customize
        # it accordingly.
        #
        # The 'name' arg will be the same as the 'name' attribute
        # of your Provider tag, and mgr will be the corresponding
        # Manager instance
        #
        # For the "open command" manager, this method must return
        # a chimerax.open_command.OpenerInfo subclass instance.
        from chimerax.open_command import OpenerInfo
        class XyzOpenerInfo(OpenerInfo):
            def open(self, session, data, file_name, **kw):
                # The 'open' method is called to open a file,
                # and must return a (list of models created,
                # status message) tuple.
                from .io import open_xyz
                return open_xyz(session, data)
        return XyzOpenerInfo()


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
