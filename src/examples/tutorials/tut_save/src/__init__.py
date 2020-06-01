# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for opening and saving files,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1

    # Implement provider methods for opening and saving files
    @staticmethod
    def run_provider(session, name, mgr):
        # 'run_provider' is called by a manager to invoke the 
        # functionality of the provider.  Since the "data formats"
        # manager never calls run_provider (all the info it needs
        # is in the Provider tag), we know that only the "open
        # command" or "save command" managers will call this
        # function, and customize it accordingly.
        #
        # The 'name' arg will be the same as the 'name' attribute
        # of your Provider tag, and mgr will be the corresponding
        # Manager instance
        #
        # For the "open command" manager, this method must return
        # a chimerax.open_command.OpenerInfo subclass instance.
        # For the "save command" manager, this method must return
        # a chimerax.save_command.SaverInfo subclass instance.
        #
        # The "open command" manager is also session.open_command,
        # and likewise the "save command" manager is
        # session.save_command.  We therefore decide what to do
        # by testing our 'mgr' argument...
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class XyzInfo(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    # The 'open' method is called to open a file,
                    # and must return a (list of models created,
                    # status message) tuple.
                    from .io import open_xyz
                    return open_xyz(session, data)
        else:
            from chimerax.save_command import SaverInfo
            class XyzInfo(SaverInfo):
                def save(self, session, path, *, structures=None):
                    # The 'save' method is called to save a file,
                    # There is no return value.
                    #
                    # This bundle supports an optional 'structures'
                    # keyword arument to the save command and
                    # therefore will have the 'structures' argument
                    # to this function provided with whatever
                    # value the user supplied, if any.  The
                    # 'save_args' property below informs the
                    # "save command" manager of the optional
                    # keywords and their value types that this
                    # bundle supports.
                    from .io import save_xyz
                    save_xyz(session, path, structures)

                @property
                def save_args(self):
                    # The 'save_args' property informs the
                    # "save command" manager of any optional
                    # bundle/format-specific keyword arguments
                    # to the 'save' command that this bundle
                    # supports.  If given by the user, they will
                    # be provided to the above 'save' method.  If
                    # there are no such keywords, you need not
                    # implement this property.
                    #
                    # This property should return a dictionary
                    # that maps the *Python* name of a keyword to
                    # an Annotation subclass.  Annotation classes
                    # are used to convert user-typed text into
                    # Python values.
                    from chimerax.atomic import StructuresArg
                    return { 'structures': StructuresArg }

        return XyzInfo()


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
