# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1     # register_command called with BundleInfo and
                        # CommandInfo instance instead of command name
                        # (when api_version==0)

    # Override method for registering commands
    @staticmethod
    def register_command(bi, ci, logger):
        # Since we only listed one command in bundle_info.xml
        # we expect only a single call to this method.
        # We pull the function to call and its argument
        # description from the cmd module, adding a
        # synopsis from bundle_info.xml if none is supplied
        # by the code.  We then register the function as
        # the command callback with the chimerax.core.commands
        # module.
        from . import cmd
        from chimerax.core.commands import register
        desc = cmd.basic_desc
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(ci.name, desc, cmd.basic)


bundle_api = _MyAPI()
