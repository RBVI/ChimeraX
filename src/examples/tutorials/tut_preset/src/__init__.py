# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for defining presets
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1

    # Override method for defining presets
    #TODO
    @staticmethod
    def register_selector(bi, si, logger):
        # bi is an instance of chimerax.core.toolshed.BundleInfo
        # si is an instance of chimerax.core.toolshed.SelectorInfo
        # logger is an instance of chimerax.core.logger.Logger

        # This method is called once for each selector listed
        # in bundle_info.xml.  Since we list only one selector,
        # we expect a single call to this method.
        from .selector import register
        return register(si.name, logger)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
