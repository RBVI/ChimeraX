# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for defining presets
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    # Override method for defining presets
    @staticmethod
    def run_provider(session, name, mgr, **kw):
        # 'session' is the current chimerax.core.session.Session object
        # 'name' is the name of the preset to execute
        # 'mgr' is the preset manager (a.k.a. session.presets)
        # 'kw', the keyword dictionary, is empty
        #
        # Note that this method is called by all managers that your
        # bundle declares providers for, so if your bundle has providers
        # for multiple different managers the the 'mgr' argument will
        # not always be the session.presets instance, and some managers
        # may provide non-empty keyword dictionaries to providers.

        from .presets import run_preset
        run_preset(session, name, mgr)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
