# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for fetching from databases,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def run_provider(session, name, mgr):
        # 'run_provider' is called by a manager to invoke the 
        # functionality of the provider.  Since we only provide
        # single provider to a single manager, we know this method
        # will only be called by the "open command" manager to
        # fetch HomoloGene data, and customize this routine accordingly.
        #
        # The 'name' arg will be the same as the 'name' attribute
        # of your Provider tag, and mgr will be the corresponding
        # Manager instance
        #
        # For the "open command" manager with type="fetch", this method
        # must return a chimerax.open_command.FetcherInfo subclass instance.
        from chimerax.open_command import FetcherInfo
        class HomoloGeneFetcherInfo(FetcherInfo):
            def fetch(self, session, identifier, format_name, ignore_cache, **kw):
                from .fetch import fetch_homologene
                return fetch_homologene(session, identifier, ignore_cache=ignore_cache, **kw)
        return HomoloGeneFetcherInfo()


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
