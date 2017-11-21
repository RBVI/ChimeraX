# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for fetching from databases,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1

    # Override method for opening file
    @staticmethod
    def fetch_from_database(session, identifier, **kw):
        # 'fetch_from_database' is called by session code to fetch
        # an entry from a network resource.
        # returns (list of models created, status message).
        #
        # 'session' is an instance of chimerax.core.session.Session
        # 'identifier' is a string for the database entry name
        # Additional keywords may include:
        # 'format_name' (string) name of the preferred download format
        # 'ignore_cache' (boolean): whether cached copy of the data may be used
        from .fetch import fetch_homologene
        return fetch_homologene(session, identifier, **kw)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
