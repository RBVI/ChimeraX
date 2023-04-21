from chimerax.open_command import FetcherInfo
from .databases import TCIADatabase

class TCIAFetcher(FetcherInfo):
    def fetch(self, session, ident, format_name, ignore_cache, **kw):
        return TCIADatabase.getImages(session, ident, ignore_cache)

fetchers = {
    'tcia': TCIAFetcher
}
