from chimerax.open_command import FetcherInfo


class TCIAFetcher(FetcherInfo):
    def fetch(self, session, ident, format_name, ignore_cache, **kw):
        from chimerax.dicom.databases import TCIADatabase
        return TCIADatabase.getImages(session, ident, ignore_cache)


fetchers = {"tcia": TCIAFetcher}
