# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Fetch zip archive at a DOI.
#
#       http://doi.org/10.5281/zenodo.46266
#
# scrape web page for a zip file and return path local copy.
#
def fetch_doi(session, doi, ignore_cache = False):
    if not '/' in doi:
        from chimerax.core.errors import UserError
        raise UserError('DOI does not contain required "/", got "%s"' % doi)

    doi_url = 'http://doi.org/%s' % doi
    from chimerax.core.fetch import cache_directories, fetch_file
    from os.path import join, isdir, basename
    dirs = cache_directories()
    if not ignore_cache:
        for d in dirs:
            path = join(d, 'DOI', doi)
            if isdir(path):
                from os import listdir
                zf = [f for f in listdir(path) if f.endswith('.zip')]
                if len(zf) == 1:
                    zp = join(path, zf[0])
                    return zp

    filename = fetch_file(session, doi_url, 'doi %s' % doi,
                          save_name = 'temp.html', save_dir = None,
                          uncompress = True, ignore_cache=True)
    # Ick. Scrape this web page looking for a zip file url.
    urls = find_link_in_html(filename, '.zip')
    if len(urls) > 1:
        from chimerax.core.errors import UserError
        raise UserError('Found multiple zip archives at DOI "%s": %s'
                        % (doi, ', '.join(urls)))
    elif len(urls) == 0:
        from chimerax.core.errors import UserError        
        raise UserError('Found no zip archives at DOI "%s"' % doi)

    file_url = urls.pop()
    filename = fetch_file(session, file_url, 'zip %s' % doi, basename(file_url), save_dir = None,
                          uncompress = False, ignore_cache=True)
    
    if dirs:
        from os import makedirs, link
        d = join(dirs[0], 'DOI', doi)
        makedirs(d, exist_ok = True)
        cfile = join(d, basename(file_url))
        link(filename, cfile)
    else:
        cfile = filename

    return cfile

# -----------------------------------------------------------------------------
# Look for file link in html of form
# <link rel="alternate" type="application/zip" href="https://zenodo.org/record/46266/files/nup84-v1.0.zip">
#
def find_link_in_html(filename, url_suffix = '.zip', mime_type = 'application/zip'):
    from html.parser import HTMLParser
    class HrefParser(HTMLParser):
        def __init__(self):
            HTMLParser.__init__(self)
            self.urls = set()
        def handle_starttag(self, tag, attrs):
            if tag == 'link':
                for aname, value in attrs:
                    if aname == 'rel' and value != 'alternate':
                        return
                for aname, value in attrs:                    
                    if aname == 'type' and value != mime_type:
                        return
                for aname, value in attrs:
                    if aname == 'href' and value.endswith(url_suffix):
                        self.urls.add(value)

    f = open(filename, 'r', encoding='utf-8')
    html = f.read()
    f.close()
    p = HrefParser()
    p.feed(html)
    return p.urls

# -----------------------------------------------------------------------------
#
def fetch_doi_archive_file(session, doi, archive_path, ignore_cache = False):

    zip_path = fetch_doi(session, doi, ignore_cache = ignore_cache)
    from zipfile import ZipFile
    zf = ZipFile(zip_path, 'r')
    full_paths = []
    for p in zf.namelist():
        if p.endswith(archive_path):
            full_paths.append(p)
    zfile = full_paths[0] if len(full_paths) == 1 else archive_path
    af = zf.open(zfile)
    return af
