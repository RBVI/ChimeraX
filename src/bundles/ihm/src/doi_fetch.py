# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Fetch zip archive at a DOI.
#
#       http://doi.org/10.5281/zenodo.46266
#
# scrape web page for a zip file and return path local copy.
#
def fetch_doi(session, doi, url, ignore_cache = False):
    if not '/' in doi:
        from chimerax.core.errors import UserError
        raise UserError('DOI does not contain required "/", got "%s"' % doi)
        
    from chimerax.core.fetch import cache_directories, fetch_file
    from os.path import join, isdir, basename
    dirs = cache_directories()
    if not ignore_cache:
        for d in dirs:
            path = join(d, 'DOI', doi)
            if isdir(path):
                from os import listdir
                if url:
                    zip_name = basename(url)
                    zf = [f for f in listdir(path) if f == zip_name]
                else:
                    zf = [f for f in listdir(path) if f.endswith('.zip')]
                if len(zf) == 1:
                    zp = join(path, zf[0])
                    return zp

    if url is None:
        zip_file_url = find_doi_zip_archive_url(session, doi)
    else:
        zip_file_url = url
    zip_filename = basename(zip_file_url)
    if dirs:
        from os import makedirs, link
        d = join(dirs[0], 'DOI', doi)
        makedirs(d, exist_ok = True)
        save_dir = d
    else:
        save_dir = None

    filename = fetch_file(session, zip_file_url, 'zip %s %s' % (doi, zip_filename), zip_filename,
                          save_dir = save_dir, uncompress = False, ignore_cache=True)

    return filename

# -----------------------------------------------------------------------------
# HTML scraping to find zip file URL used with Zenodo file sharing site.
#
def find_doi_zip_archive_url(session, doi):
    from chimerax.core.fetch import fetch_file
    doi_url = 'http://doi.org/%s' % doi
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
    return file_url

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
def fetch_doi_archive_file(session, doi, url, archive_path, mode = 'r', ignore_cache = False):

    zip_path = fetch_doi(session, doi, url, ignore_cache = ignore_cache)
    from zipfile import ZipFile, BadZipFile
    try:
        zf = ZipFile(zip_path, 'r')
    except BadZipFile:
        session.logger.warning('Could not open zip file %s' % zip_path)
        raise
    full_paths = []
    for p in zf.namelist():
        if p.endswith(archive_path):
            full_paths.append(p)
    zfile = full_paths[0] if len(full_paths) == 1 else archive_path
    af = zf.open(zfile)	# Returns bytes stream 
    if mode == 'r':
        import io
        af = io.TextIOWrapper(af)
    return af

# -----------------------------------------------------------------------------
#
def unzip_archive(session, doi, url, directory = None, ignore_cache = False):
    zip_path = fetch_doi(session, doi, url, ignore_cache = ignore_cache)
    from zipfile import ZipFile
    zf = ZipFile(zip_path, 'r')

    if directory is None:
        directory = zip_path[:-4] if zip_path.endswith('.zip') else (zip_path + '_extracted')
        from os.path import isdir
        if isdir(directory):
            extracted = True
        else:
            extracted = False
            from os import mkdir
            mkdir(directory)
    else:
        # Check if zip file already extracted in specified directory
        # TODO: Should protect against absolute and relative paths in zip archive.
        extracted = False
        nl = zf.namelist()
        from os.path import exists, join
        for f in nl:
            if exists(join(directory, f)):
                extracted = True
                break
    try:
        if not extracted:
            zf.extractall(path = directory)
    finally:
        zf.close()

    return directory
