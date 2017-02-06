# vim: set expandtab shiftwidth=4 softtabstop=4:

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

_database_fetches = {}
_cache_dirs = []


# -----------------------------------------------------------------------------
#
def fetch_file(session, url, name, save_name, save_dir, *,
               uncompress=False, ignore_cache=False, check_certificates=True,
               log='info'):
    from os import path, makedirs
    cache_dirs = cache_directories()
    if not ignore_cache and save_dir is not None:
        for d in cache_dirs:
            filename = path.join(d, save_dir, save_name)
            if path.exists(filename):
                msg = 'Fetching %s from local cache: %s' % (name, filename)
                if log == 'info':
                    session.logger.info(msg)
                elif log == 'status':
                    session.logger.status(msg)
                return filename

    if save_dir is None:
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=save_name)
        filename = f.name
        f.close()
    else:
        dirname = path.join(cache_dirs[0], save_dir)
        filename = path.join(dirname, save_name)
        makedirs(dirname, exist_ok=True)

    from urllib.request import URLError, Request
    from chimerax import app_dirs
    headers = {"User-Agent": html_user_agent(app_dirs)}
    request = Request(url, unverifiable=True, headers=headers)
    try:
        retrieve_url(request, filename, uncompress=uncompress, logger=session.logger,
                     check_certificates=check_certificates, name=name)
    except URLError as e:
        from .errors import UserError
        raise UserError('Fetching url %s failed:\n%s' % (url, str(e)))
    return filename


# -----------------------------------------------------------------------------
#
def cache_directories():
    from os import path
    from chimerax import app_dirs
    if len(_cache_dirs) == 0:
        cache_dir = path.join('~', 'Downloads', app_dirs.appname)
        cache_dir = path.expanduser(cache_dir)
        _cache_dirs.append(cache_dir)
    return _cache_dirs


# -----------------------------------------------------------------------------
#
def retrieve_url(request, filename, logger=None, uncompress=False,
                 update=False, check_certificates=True, name=None):
    """Return requested URL in filename

    :param request: a :py:class:`urlib.request.Request`
    :param filename: where to save the contents of the URL
    :param name: string to use to identify the data in status messages
    :returns: the filename if successful
    :raises urllib.request.URLError: if unsuccessful


    If the filename already exists, fetch the HTTP headers for the
    URL and check the last modified date to see if there is a newer
    version or not.  If there isn't a newer version, return the filename.
    If there is a newer version, or if the filename does not exist,
    save the URL in the filename, and set the file's modified date to
    the HTTP last modified date, and return the filename.
    """
    import os
    if name is None:
        name = os.path.basename(filename)
    from urllib.request import urlopen
    last_modified = None
    if update and os.path.exists(filename):
        if logger:
            logger.status('check for newer version of %s' % name, secondary=True)
        info = os.stat(filename)
        request.method = 'HEAD'
        with urlopen(request) as response:
            d = response.headers['Last-modified']
            last_modified = _convert_to_timestamp(d)
        if last_modified is None and logger:
            logger.warning('Invalid date "%s" for %s' % (d, request.full_url))
        if last_modified is None or last_modified <= info.st_mtime:
            return filename
        request.method = 'GET'
    try:
        request.headers['Accept-encoding'] = 'gzip, identity'
        if check_certificates:
            ssl_context = None
        else:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        with urlopen(request, context=ssl_context) as response:
            compressed = uncompress
            if not compressed:
                ce = response.headers['Content-Encoding']
                if ce:
                    compressed = ce.casefold() in ('gzip', 'x-gzip')
                ct = response.headers['Content-Type']
                if ct:
                    compressed = compressed or ct.casefold() in (
                            'application/gzip', 'application/x-gzip')
            if logger:
                logger.info('Fetching%s %s from %s' % (
                    " compressed" if compressed else "", name,
                    request.get_full_url()))
            d = response.headers['Last-modified']
            last_modified = _convert_to_timestamp(d)
            import shutil
            with open(filename, 'wb') as f:
                # TODO: Put code in here that uses response.read(num_bytes) to read
                # the fetched file a chunk at a time and output status messages about
                # the progress.
                if compressed:
                    import gzip
                    with gzip.GzipFile(fileobj=response) as uncompressed:
                        shutil.copyfileobj(uncompressed, f)
                else:
                        shutil.copyfileobj(response, f)
        if last_modified is not None:
            os.utime(filename, (last_modified, last_modified))
        if logger:
            logger.status('%s fetched' % name, secondary=True, blank_after=5)
        return filename
    except:
        if os.path.exists(filename):
            os.remove(filename)
        if logger:
            logger.status('Error fetching %s' % name, secondary=True, blank_after=15)
        raise


# -----------------------------------------------------------------------------
#
def html_user_agent(app_dirs):
    """"Return HTML User-Agent header according to RFC 2068

    Parameters
    ----------
    app_dirs : a :py:class:`appdirs.AppDirs` instance (chimerax.app_dirs)

    Notes
    -----
    The user agent may have single quote characters in it.

    Typical use::

        url = "http://www.example.com/example_file"
        from urllib.request import URLError, Request
        request = Request(url, unverifiable=True, headers={
            "User-Agent": html_user_agent(chimerax.app_dirs),
        })
        try:
            retrieve_url(request, filename, session.logger)
        except URLError as e:
            from chimerax.core.errors import UsereError
            raise UserError(str(e))
    """
    # The name, author, and version must be "tokens"
    #
    #   token          = 1*<any CHAR except CTLs or tspecials>
    #   CTLs           = <any US-ASCII control character
    #                     (octets 0 - 31) and DEL (127)>
    #   tspecials      = "(" | ")" | "<" | ">" | "@"
    #                    | "," | ";" | ":" | "\" | <">
    #                    | "/" | "[" | "]" | "?" | "="
    #                    | "{" | "}" | SP | HT
    #   comment        = "(" *( ctext | comment ) ")"
    #   ctext          = <any TEXT excluding "(" and ")">
    #   TEXT           = <any OCTET except CTLs,
    #                     but including LWS>
    #   LWS            = [CRLF] 1*( SP | HT )

    ctls = ''.join(chr(x) for x in range(32)) + chr(127)
    tspecials = '()<>@,;:\"/[]?={} \t'
    bad = ctls + tspecials

    def token(text):
        return ''.join([c for c in text if c not in bad])

    def comment(text):
        # TODO: check for matched parenthesis
        from html import escape
        return escape(text)

    app_author = app_dirs.appauthor
    app_name = app_dirs.appname
    app_version = app_dirs.version

    user_agent = ''
    if app_author is not None:
        user_agent = "%s-" % token(app_author)
    user_agent += token(app_name)
    if app_version is not None:
        user_agent += "/%s" % token(app_version)
    import platform
    system = platform.system()
    if system:
        user_agent += " (%s)" % comment(system)
    return user_agent


# -----------------------------------------------------------------------------
#
def _convert_to_timestamp(date):
    # covert HTTP date to POSIX timestamp
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date).timestamp()
    except TypeError:
        return None


# -----------------------------------------------------------------------------
#
def register_fetch(database_name, fetch_function, file_format,
                   prefixes=(), is_default_format=False, example_id=None):
    d = fetch_databases()
    df = d.get(database_name, None)
    if df is None:
        d[database_name] = df = DatabaseFetch(database_name, file_format)
    df.add_format(file_format, fetch_function)
    if is_default_format:
        df.default_format = file_format
    if example_id:
        df.example_id = example_id
    for p in prefixes:
        df.prefix_format[p] = file_format


# -----------------------------------------------------------------------------
#
def fetch_databases():
    return _database_fetches


# -----------------------------------------------------------------------------
#
def database_formats(from_database):
    return fetch_databases()[from_database].fetch_function.keys()


# -----------------------------------------------------------------------------
#
def fetch_from_database(session, from_database, id, format=None, name=None, ignore_cache=False, **kw):
    d = fetch_databases()
    df = d[from_database]
    from .logger import Collator
    with Collator(session.logger, "Summary of feedback from opening %s fetched from %s" % (id, from_database)):
        models, status = df.fetch(session, id, format=format, ignore_cache=ignore_cache, **kw)
    if name is not None:
        for m in models:
            m.name = name
    return models, status


# -----------------------------------------------------------------------------
#
def fetch_from_prefix(prefix):
    d = fetch_databases()
    for db in d.values():
        if prefix in db.prefix_format:
            return db.database_name, db.prefix_format[prefix]
    return None, None


# -----------------------------------------------------------------------------
#
def prefixes():
    d = fetch_databases()
    return sum((tuple(db.prefix_format.keys()) for db in d.values()), ())


# -----------------------------------------------------------------------------
#
class DatabaseFetch:

    def __init__(self, database_name, default_format=False, example_id=None):
        self.database_name = database_name
        self.default_format = default_format
        self.fetch_function = {}		# Map format to fetch function
        self.prefix_format = {}

    def add_format(self, format_name, fetch_function):
        # fetch_function() takes session and database id arguments, returns model list.
        from . import io
        f = io.format_from_name(format_name)
        if f is None:
            self.fetch_function[format_name] = fetch_function
        else:
            for name in f.nicknames:
                self.fetch_function[name] = fetch_function

    def fetch(self, session, database_id, format=None, ignore_cache=False, **kw):
        f = self.default_format if format is None else format
        fetch = self.fetch_function[f]
        return fetch(session, database_id, ignore_cache=ignore_cache, **kw)
