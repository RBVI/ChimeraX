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
               uncompress=False, ignore_cache=False, check_certificates=True):
    """fetch file from URL

    :param session: a ChimeraX :py:class:`~chimerax.core.session.Session`
    :param url: the URL to fetch
    :param name: string to use to identify the data in status messages
    :param save_name: where to save the contents of the URL
    :param save_dir: the cache subdirectory or None for a temporary file
    :param uncompress: contents are compressed (False)
    :param ignore_cache: skip checking for cached file (False)
    :param check_certificates: confirm https certificate (True)
    :returns: the filename
    :raises UserError: if unsuccessful
    """
    from os import path, makedirs
    cache_dirs = cache_directories()
    if not ignore_cache and save_dir is not None:
        for d in cache_dirs:
            filename = path.join(d, save_dir, save_name)
            if path.exists(filename):
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

    from urllib.request import URLError
    try:
        retrieve_url(url, filename, uncompress=uncompress, logger=session.logger,
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
def retrieve_url(url, filename, *, logger=None, uncompress=False,
                 update=False, check_certificates=True, name=None):
    """Return requested URL in filename

    :param url: the URL to retrive
    :param filename: where to save the contents of the URL
    :param name: string to use to identify the data in status messages
    :param logger: logger instance to use for status and warning messages
    :param uncompress: if true, then uncompress the content
    :param update: if true, then existing file is okay if newer than web version
    :param check_certificates: if true
    :returns: None if an existing file, otherwise the content type
    :raises urllib.request.URLError: if unsuccessful


    If 'update' and the filename already exists, fetch the HTTP headers for
    the URL and check the last modified date to see if there is a newer
    version or not.  If there isn't a newer version, return the filename.
    If there is a newer version, or if the filename does not exist,
    save the URL in the filename, and set the file's modified date to
    the HTTP last modified date, and return the filename.
    """
    import os
    if name is None:
        name = os.path.basename(filename)
    from urllib.request import Request, urlopen
    from chimerax import app_dirs
    headers = {"User-Agent": html_user_agent(app_dirs)}
    request = Request(url, unverifiable=True, headers=headers)
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
            return
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
            ct = response.headers['Content-Type']
            if not compressed:
                ce = response.headers['Content-Encoding']
                if ce:
                    compressed = ce.casefold() in ('gzip', 'x-gzip')
                if ct:
                    compressed = compressed or ct.casefold() in (
                        'application/gzip', 'application/x-gzip')
                    ct = 'application/octet-stream'
            if logger:
                logger.info('Fetching%s %s from %s' % (
                    " compressed" if compressed else "", name,
                    request.get_full_url()))
            d = response.headers['Last-modified']
            last_modified = _convert_to_timestamp(d)
            content_length = response.headers['Content-Length']
            if content_length is not None:
                content_length = int(content_length)
            with open(filename, 'wb') as f:
                if compressed:
                    read_and_uncompress(response, f, name, content_length, logger)
                else:
                    read_and_report_progress(response, f, name, content_length, logger)
        if last_modified is not None:
            os.utime(filename, (last_modified, last_modified))
        if logger:
            logger.status('%s fetched' % name, secondary=True, blank_after=5)
        return ct
    except:
        if os.path.exists(filename):
            os.remove(filename)
        if logger:
            logger.status('Error fetching %s' % name, secondary=True, blank_after=15)
        raise


# -----------------------------------------------------------------------------
#
def read_and_uncompress(file_in, file_out, name, content_length, logger, chunk_size=1048576):

    # Read compressed data into buffer reporting progress.
    from io import BytesIO
    cdata = BytesIO()
    read_and_report_progress(file_in, cdata, name, content_length, logger, chunk_size)
    cdata.seek(0)

    # Decompress data to file.
    import gzip
    import shutil
    with gzip.GzipFile(fileobj=cdata) as uncompressed:
        shutil.copyfileobj(uncompressed, file_out)


# -----------------------------------------------------------------------------
#
def read_and_report_progress(file_in, file_out, name, content_length, logger, chunk_size=1048576):
    tb = 0
    while True:
        bytes = file_in.read(chunk_size)
        if bytes:
            file_out.write(bytes)
        else:
            break
        tb += len(bytes)
        if content_length:
            msg = 'Fetching %s, %.3g of %.3g Mbytes received' % (name, tb / 1048576, content_length / 1048576)
        else:
            msg = 'Fetching %s, %.3g Mbytes received' % (name, tb / 1048576)
        logger.status(msg)


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
            retrieve_url(request, filename, logger=session.logger)
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
def fetch_web(session, url, **kw):
    # TODO: deal with content encoding for text formats
    import os
    from urllib import parse
    from . import io
    cache_dir = os.path.expanduser(os.path.join('~', 'Downloads'))
    o = parse.urlparse(url)
    path = parse.unquote(o.path)
    basename = os.path.basename(path)
    nominal_format, basename, compression_ext = io.deduce_format(basename, no_raise=True)
    if nominal_format is not None and nominal_format.name == 'HTML':
        # Let the help viewer fetch it's own files
        try:
            import chimerax.help_viewer as browser
        except ImportError:
            from .errors import UserError
            raise UserError('Help viewer is not installed')
        browser.show_url(session, url)
        return [], "Opened %s" % url
    base, ext = os.path.splitext(basename)
    filename = os.path.join(cache_dir, '%s%s' % (base, ext))
    count = 0
    while os.path.exists(filename):
        count += 1
        filename = os.path.join(cache_dir, '%s(%d)%s' % (base, count, ext))
    uncompress = compression_ext is not None
    content_type = retrieve_url(url, filename, logger=session.logger, uncompress=uncompress)
    session.logger.info('Downloaded %s to %s' % (basename, filename))
    mime_format = None
    if 'format' in kw:
        format_name = kw['format']
        del kw['format']
        mime_format = io.format_from_name(format_name)
    if mime_format is None:
        for mime_format in io.formats():
            if content_type in mime_format.mime_types:
                break
        else:
            if content_type != 'application/octet-stream':
                session.logger.info('Unrecognized mime type: %s' % content_type)
            mime_format = nominal_format
    if mime_format is None:
        from .errors import UserError
        raise UserError('Unable to deduce format of %s' % url)
    if mime_format != nominal_format:
        session.logger.info('mime type (%s), does not match file name extension (%s)' % (content_type, ext))
        new_ext = mime_format.extensions[0]
        new_filename = os.path.join(cache_dir, '%s%s' % (base, new_ext))
        count = 0
        while os.path.exists(new_filename):
            count += 1
            new_filename = os.path.join(cache_dir, '%s(%d)%s' % (base, count, new_ext))
        session.logger.info('renaming "%s" to "%s"' % (filename, new_filename))
        os.rename(filename, new_filename)
        nominal_format = mime_format
        filename = new_filename
    return io.open_data(session, filename, format=nominal_format.name, **kw)


# -----------------------------------------------------------------------------
#
def register_web_fetch():
    def fetch_http(session, scheme_specific_part, **kw):
        return fetch_web(session, 'http:' + scheme_specific_part, **kw)
    register_fetch('http', fetch_http, None, prefixes=['http'])

    def fetch_https(session, scheme_specific_part, **kw):
        return fetch_web(session, 'https:' + scheme_specific_part, **kw)
    register_fetch('https', fetch_https, None, prefixes=['https'])

    def fetch_ftp(session, scheme_specific_part, **kw):
        return fetch_web(session, 'ftp:' + scheme_specific_part, **kw)
    register_fetch('ftp', fetch_ftp, None, prefixes=['ftp'])


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
