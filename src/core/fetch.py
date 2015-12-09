# vim: set expandtab shiftwidth=4 softtabstop=4:

# -----------------------------------------------------------------------------
#
def fetch_file(session, url, name, save_name, save_dir, uncompress = False,
               cache_dir = '~/Downloads/Chimera', ignore_cache = False, check_certificates=True):
    filename = "%s/%s/%s" % (cache_dir, save_dir, save_name)
    from os.path import expanduser, exists, dirname
    filename = expanduser(filename)
    if exists(filename):
        return filename

    dirname = dirname(filename)
    import os
    os.makedirs(dirname, exist_ok=True)

    from urllib.request import URLError, Request
    headers = {"User-Agent": html_user_agent(session.app_dirs)}
    request = Request(url, unverifiable=True, headers=headers)
    try:
        retrieve_cached_url(request, filename, uncompress=uncompress, logger=session.logger,
                            check_certificates=check_certificates)
    except URLError as e:
        from .errors import UserError
        raise UserError('Fetching url %s failed:\n%s' % (url,str(e)))
    return filename

# -----------------------------------------------------------------------------
#
def retrieve_cached_url(request, filename, logger=None,
                        uncompress=False, update=False, check_certificates=True):
    """Return requested URL in (cached) filename

    :param request: a :py:class:`urlib.request.Request`
    :param filename: where to cache the URL
    :returns: the filename if successful
    :raises urllib.request.URLError: if unsuccessful


    If the filename already exists, fetch the HTTP headers for the
    URL and check the last modified date to see if there is a newer
    version or not.  If there isn't a newer version, return the filename.
    If there is a newer version, or if the filename does not exist,
    save the URL in the filename, and set the file's modified date to
    the HTTP last modified date, and return the filename.
    """
    # Last-Modified: Mon, 19 Sep 2011 22:46:21 GMT
    import os
    from urllib.request import urlopen
    last_modified = None
    if update and os.path.exists(filename):
        if logger:
            logger.status('check for newer version of %s' % filename)
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
        if logger:
            logger.status('fetching %s' % filename)
        request.headers['Accept-encoding'] = 'gzip, identity'
        if check_certificates:
            ssl_context = None
        else:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        with urlopen(request, context = ssl_context) as response:
            compressed = (response.headers['Content-Encoding'] == 'gzip' or uncompress)
            d = response.headers['Last-modified']
            last_modified = _convert_to_timestamp(d)
            import shutil
            with open(filename, 'wb') as f:
                if compressed:
                    import gzip
                    with gzip.GzipFile(fileobj=response) as uncompressed:
                        shutil.copyfileobj(uncompressed, f)
                else:
                        shutil.copyfileobj(response, f)
        if last_modified is not None:
            os.utime(filename, (last_modified, last_modified))
        return filename
    except:
        if os.path.exists(filename):
            os.remove(filename)
        raise

# -----------------------------------------------------------------------------
#
def html_user_agent(app_dirs):
    """"Return HTML User-Agent header according to RFC 2068

    Parameters
    ----------
    app_dirs : a :py:class:`appdirs.AppDirs` instance (session.app_dirs)

    Notes
    -----

    Typical use::

        url = "http://www.example.com/example_file"
        from urllib.request import URLError, Request
        from chimerax.core import utils
        request = Request(url, unverifiable=True, headers={
            "User-Agent": utils.html_user_agent(session.app_dirs),
        })
        try:
            utils.retrieve_cached_url(request, filename, session.logger)
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
        # TODO: strip appropriate CTLs
        return text

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
