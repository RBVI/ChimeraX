# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
utils: Generically useful stuff that doesn't fit elsewhere
==========================================================
"""


# from Mike C. Fletcher's BasicTypes library
# via http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
# Except called flattened, like sorted, since it is nondestructive
def flattened(input, return_types=(list, tuple, set)):
    """Return new flattened version of input

    Parameters
    ----------
    input : a sequence instance (list, tuple, or set)

    Returns
    -------
    A sequence of the same type as the input.
    """
    return_type = type(input)
    output = list(input)
    i = 0
    while i < len(output):
        while isinstance(output[i], return_types):
            if not output[i]:
                output.pop(i)
                i -= 1
                break
            else:
                output[i:i + 1] = output[i]
        i += 1
    if return_type == list:
        return output
    return return_type(output)


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
        from chimera.core import utils
        request = Request(url, unverifiable=True, headers={
            "User-Agent": utils.html_user_agent(session.app_dirs),
        })
        try:
            utils.retrieve_cached_url(request, filename, session.logger)
        except URLError as e:
            from chimera.core.errors import UsereError
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


def _convert_to_timestamp(date):
    # covert HTTP date to POSIX timestamp
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date).timestamp()
    except TypeError:
        return None


def retrieve_cached_url(request, filename, logger=None):
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
    if os.path.exists(filename):
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
        with urlopen(request) as response:
            compressed = response.headers['Content-Encoding'] == 'gzip'
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
        os.remove(filename)
        raise
