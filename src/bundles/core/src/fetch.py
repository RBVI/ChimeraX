# vim: set expandtab shiftwidth=4 softtabstop=4:

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

"""
fetch: Retrieve files from a network source
===========================================

The fetch module provides functions for fetching
data from a network source.  Low-level functions
such as fetch_file do not interpret the returned
content, while high-level functions such as
fetch_web maps the content to a particular format
and tries to create models from the content.
"""

_database_fetches = {}
_cache_dirs = []
_timeout_cache = {}
TIMEOUT_CACHE_VALID = 600  # 10 minutes in seconds


# -----------------------------------------------------------------------------
#
def fetch_file(session, url, name, save_name, save_dir, *,
               uncompress=False, transmit_compressed=True,
               ignore_cache=False, check_certificates=True,
               timeout=60, error_status=True):
    """fetch file from URL

    :param session: a ChimeraX :py:class:`~chimerax.core.session.Session`
    :param url: the URL to fetch
    :param name: string to use to identify the data in status messages
    :param save_name: where to save the contents of the URL
    :param save_dir: the cache subdirectory or None for a temporary file
    :param uncompress: contents are compressed (False)
    :param ignore_cache: skip checking for cached file (False)
    :param check_certificates: confirm https certificate (True)
    :param timeout: maximum time to wait for http response
    :param error_status: whether to give a status message if fetching fails
    :returns: the filename
    :raises UserError: if unsuccessful
    """
    from os import path, makedirs
    from urllib.request import URLError, urlparse
    from .errors import UserError
    import time
    in_timeout_cache = False
    if _timeout_cache:
        hostname = urlparse(url).hostname
        if hostname in _timeout_cache:
            cur_time = time.time()
            prev_time = _timeout_cache[hostname]
            if prev_time + TIMEOUT_CACHE_VALID < cur_time:
                del _timeout_cache[hostname]
            else:
                in_timeout_cache = True
                ignore_cache = False
    cache_dirs = cache_directories()
    if not ignore_cache and save_dir is not None:
        for d in cache_dirs:
            filename = path.join(d, save_dir, save_name)
            if path.exists(filename):
                return filename
    if in_timeout_cache:
        raise UserError(f'{hostname} failed to respond')

    if save_dir is None:
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix=save_name)
        filename = f.name
        f.close()
    else:
        dirname = path.join(cache_dirs[0], save_dir)
        filename = path.join(dirname, save_name)
        makedirs(dirname, exist_ok=True)

    try:
        retrieve_url(url, filename, uncompress=uncompress, transmit_compressed=transmit_compressed,
                     logger=session.logger, check_certificates=check_certificates, name=name,
                     timeout=timeout, error_status=error_status)
    except (URLError, EOFError) as err:
        raise UserError('Fetching url %s failed:\n%s' % (url, str(err)))
    return filename


# -----------------------------------------------------------------------------
#
def cache_directories():
    if len(_cache_dirs) == 0:
        try:
            from chimerax import app_dirs
            app_name = app_dirs.appname
        except ImportError:
            app_name = 'ChimeraX'
        from os import path
        cache_dir = path.join('~', 'Downloads', app_name)
        cache_dir = path.expanduser(cache_dir)
        _cache_dirs.append(cache_dir)
    return _cache_dirs


# -----------------------------------------------------------------------------
#
def retrieve_url(url, filename, *, logger=None, uncompress=False, transmit_compressed=True,
                 update=False, check_certificates=True, name=None, timeout=60, error_status=True):
    """Return requested URL in filename

    :param url: the URL to retrive
    :param filename: where to save the contents of the URL
    :param name: string to use to identify the data in status messages
    :param logger: logger instance to use for status and warning messages
    :param uncompress: if true, then uncompress the content
    :param update: if true, then existing file is okay if newer than web version
    :param check_certificates: if true
    :param timeout: maximum time to wait for http response
    :param error_status: whether to give a status message if fetching fails
    :returns: None if an existing file, otherwise the content type
    :raises urllib.request.URLError or EOFError: if unsuccessful

    If 'update' and the filename already exists, fetch the HTTP headers for
    the URL and check the last modified date to see if there is a newer
    version or not.  If there isn't a newer version, return the filename.
    If there is a newer version, or if the filename does not exist,
    save the URL in the filename, and set the file's modified date to
    the HTTP last modified date, and return the filename.
    """
    import os
    import time
    from urllib.request import Request, urlopen, urlparse, URLError
    from chimerax import app_dirs
    from .errors import UserError
    if name is None:
        name = os.path.basename(filename)
    hostname = urlparse(url).hostname
    if _timeout_cache:
        if hostname in _timeout_cache:
            cur_time = time.time()
            prev_time = _timeout_cache[hostname]
            if prev_time + TIMEOUT_CACHE_VALID < cur_time:
                del _timeout_cache[hostname]
            else:
                raise UserError(f'{hostname} failed to respond')
    headers = {"User-Agent": html_user_agent(app_dirs)}
    request = Request(url, unverifiable=True, headers=headers)
    last_modified = None
    if update and os.path.exists(filename):
        if logger:
            logger.status('check for newer version of %s' % name, secondary=True)
        info = os.stat(filename)
        request.method = 'HEAD'
        try:
            with urlopen(request, timeout=timeout) as response:
                d = response.headers['Last-modified']
                last_modified = _convert_to_timestamp(d)
            if last_modified is None and logger:
                logger.warning('Invalid date "%s" for %s' % (d, request.full_url))
            if last_modified is None or last_modified <= info.st_mtime:
                return
        except URLError:
            pass
        request.method = 'GET'
    try:
        request.headers['Accept-encoding'] = 'gzip, identity' if transmit_compressed else 'identity'
        if check_certificates:
            ssl_context = None
        else:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        with urlopen(request, timeout=timeout, context=ssl_context) as response:
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
    except Exception as err:
        if os.path.exists(filename):
            os.remove(filename)
        if logger and error_status:
            logger.status('Error fetching %s' % name, secondary=True, blank_after=15)
        import socket
        if isinstance(err, URLError) and isinstance(err.reason, (TimeoutError, socket.timeout)):
            _timeout_cache[hostname] = time.time()
            raise UserError(f'{hostname} failed to respond')
        if isinstance(err, socket.timeout):
            raise UserError(f'{hostname} failed to respond')
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

    if content_length is not None and tb != content_length:
        # In ChimeraX bug #2747 zero bytes were read and no error reported.
        from urllib.request import URLError
        raise URLError('Got %d bytes when %d were expected' % (tb, content_length))


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
    # want system to resemble "OS VERSION ARCH"
    system = platform.system()
    if system == "Darwin":
        mac_ver = platform.mac_ver()
        system = f"{system} {mac_ver[0]} {mac_ver[2]}"
    elif system == "Windows":
        u = platform.uname()
        system = f"{system} {u.version} {u.machine}"
    elif system == "Linux":
        import distro
        system = f"{system} {distro.name()} {distro.version(best=True)} {platform.machine()}"
    else:
        system = f"{system} {platform.version()} {platform.machine()}"
    if system:
        user_agent += f" ({comment(system)})"
    return user_agent


# -----------------------------------------------------------------------------
#
def _convert_to_timestamp(date):
    # covert HTTP date to POSIX timestamp
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date).timestamp()
    except (TypeError, ValueError):
        return None
