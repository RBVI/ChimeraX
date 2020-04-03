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

def fetch_web(session, url, ignore_cache=False, new_tab=False, data_format=None, **kw):
    # TODO: deal with content encoding for text formats
    # TODO: how would "ignore_cache" work?
    import os
    from urllib import parse
    cache_dir = os.path.expanduser(os.path.join('~', 'Downloads'))
    o = parse.urlparse(url)
    path = parse.unquote(o.path)
    basename = os.path.basename(path)
    from chimerax.data_formats import NoFormatError
    use_html = False
    if data_format is None:
        try:
            nominal_format = session.data_formats.open_format_from_file_name(basename)
        except NoFormatError:
            use_html = True
            nomimal_format = None
        else:
            use_html = nominal_format.name == 'HTML'
    else:
        nominal_format = session.data_formats[data_format]
        use_html = nominal_format.name == 'HTML'
    if use_html and not url.startswith('ftp:'):
        return session.open_command.open_data(url, format='HTML',
            ignore_cache=ignore_cache, new_tab=new_tab, **kw)
    base, ext = os.path.splitext(basename)
    filename = os.path.join(cache_dir, '%s%s' % (base, ext))
    count = 0
    while os.path.exists(filename):
        count += 1
        filename = os.path.join(cache_dir, '%s(%d)%s' % (base, count, ext))
    from chimerax import io
    uncompress = io.remove_compression_suffix(basename) != basename
    content_type = retrieve_url(url, filename, logger=session.logger,
        uncompress=uncompress)
    session.logger.info('Downloaded %s to %s' % (basename, filename))
    if data_format is None:
        for fmt in session.open_command.open_data_formats:
            if content_type in fmt.mime_types:
                nominal_format = fmt
                break
        else:
            if content_type and content_type != 'application/octet-stream':
                session.logger.info('Unrecognized mime type: %s' % content_type)
            if nominal_format is None:
                from chimerax.core.errors import UserError
                raise UserError('Unable to deduce format of %s' % url)
    if ext not in nominal_format.suffixes:
        session.logger.info('mime type (%s), does not match file name extension (%s)'
            % (content_type, ext))
        new_ext = nominal_format.suffixes[0]
        new_filename = os.path.join(cache_dir, '%s%s' % (base, new_ext))
        count = 0
        while os.path.exists(new_filename):
            count += 1
            new_filename = os.path.join(cache_dir, '%s(%d)%s' % (base, count, new_ext))
        session.logger.info('renaming "%s" to "%s"' % (filename, new_filename))
        os.rename(filename, new_filename)
        filename = new_filename
    return session.open_command.open_data(filename, format=nominal_format.name, **kw)

def retrieve_url(url, filename, *, logger=None, uncompress=False,
                 update=False, check_certificates=True, name=None, timeout=60):
    """Return requested URL in filename

    :param url: the URL to retrive
    :param filename: where to save the contents of the URL
    :param name: string to use to identify the data in status messages
    :param logger: logger instance to use for status and warning messages
    :param uncompress: if true, then uncompress the content
    :param update: if true, then existing file is okay if newer than web version
    :param check_certificates: if true
    :returns: None if an existing file, otherwise the content type
    :raises urllib.request.URLError or EOFError if unsuccessful


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
        with urlopen(request, timeout=timeout) as response:
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
    except:
        if os.path.exists(filename):
            os.remove(filename)
        if logger:
            logger.status('Error fetching %s' % name, secondary=True, blank_after=15)
        raise

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

def _convert_to_timestamp(date):
    # covert HTTP date to POSIX timestamp
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date).timestamp()
    except TypeError:
        return None

def read_and_report_progress(file_in, file_out, name, content_length, logger,
        chunk_size=1048576):
    tb = 0
    while True:
        bytes = file_in.read(chunk_size)
        if bytes:
            file_out.write(bytes)
        else:
            break
        tb += len(bytes)
        if content_length:
            msg = 'Fetching %s, %.3g of %.3g Mbytes received' % (name, tb / 1048576,
                content_length / 1048576)
        else:
            msg = 'Fetching %s, %.3g Mbytes received' % (name, tb / 1048576)
        logger.status(msg)

    if content_length is not None and tb != content_length:
        # In ChimeraX bug #2747 zero bytes were read and no error reported.
        from urllib.request import URLError
        raise URLError('Got %d bytes when %d were expected' % (tb, content_length))

def read_and_uncompress(file_in, file_out, name, content_length, logger,
        chunk_size=1048576):

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

