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

from chimerax.core.fetch import retrieve_url


def fetch_web(session, url, ignore_cache=False, new_tab=False, data_format=None, **kw):
    # TODO: deal with content encoding for text formats
    # TODO: how would "ignore_cache" work?
    import os
    from urllib import parse
    from urllib.request import URLError
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
    try:
        content_type = retrieve_url(url, filename, logger=session.logger,
            uncompress=uncompress)
    except URLError as err:
        from chimerax.core.errors import UserError
        raise UserError(str(err))
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
