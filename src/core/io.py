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

"""
io: Manage file formats that can be opened and saved
=====================================================

The io module keeps track of the functions that can open and save data
in various formats.

I/O sources and destinations are specified as filenames, and the appropriate
open or export function is found by deducing the format from the suffix of the filename.
An additional compression suffix, *i.e.*, ``.gz``,
indicates that the file is or should be compressed.

All data I/O is in binary.
"""

__all__ = [
    'register_format',
    'register_compression',
    'open_data',
    'open_multiple_data',
    'formats',
    'format_from_name',
    'deduce_format',
    'compression_suffixes',
]

_compression = {}


def register_compression(suffix, stream_type):
    _compression[suffix] = stream_type


def _init_compression():
    try:
        import gzip
        register_compression('.gz', gzip.open)
    except ImportError:
        pass
    try:
        import bz2
        register_compression('.bz2', bz2.open)
    except ImportError:
        pass
    try:
        import lzma
        register_compression('.xz', lzma.open)
    except ImportError:
        pass
_init_compression()


def compression_suffixes():
    return _compression.keys()

# some well known file format categories
SCRIPT = "Command script"


class FileFormat:
    """Keep tract of information about various data sources

    ..attribute:: category

        Type of data (STRUCTURE, SEQUENCE, etc.)

    ..attribute:: extensions

        Sequence of filename extensions in lowercase
        starting with period (or empty)

    ..attribute:: short_names

        Short names for format.

    ..attribute:: mime_types

        Sequence of associated MIME types (or empty)

    ..attribute:: reference

        URL reference to specification

    ..attribute:: encoding

        None if a binary format (default), otherwise text encoding, *e.g.*, **utf-8**

    ..attribute:: dangerous

        True if can execute arbitrary code (*e.g.*, scripts)

    ..attribute:: icon

        Pathname of icon.

    ..attribute:: open_func

        Function that opens files: func(session, stream/filename, name=None)

    ..attribute:: requires_filename

        True if open function a filename

    ..attribute:: export_func

        Function that exports files: func(stream)

    ..attribute:: export_notes

        Additional information to show in export dialogs
    """

    def __init__(self, format_name, category, extensions, short_names, mime, reference,
                 dangerous, icon, encoding):
        self.name = format_name
        self.category = category
        self.extensions = extensions
        self.short_names = short_names
        self.mime_types = mime
        self.reference = reference
        self.dangerous = dangerous
        self.icon = icon
        self.encoding = encoding

        self.open_func = None
        self.requires_filename = False
        self.export_func = None
        self.export_notes = None
        self.batch = False

_file_formats = {}


def register_format(format_name, category, extensions, short_names=None,
                    *, mime=(), reference=None, dangerous=None, icon=None,
                    encoding=None, **kw):
    """Register file format's I/O functions and meta-data

    :param format_name: format's name
    :param category: says what kind of data the should be classified as.
    :param extensions: is a sequence of filename suffixes starting
       with a period.  If the format doesn't open from a filename
       (*e.g.*, PDB ID code), then extensions should be an empty sequence.
    :param short_names: abbreviated names for the format.  If not given,
       it defaults to a lowercase version of the format name.
    :param mime: is a sequence of mime types, possibly empty.
    :param reference: a URL link to the specification.
    :param dangerous: should be True for formats that can write/delete
       a users's files.  False by default except for the SCRIPT category.

    .. todo::
        possibly break up in to multiple functions
    """
    if dangerous is None:
        # scripts are inherently dangerous
        dangerous = category == SCRIPT
    if extensions is not None:
        if isinstance(extensions, str):
            extensions = [extensions]
        exts = [s.lower() for s in extensions]
    else:
        exts = ()
    if short_names is None:
        short_names = (format_name.casefold(),)
    elif isinstance(short_names, str):
        short_names = [short_names]
    if mime is None:
        mime = ()
    elif isinstance(mime, str):
        mime = [mime]
    ff = _file_formats[format_name] = FileFormat(format_name,
        category, exts, short_names, mime, reference, dangerous, icon, encoding)
    other_kws = set(['open_func', 'requires_filename',
                     'export_func', 'export_notes', 'batch'])
    for attr in kw:
        if attr in other_kws:
            setattr(ff, attr, kw[attr])
        else:
            raise TypeError('Unexpected keyword argument %r' % attr)

    return ff


def formats(open=True, export=True, source_is_file=False):
    """Returns list of known formats."""
    fmts = []
    for f in _file_formats.values():
        if source_is_file and not f.extensions:
            continue
        if (open and f.open_func) or (export and f.export_func):
            fmts.append(f)
    return fmts


def format_from_name(name):
    for f in _file_formats.values():
        if f.name == name:
            return f
    return None


def deduce_format(filename, has_format=None, savable=False):
    """Figure out named format associated with filename

    Return tuple of deduced format, the unmangled filename,
    and the compression format (if present).
    """
    format_name = None
    compression = None
    if has_format:
        # Allow has_format to be a file type prefix.
        fmt = _file_formats.get(has_format, None)
        if fmt is None:
            for f in _file_formats.values():
                if has_format in f.short_names and (f.open_func or (savable and f.export_func)):
                    fmt = f
                    break
        stripped, compression = determine_compression(filename)
    elif format_name is None:
        stripped, compression = determine_compression(filename)
        import os
        base, ext = os.path.splitext(stripped)
        if not ext:
            from .errors import UserError
            raise UserError("Missing filename suffix %s" % filename)
        ext = ext.casefold()
        fmt = None
        for f in _file_formats.values():
            if ext in f.extensions and (f.open_func or (savable and f.export_func)):
                fmt = f
                break
        if fmt is None:
            from .errors import UserError
            raise UserError("Unrecognized file suffix '%s'" % ext)
    return fmt, filename, compression


def print_file_suffixes():
    """Print user-friendly list of supported files suffixes"""

    combine = {}
    for format_name, info in _file_formats.items():
        names = combine.setdefault(info.category, [])
        names.append(format_name)
    categories = list(combine)
    categories.sort(key=str.casefold)
    print('Supported file suffixes:')
    print('  o = open, s = save')
    for k in categories:
        print("\n%s:" % k)
        names = combine[k]
        names.sort(key=str.casefold)
        for format_name in names:
            info = _file_formats[format_name]
            o = 'o' if info.open_func else ' '
            e = 's' if info.export_func else ' '
            if info.extensions:
                exts = ': ' + ', '.join(info.extensions)
            else:
                exts = ''
            print('%c%c  %s%s' % (o, e, format_name, exts))

    # if _compression:
    #    for ext in combine[k]:
    #        fmts += ';' + ';'.join('*%s%s' % (ext, c)
    #                               for c in _compression.keys())


def open_data(session, filespec, format=None, name=None, **kw):
    """open a (compressed) file

    :param filespec: filename
    :param format: file as if it has the given format
    :param name: optional name used to identify data source

    If a file format requires a filename, then compressed files are
    uncompressed into a temporary file before calling the open function.
    """

    from .errors import UserError
    fmt, filename, compression = deduce_format(filespec, has_format=format)
    open_func = fmt.open_func
    if open_func is None:
        raise UserError("unable to open %s files" % fmt.name)
    enc = fmt.encoding
    if enc is None:
        mode = 'rb'
    else:
        mode = 'rt'
    filename, dname, stream = _compressed_open(filename, compression, mode, encoding=enc)
    if fmt.requires_filename and not filename:
        # copy compressed file to real file
        import tempfile
        exts = fmt.extensions
        suffix = exts[0] if exts else ''
        tf = tempfile.NamedTemporaryFile(prefix='chtmp', suffix=suffix)
        while 1:
            data = stream.read()
            if not data:
                break
            tf.write(data)
        tf.seek(0)
        stream = tf
        # TODO: Windows might need tf to be closed before reading with
        # a different file descriptor

    kw["filespec"] = filename
    if fmt.category == SCRIPT:
        with session.in_script:
            models, status = open_func(session, stream, dname, **kw)
    else:
        models, status = open_func(session, stream, dname, **kw)

    if not stream.closed:
        stream.close()

    if name is not None:
        for m in models:
            m.name = name

    return models, status


def open_multiple_data(session, filespecs, format=None, name=None, **kw):
    '''Open one or more files, including handling formats where multiple files
    contribute to a single model, such as image stacks.'''
    if isinstance(filespecs, str):
        filespecs = [filespecs]

    batch = {}
    unbatched = []
    for filespec in filespecs:
        fmt, filename, compression = deduce_format(filespec, has_format=format)
        if fmt is not None and fmt.batch:
            if fmt in batch:
                batch[fmt].append(filespec)
            else:
                batch[fmt] = [filespec]
        else:
            unbatched.append(filespec)

    mlist = []
    status_lines = []
    import os.path
    for fmt, paths in batch.items():
        name = os.path.basename(paths[0]) if name is None else name
        open_func = fmt.open_func
        models, status = open_func(session, paths, name)
        mlist.extend(models)
        status_lines.append(status)
    for fspec in unbatched:
        models, status = open_data(session, fspec, format=format, name=name, **kw)
        mlist.extend(models)
        status_lines.append(status)

    return mlist, '\n'.join(status_lines)


def export(session, filename, **kw):
    from .safesave import SaveBinaryFile, SaveTextFile, SaveFile
    fmt, filename, compression = deduce_format(filename)
    func = fmt.export_func
    enc = fmt.encoding
    if not compression:
        if enc is None:
            with SaveBinaryFile(filename) as stream:
                return func(session, stream, **kw)
        else:
            with SaveTextFile(filename) as stream:
                return func(session, stream, encoding=enc, **kw)
    else:
        stream_type = _compression[compression]

        if enc is None:
            mode = 'wb'
        else:
            mode = 'wt'

        def open_compressed(filename):
            return stream_type(filename, mode=mode, encoding=enc)
        with SaveFile(filename, open=open_compressed) as stream:
            return func(session, stream, **kw)


def determine_compression(filename):
    """Check file name for compression suffixes

    Returns the file name with the compression suffix if any stripped, and
    the compression suffix (None if no compression suffix).
    """
    for compression in compression_suffixes():
        if filename.endswith(compression):
            stripped = filename[:-len(compression)]
            break
    else:
        stripped = filename
        compression = None
    return stripped, compression


def _compressed_open(filename, compression, *args, **kw):
    import os.path
    from .errors import UserError
    filename = os.path.expanduser(os.path.expandvars(filename))
    if not compression:
        try:
            stream = open(filename, *args, **kw)
            name = os.path.basename(filename)
        except OSError as e:
            raise UserError(e)
    else:
        stream_type = _compression[compression]
        try:
            stream = stream_type(filename, *args, **kw)
            name = os.path.basename(os.path.splitext(filename)[0])
            filename = None
        except OSError as e:
            raise UserError(e)

    return filename, name, stream


def open_filename(filename, *args, **kw):
    """Open a file/URL with or without compression

    Takes the same arguments as built-in open and returns a file-like
    object.  However, `filename` can also be a file-like object itself,
    in which case it is simple returned.  Also, if `filename` is a string
    that begins with "http:", then it is interpreted as an URL.

    If the file is opened for input, compression is checked for and
    handled automatically.  If the file is opened for output, the `compress`
    keyword can be used to force or suppress compression.  If the keyword
    is not given, then compression will occur if the file name ends in
    the appropriate suffix (*e.g.* '.gz').  If compressing, you can supply
    `args` compatible with the appropriate "open" function (*e.g.* gzip.open).

    '~' is expanded unless the `expand_user` keyword is specified as False.

    Uncompressed non-binary files will be opened for reading with universal
    newline support.
    """

    if not isinstance(filename, str):
        # a "file-like" object -- just return it after making sure
        # that .close() will work
        if not hasattr(filename, "close") or not callable(filename.close):
            filename.close = lambda: False
        return filename

    if filename.startswith("http:"):
        from urllib.request import urlopen
        return urlopen(filename)

    stripped, compression = determine_compression(filename)
    path, fname, stream = _compressed_open(filename, compression, *args, **kw)
    return stream


def gunzip(gzpath, path, remove_gz=True):

    import gzip
    gzf = gzip.open(gzpath)
    import builtins
    f = builtins.open(path, 'wb')
    f.write(gzf.read())
    f.close()
    gzf.close()
    if remove_gz:
        import os
        os.remove(gzpath)
