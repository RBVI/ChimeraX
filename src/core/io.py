# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
io: manage file formats that can be opened and exported
=======================================================

The io module keeps track of the functions that can open, fetch, and export
data in various formats.

I/O sources and destinations are specified as filenames, and the appropriate
open or export function is found by deducing the format from the suffix of the
filename.
An additional compression suffix, *i.e.*, ``.gz``,
indicates that the file is or should be compressed.
In addition to reading data from files,
data can be fetched from the Internet.
In that case, instead of a filename,
the data source is specified as prefix:identifier, *e.g.*, ``pdb:1gcn``, where
the prefix identifies the data format, and the identifier selects the data.

All data I/O is in binary.
"""

__all__ = [
    'register_format',
    'register_open',
    'register_fetch',
    'register_export',
    'register_compression',
    'SCRIPT',
    'formats',
    'open',
    'prefixes',
    'extensions',
    'open_function',
    'fetch_function',
    'export_function',
    'mime_types',
    'requires_filename',
    'dangerous',
    'category',
    'format_names',
    'categorized_formats',
    'deduce_format',
    'compression_suffixes',
    'DYNAMICS',
    'GENERIC3D',
    'SCRIPT',
    'SEQUENCE',
    'SESSION',
    'STRUCTURE',
    'SURFACE',
    'VOLUME',
]

# Known I/O format catagories
DYNAMICS = "Molecular trajectory"
GENERIC3D = "Generic 3D objects"
SCRIPT = "Command script"
SEQUENCE = "Sequence alignment"
SESSION = "Session data"
STRUCTURE = "Molecular structure"
SURFACE = "Molecular surface"
VOLUME = "Volume data"

_compression = {}


def register_compression(suffix, stream_type):
    _compression[suffix] = stream_type


def _init_compression():
    try:
        import gzip
        register_compression('.gz', gzip.GzipFile)
    except ImportError:
        pass
    try:
        import bz2
        register_compression('.bz2', bz2.BZ2File)
    except ImportError:
        pass
    try:
        import lzma
        register_compression('.xz', lzma.LZMAFile)
    except ImportError:
        pass
_init_compression()


def compression_suffixes():
    return _compression.keys()

# some well known file format categories
SCRIPT = "Command script"


class _FileFormatInfo:
    """Keep tract of information about various data sources

    ..attribute:: category

        Type of data (STRUCTURE, SEQUENCE, etc.)

    ..attribute:: extensions

        Sequence of filename extensions in lowercase
        starting with period (or empty)

    ..attribute:: prefixes

        sequence of URL-style prefixes (or empty)

    ..attribute:: mime_types

        sequence of associated MIME types (or empty)

    ..attribute:: reference

        URL reference to specification

    ..attribute:: dangerous

        True if can execute arbitrary code (*e.g.*, scripts)

    ..attribute:: open_func

        function that opens files: func(session, stream/filename, name=None)

    ..attribute:: requires_filename

        True if open function a filename

    ..attribute:: fetch_func

        function that opens internet files:
            func(prefixed_name, identifier=None)

    ..attribute:: export_func

        function that exports files: func(stream)

    ..attribute:: export_notes

        additional information to show in export dialogs
    """

    def __init__(self, category, extensions, prefixes, mime, reference,
                 dangerous):
        self.category = category
        self.extensions = extensions
        self.prefixes = prefixes
        self.mime_types = mime
        self.reference = reference
        self.dangerous = dangerous

        self.open_func = None
        self.requires_filename = False
        self.fetch_func = None
        self.export_func = None
        self.export_notes = None
        self.batch = False

_file_formats = {}


def register_format(format_name, category, extensions, prefixes=(), mime=(),
                    reference=None, dangerous=None, **kw):
    """Register file format's I/O functions and meta-data

    :param format_name: format's name
    :param category: says what kind of data the should be classified as.
    :param extensions: is a sequence of filename suffixes starting
       with a period.  If the format doesn't open from a filename
       (*e.g.*, PDB ID code), then extensions should be an empty sequence.
    :param prefixes: is a sequence of filename prefixes (no ':'),
       possibily empty.
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
    if prefixes is None:
        prefixes = ()
    elif isinstance(prefixes, str):
        prefixes = [prefixes]
    if prefixes and not fetch_function:
        import sys
        print("missing fetch function for format with prefix support:",
              format_name, file=sys.stderr)
    if mime is None:
        mime = ()
    elif isinstance(mime, str):
        mime = [mime]
    ff = _file_formats[format_name] = _FileFormatInfo(category, exts, prefixes,
                                                      mime, reference,
                                                      dangerous)
    for attr in ['open_func', 'requires_filename', 'fetch_func',
                 'export_func', 'export_notes', 'batch']:
        if attr in kw:
            setattr(ff, attr, kw[attr])


def formats():
    """Return all known format names"""
    return list(_file_formats.keys())


def prefixes(format_name):
    """Return filename prefixes for named format.

    prefixes(format_name) -> [filename-prefix(es)]
    """
    try:
        return _file_formats[format_name].prefixes
    except KeyError:
        return ()


def register_open(format_name, open_function, requires_filename=False):
    """register a function that reads data from a stream

    :param open_function: function taking an I/O stream or filename and
        returns a 2-tuple with a list of models and a status message
    :param requires_filename: True if first argument must be a filename
    """
    try:
        fi = _file_formats[format_name]
    except KeyError:
        raise ValueError("Unknown data type")
    fi.open_func = open_function
    fi.requires_filename = requires_filename


def register_fetch(format_name, fetch_function):
    """register a function that fetches data from the Internet

    :param fetch_fuction: function that takes an identifier,
        and returns an I/O stream for reading data, and identifying name.
        Usually the name is the same as the identifier.
    """
    try:
        fi = _file_formats[format_name]
    except KeyError:
        raise ValueError("Unknown data type")
    fi.fetch_func = fetch_function


def register_export(format_name, export_function, export_notes=''):
    try:
        fi = _file_formats[format_name]
    except KeyError:
        raise ValueError("Unknown data type")
    fi.export_func = export_function
    fi.export_notes = export_notes


def extensions(format_name):
    """Return filename extensions for named format.

    extensions(format_name) -> [filename-extension(s)]
    """
    try:
        exts = _file_formats[format_name].extensions
    except KeyError:
        return ()
    return exts


def open_function(format_name):
    """Return open callback for named format.

    open_function(format_name) -> function
    """
    try:
        return _file_formats[format_name].open_func
    except KeyError:
        return None


def fetch_function(format_name):
    """Return fetch callback for named format.

    fetch_function(format_name) -> function
    """
    try:
        return _file_formats[format_name].fetch_func
    except KeyError:
        return None


def export_function(format_name):
    """Return export callback for named format.

    export_function(format_name) -> function
    """
    try:
        return _file_formats[format_name].export_func
    except KeyError:
        return None


def mime_types(format_name):
    """Return mime types for named format."""
    try:
        return _file_formats[format_name].mime_types
    except KeyError:
        return None


def requires_filename(format_name):
    """Return whether named format can needs a seekable file"""
    try:
        return _file_formats[format_name].requires_filename
    except KeyError:
        return False


def batched_format(format_name):
    """Return whether format reader opens wants all paths as a group,
    such as reading image stacks as volumes."""
    try:
        return _file_formats[format_name].batch
    except KeyError:
        return False


def dangerous(format_name):
    """Return whether named format can write to files"""
    try:
        return _file_formats[format_name].dangerous
    except KeyError:
        return False


def category(format_name):
    """Return category of named format"""
    try:
        return _file_formats[format_name].category
    except KeyError:
        return "Unknown"


def format_names(open=True, export=False, source_is_file=False):
    """Return known format names.

    formats() -> [format-name(s)]
    """
    names = []
    for t, info in _file_formats.items():
        if open and not info.open_func:
            continue
        if export and not info.export_func:
            continue
        if not source_is_file or info.extensions:
            names.append(t)
    return names


def categorized_formats(open=True, export=False):
    """Return known formats by category

    categorized_formats() -> { category: formats() }
    """
    result = {}
    for format_name, info in _file_formats.items():
        if open and not info.open_func:
            continue
        if export and not info.export_func:
            continue
        names = result.setdefault(info.category, [])
        names.append(format_name)
    return result


def deduce_format(filename, has_format=None, prefixable=True):
    """Figure out named format associated with filename

    Return tuple of deduced format name, whether it was a prefix
    reference, the unmangled filename, and the compression format
    (if present).  If it is a prefix reference, then it needs to
    be fetched.
    """
    format_name = None
    prefixed = False
    compression = None
    if has_format:
        format_name = has_format
        import os
        for compression in compression_suffixes():
            if filename.endswith(compression):
                stripped = filename[:-len(compression)]
                break
        else:
            compression = None
    elif (prefixable and len(filename) >= 2 and ':' in filename and
            filename[1] != ':'):
        # format may be specified as colon-separated prefix
        # ignoring Windows drive letters
        prefix, fname = filename.split(':', 1)
        prefix = prefix.casefold()
        for t, info in _file_formats.items():
            if prefix in info.prefixes:
                format_name = t
                filename = fname
                prefixed = True
                break
        if format_name is None:
            from .cli import UserError
            raise ValueError("'%s' is not a not prefix" % prefix)
    elif format_name is None:
        import os
        for compression in compression_suffixes():
            if filename.endswith(compression):
                stripped = filename[:-len(compression)]
                break
        else:
            stripped = filename
            compression = None
        base, ext = os.path.splitext(stripped)
        if not ext:
            from .cli import UserError
            raise UserError("Missing filename suffix")
        ext = ext.casefold()
        for t, info in _file_formats.items():
            if ext in info.extensions:
                format_name = t
                break
        if format_name is None:
            from .cli import UserError
            raise UserError("Unrecognized filename suffix")
    return format_name, prefixed, filename, compression


def print_file_types():
    """Return file name filter suitable for Open File dialog for WX"""

    combine = {}
    for format_name, info in _file_formats.items():
        names = combine.setdefault(info.category, [])
        names.append(format_name)
    categories = list(combine)
    categories.sort(key=str.casefold)
    print('Supported file types:')
    print('  o = open, e = export')
    for k in categories:
        print("\n%s:" % k)
        names = combine[k]
        names.sort(key=str.casefold)
        for format_name in names:
            info = _file_formats[format_name]
            o = 'o' if info.open_func else ' '
            e = 'e' if info.export_func else ' '
            if info.extensions:
                exts = ': ' + ', '.join(info.extensions)
            else:
                exts = ''
            print('%c%c  %s%s' % (o, e, format_name, exts))

    # if _compression:
    #    for ext in combine[k]:
    #        fmts += ';' + ';'.join('*%s%s' % (ext, c)
    #                               for c in _compression.keys())


def wx_export_file_filter(category=None, all=False):
    """Return file name filter suitable for Export File dialog for WX"""

    result = []
    for t, info in _file_formats.items():
        if not info.export_func:
            continue
        if category and info.category != category:
            continue
        exts = ', '.join(info.extensions)
        fmts = ';'.join('*%s' % ext for ext in info.extensions)
        result.append("%s files (%s)|%s" % (t, exts, fmts))
    if all:
        result.append("All files (*.*)|*.*")
    if not result:
        if not category:
            files = "any"
        else:
            files = "\"%s\"" % category
        raise ValueError("No filters for %s files" % files)
    result.sort(key=str.casefold)
    return '|'.join(result)


def wx_open_file_filter(all=False):
    """Return file name filter suitable for Open File dialog for WX"""

    combine = {}
    for t, info in _file_formats.items():
        if not info.open_func:
            continue
        exts = combine.setdefault(info.category, [])
        exts.extend(info.extensions)
    result = []
    for k in combine:
        exts = ', '.join(combine[k])
        fmts = ';'.join('*%s' % ext for ext in combine[k])
        if _compression:
            for ext in combine[k]:
                fmts += ';' + ';'.join('*%s%s' % (ext, c)
                                       for c in _compression.keys())
        result.append("%s files (%s)|%s" % (k, exts, fmts))
    result.sort(key=str.casefold)
    if all:
        result.insert(0, "All files (*.*)|*.*")
    return '|'.join(result)


_builtin_open = open


def open(session, filespec, as_a=None, label=None, **kw):
    """open a (compressed) file

    :param filespec: '''prefix:id''' or a (compressed) filename
    :param as_a: file as if it has the given format
    :param label: optional name used to identify data source

    If a file format requires a filename, then compressed files are
    uncompressed into a temporary file before calling the open function.
    """

    from chimera.core.cli import UserError
    format_name, prefix, filename, compression = deduce_format(
        filespec, has_format=as_a)
    open_func = open_function(format_name)
    if open_func is None:
        raise UserError("unable to open %s files" % format_name)
    if prefix:
        fetch_func = fetch_function(format_name)
        if fetch_func is None:
            raise UserError("unable to fetch %s files" % format_name)
        stream, name = fetch_func(session, filename)
        if hasattr(filename, 'read'):
            filename = None
        else:
            filename = stream
            stream = _builtin_open(filename, 'rb')
    else:
        import os.path
        if not compression:
            filename = os.path.expanduser(os.path.expandvars(filename))
            try:
                stream = _builtin_open(filename, 'rb')
                name = os.path.basename(filename)
            except OSError as e:
                raise UserError(e)
        else:
            stream_type = _compression[compression]
            try:
                stream = stream_type(filename)
                name = os.path.basename(os.path.splitext(filename)[0])
                filename = None
            except OSError as e:
                raise UserError(e)
    if requires_filename(format_name) and not filename:
        # copy compressed file to real file
        import tempfile
        exts = extensions(format_name)
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
    models, status = open_func(session, stream, name, **kw)
    if label is not None:
        for m in models:
            m.name = label
    return models, status


def open_multiple(session, filespecs, **kw):
    '''Open one or more files, including handling formats where multiple files
    contribute to a single model, such as image stacks.'''
    if isinstance(filespecs, str):
        filespecs = [filespecs]

    batch = {}
    unbatched = []
    for filespec in filespecs:
        format_name, prefix, filename, compression = deduce_format(filespec)
        if format_name is not None and batched_format(format_name):
            if format_name in batch:
                batch[format_name].append(filespec)
            else:
                batch[format_name] = [filespec]
        else:
            unbatched.append(filespec)

    mlist = []
    status = None
    import os.path
    for format_name, paths in batch.items():
        name = os.path.basename(paths[0])
        open_func = open_function(format_name)
        models, status = open_func(session, paths, name)
        mlist.extend(models)
    for fspec in unbatched:
        models, status = open(session, fspec, **kw)
        mlist.extend(models)

    return mlist, status


def export(session, filename, **kw):
    from .safesave import SaveBinaryFile, SaveFile
    format_name, prefix, filename, compression = deduce_format(
        filename, prefixable=False)
    func = export_function(format_name)
    if not compression:
        with SaveBinaryFile(filename) as stream:
            return func(session, stream, **kw)
    else:
        stream_type = _compression[compression]
        open_compressed = lambda filename: stream_type(filename, 'wb')
        with SaveFile(filename, open=open_compressed) as stream:
            return func(session, stream, **kw)


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
