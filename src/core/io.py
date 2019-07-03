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
    'open_filename',
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

    ..attribute:: name

        Official name for format.

    ..attribute:: category

        Type of data (STRUCTURE, SEQUENCE, etc.)

    ..attribute:: extensions

        Sequence of filename extensions in lowercase
        starting with period (or empty)

    ..attribute:: allow_directory

        Whether format can be read from a directory.

    ..attribute:: nicknames

        Alternative names for format, usually includes a short abbreviation.

    ..attribute:: mime_types

        Sequence of associated MIME types (or empty)

    ..attribute:: synopsis

        Short description of format

    ..attribute:: reference

        URL reference to specification

    ..attribute:: encoding

        None if a binary format (default), otherwise text encoding, *e.g.*, **utf-8**

    ..attribute:: dangerous

        True if can execute arbitrary code (*e.g.*, scripts)

    ..attribute:: icon

        Pathname of icon.

    ..attribute:: open_func

        Function that opens files: func(session, stream/filename)

    ..attribute:: export_func

        Function that exports files: func(stream)

    ..attribute:: export_notes

        Additional information to show in export dialogs
    """

    def __init__(self, format_name, category, extensions, nicknames, mime, reference,
                 dangerous, icon, encoding, synopsis):
        self.name = format_name
        self.category = category
        self.extensions = extensions
        self.allow_directory = False
        self.nicknames = nicknames
        self.mime_types = mime
        self.dangerous = dangerous
        self.icon = icon
        self.encoding = encoding
        self.synopsis = synopsis

        if reference:
            # sanitize URL
            from urllib import parse
            r = list(parse.urlsplit(reference))
            r[1:5] = [parse.quote(p) for p in r[1:5]]
            reference = parse.urlunsplit(r)
        self.reference = reference

        self._open_func = self._boot_open_func = None
        self.export_func = self._boot_export_func = None
        self.export_notes = None
        self.batch = False

    def has_open_func(self):
        """Test for open function without bootstrapping"""
        return (self._boot_open_func is not None or
            self._open_func is not None)

    def _get_open_func(self):
        if self._boot_open_func:
            self._open_func = self._boot_open_func()
            self._boot_open_func = None
        return self._open_func

    def _set_open_func(self, func):
        self._open_func = func
        self._boot_open_func = None

    open_func = property(_get_open_func, _set_open_func)

    def has_export_func(self):
        """Test for export function without bootstrapping"""
        return (self._boot_export_func is not None or
            self._export_func is not None)

    def _get_export_func(self):
        if self._boot_export_func:
            self._export_func = self._boot_export_func()
            self._boot_export_func = None
        return self._export_func

    def _set_export_func(self, func):
        self._export_func = func
        self._boot_export_func = None

    export_func = property(_get_export_func, _set_export_func)

    def export(self, session, path, format_name, **kw):
        from .errors import UserError
        if self.export_func is None:
            raise UserError("Save %r files is not supported" % self.name)
        import inspect
        params = inspect.signature(self.export_func).parameters
        if len(params) < 2:
            raise UserError("%s-opening function is missing mandatory session, path arguments"
                % format_name)
        if list(params.keys())[0] != "session":
            raise UserError("First param of %s-saving function is not 'session'" % self.name)
        if list(params.keys())[1] != "path":
            raise UserError("Second param of %s-saving function must be 'path'" % self.name)
        if 'format_name' in params:
            kw['format_name'] = format_name

        check_keyword_compatibility(self.export_func, session, path, **kw)
        try:
            result = self.export_func(session, path, **kw)
        except IOError as e:
            from .errors import UserError
            raise UserError(e)
        return result

_file_formats = {}


def register_format(format_name, category, extensions, nicknames=None,
                    *, mime=(), reference=None, dangerous=None, icon=None,
                    encoding=None, synopsis=None, allow_directory=None, **kw):
    """Register file format's I/O functions and meta-data

    :param format_name: format's name
    :param category: says what kind of data the should be classified as.
    :param extensions: is a sequence of filename suffixes starting
       with a period.  If the format doesn't open from a filename
       (*e.g.*, PDB ID code), then extensions should be an empty sequence.
    :param nicknames: abbreviated names for the format.  If not given,
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
    if nicknames is None:
        nicknames = (format_name.casefold(),)
    elif isinstance(nicknames, str):
        nicknames = [nicknames]
    if mime is None:
        mime = ()
    elif isinstance(mime, str):
        mime = [mime]
    if not synopsis:
        synopsis = format_name
    ff = _file_formats[format_name] = FileFormat(
        format_name, category, exts, nicknames, mime, reference, dangerous,
        icon, encoding, synopsis)
    if allow_directory is not None:
        ff.allow_directory = allow_directory
    other_kws = set(['open_func', 'export_func', 'export_notes', 'batch'])
    for attr in kw:
        if attr in other_kws:
            setattr(ff, attr, kw[attr])
        else:
            raise TypeError('Unexpected keyword argument %r' % attr)

    return ff


def deregister_format(format_name):
    try:
        del _file_formats[format_name]
    except KeyError:
        pass


def formats(open=True, export=True, source_is_file=False):
    """Returns list of known formats."""
    fmts = []
    for f in _file_formats.values():
        if source_is_file and not f.extensions:
            continue
        if (open and f.has_open_func()) or (export and f.has_export_func()):
            fmts.append(f)
    return fmts


def format_from_name(name):
    try:
        return _file_formats[name]
    except KeyError:
        return None


def deduce_format(filename, has_format=None, open=True, save=False, no_raise=False):
    """Figure out named format associated with filename.
    If open is True then the format must have an open method.
    If save is True then the format must have a save method.

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
                if has_format in f.nicknames and (not open or f.has_open_func()) and (not save or f.has_export_func()):
                    fmt = f
                    break
        stripped, compression = determine_compression(filename)
    elif format_name is None:
        stripped, compression = determine_compression(filename)
        import os
        base, ext = os.path.splitext(stripped)
        if not ext:
            if no_raise:
                return None, filename, compression
            from .errors import UserError
            raise UserError("Missing filename suffix %s" % filename)
        ext = ext.casefold()
        fmt = None
        for f in _file_formats.values():
            if ext in f.extensions and (not open or f.has_open_func()) and (not save or f.has_export_func()):
                fmt = f
                break
        if fmt is None:
            if no_raise:
                return None, filename, compression
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
            o = 'o' if info.has_open_func() else ' '
            e = 's' if info.has_export_func() else ' '
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
    import inspect
    params = inspect.signature(open_func).parameters
    if len(params) < 2:
        raise UserError("%s-opening function is missing mandatory session, path/stream arguments"
            % fmt.name)
    if list(params.keys())[0] != "session":
        raise UserError("First param of %s-opening function is not 'session'" % fmt.name)
    if list(params.keys())[1] not in ("path", "stream"):
        raise UserError("Second param of %s-opening function must be 'path' or 'stream'" % fmt.name)
    import os.path
    filename = os.path.expanduser(os.path.expandvars(filename))
    delete_when_done = False
    provide_stream = list(params.keys())[1] == "stream"
    if provide_stream:
        enc = fmt.encoding
        if enc is None:
            mode = 'rb'
        else:
            mode = 'rt'
        filename, dname, stream = _compressed_open(filename, compression, mode, encoding=enc)
        args = (session, stream)
    elif compression:
        # need to use temporary file
        import tempfile
        filename, dname, stream = _compressed_open(filename, compression, 'rb')
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(stream.read())
            tf.close()
        filename = tf.name
        delete_when_done = True
        args = (session, filename)
    else:
        args = (session, filename)
        dname = os.path.basename(filename)

    if 'format_name' in params:
        kw['format_name'] = fmt.nicknames[0]
    if 'file_name' in params:
        kw['file_name'] = dname

    try:
        if fmt.category == SCRIPT:
            with session.in_script:
                models, status = open_func(*args, **kw)
        else:
            models, status = open_func(*args, **kw)
    finally:
        if provide_stream and not stream.closed:
            stream.close()
        if delete_when_done:
            try:
                os.remove(filename)
            except OSError:
                pass

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
    for fmt, paths in batch.items():
        mname = model_name_from_path(paths[0]) if name is None else name
        open_func = fmt.open_func
        models, status = open_func(session, paths, mname, **kw)
        mlist.extend(models)
        status_lines.append(status)
    for fspec in unbatched:
        models, status = open_data(session, fspec, format=format, name=name, **kw)
        mlist.extend(models)
        status_lines.append(status)

    return mlist, '\n'.join(status_lines)

def model_name_from_path(path):
    from os.path import basename, dirname
    name = basename(path)
    if name.strip() == '':
        # Path is a directory with trailing "/".  Use directory name.
        name = basename(dirname(path))
    return name

def export(session, filename, **kw):
    from .safesave import SaveBinaryFile, SaveTextFile, SaveFile
    fmt, filename, compression = deduce_format(filename, save = True, open = False)
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
    """Supported API. Open a file/URL with or without compression

    Takes the same arguments as built-in open and returns a file-like
    object.  However, `filename` can also be a file-like object itself,
    in which case it is simply returned.  Also, if `filename` is a string
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

    if filename.startswith(("http:", "https:")):
        from urllib.request import urlopen
        return urlopen(filename)

    stripped, compression = determine_compression(filename)
    import os.path
    filename = os.path.expanduser(os.path.expandvars(filename))
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


def check_keyword_compatibility(f, *args, **kw):
    import inspect
    sig = inspect.signature(f)
    # If function takes arbitrary keywords, it is compatible
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return
    # If we cannot bind the arguments, raise TypeError and
    # let caller handle it
    sig.bind(*args, **kw)
