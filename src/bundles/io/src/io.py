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

from .compression import handle_compression, get_compression_type

def open_input(source, encoding=None, *, compression=None):
    """
    Open possibly compressed input for reading.

    *source* can be path or a stream.  If a stream, it is simply returned.
    If *encoding* is 'None', open as binary.
    If *compression* is None, whether to use compression and what type will be determined off the file name.

    Also, if *source* is a string that begins with "http:" or "https:", then it is interpreted as an URL.
    The encoding of the data returned by the URL is attempted to be determined by examining
    Content-Encoding and/or Content-Type headers, but if those are missing then *encoding* is used
    instead (binary if *encoding* is None).
    """

    if _is_stream(source):
        return source

    if source.startswith(("http:", "https:")):
        from urllib.request import urlopen
        result = urlopen(source)
        content_encoding = result.getheader('Content-Encoding')
        if content_encoding:
            encoding = content_encoding
        else:
            content_type = result.getheader('Content-Type')
            if content_type:
                if 'text' in content_type and 'charset=' in content_type:
                    encoding = content_type.split('charset=')[-1]
        if encoding:
            from io import StringIO
            return StringIO(result.read().decode(encoding))
        return result

    mode = 'rt' if encoding else 'rb'
    fs_source = file_system_file_name(source)
    compression_type = get_compression_type(fs_source, compression)
    if compression_type:
        return handle_compression(compression_type, fs_source,
            mode=mode, encoding=encoding)
    return open(fs_source, mode, encoding=encoding)

def open_output(output, encoding=None, *, append=False, compression=None):
    """
    Open output for (possibly compressed) writing.

    *output* can be path or a stream.  If a stream, it is simply returned.
    If *encoding* is 'None', open as binary.
    If *compression* is None, whether to use compression and what type will be determined off the file name.
    """
    if _is_stream(output):
        return output
    fs_output = file_system_file_name(output)
    compression_type = get_compression_type(fs_output, compression)
    base_mode = 'a' if append else 'w'
    mode = base_mode + ('t' if encoding else 'b')
    if compression_type:
        return handle_compression(compression_type, fs_output,
            mode=mode, encoding=encoding)
    return open(fs_output, mode, encoding=encoding)

def file_system_file_name(file_name):
    import os.path
    file_name = os.path.expanduser(file_name)
    try:
        hash_pos = file_name.rindex('#')
        dot_pos = file_name.rindex('.')
    except ValueError:
        return file_name

    # get real file name for file.html#anchor
    if dot_pos < hash_pos:
        return file_name[:hash_pos]
    return file_name

def _is_stream(source):
    if isinstance(source, str):
        return False
    # ensure that 'close' works on the stream...
    if not hasattr(source, "close") or not callable(source.close):
        source.close = lambda: False
    return True
