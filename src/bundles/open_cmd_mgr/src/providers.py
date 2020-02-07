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

def get_compression(name, path, encoding):
    if name == "gzip":
        from gzip import open as open_compressed
    elif name == "bz2":
        from bz2 import open as open_compressed
    elif name == "lzma":
        from lzma import open as open_compressed
    else:
        raise ValueError("Don't know how to handle compression type '%s'" % name)
    if encoding:
        return open_compressed(path, mode='r', encoding=encoding)
    return open_compressed(path)

