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

class UnwantedCompressionError(IOError):
    pass

suffix_to_type = {
    ".bz2": "bz2",
    ".gz": "gzip",
    ".xz": "lzma"
}

def get_compression_type(file_name, requested_compression):
    for suffix, comp_type in suffix_to_type.items():
        if requested_compression:
            if requested_compression == comp_type:
                return comp_type
        elif file_name.endswith(suffix):
            if requested_compression is False:
                raise UnwantedCompressionError("Cannot handled compressed files")
            return comp_type
    if requested_compression:
        raise ValueError("Don't know requested compression type '%s'; known types are:"
            " %s" % (requested_compression,
            ", ".join([t for t in suffix_to_type.values()])))
    return None

def handle_compression(name, path, **kw):
    if name == "gzip":
        from gzip import open as open_compressed
    elif name == "bz2":
        from bz2 import open as open_compressed
    elif name == "lzma":
        from lzma import open as open_compressed
    else:
        raise ValueError("Don't know how to handle compression type '%s'" % name)
    stream = open_compressed(path, **kw)
    # since these return the fileno of the compressed file, mark them as compression
    # streams so that the PDB reader knows not to try to use the fileno!
    stream.from_compressed_source = True
    return stream

def remove_compression_suffix(file_name):
    for suffix in suffix_to_type.keys():
        if file_name.endswith(suffix):
            file_name = file_name[:-len(suffix)]
            break
    return file_name
