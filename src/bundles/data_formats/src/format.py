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

"""TODO: handled by 'open_data' manager
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
"""

from chimerax.core.state import State
class DataFormat(State):
    """Keep tract of information about various data sources

    ..attribute:: name

        Official name for format.

    ..attribute:: category

        Type of data (STRUCTURE, SEQUENCE, etc.)

    ..attribute:: suffixes

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

    ..attribute:: reference_url

        URL reference to specification

    ..attribute:: encoding

        None if a binary format (default), otherwise text encoding, *e.g.*, **utf-8**

    ..attribute:: insecure

        True if can execute arbitrary code (*e.g.*, scripts)

    ..attribute:: allow_directory

        True if the format is opened/saved as a directory.  This is the only case where
        'suffixes' can be empty.

    """
    attr_names = ['name', 'category', 'suffixes', 'nicknames', 'mime_types', 'reference_url', 'insecure',
        'encoding', 'synopsis', 'allow_directory']

    def __init__(self, format_name, category, suffixes, nicknames, mime_types,
            reference_url, insecure, encoding, synopsis, allow_directory):
        self.name = format_name
        self.category = category
        self.suffixes = suffixes
        self.nicknames = nicknames
        self.mime_types = mime_types
        self.insecure = insecure
        self.encoding = encoding
        self.synopsis = synopsis if synopsis else format_name
        self.allow_directory = allow_directory

        if reference_url:
            # sanitize URL
            from urllib import parse
            r = list(parse.urlsplit(reference_url))
            r[1:5] = [parse.quote(p) for p in r[1:5]]
            reference_url = parse.urlunsplit(r)
        self.reference_url = reference_url

    def take_snapshot(self, session, flags):
        return { attr_name: getattr(self, attr_name) for attr_name in self.attr_names }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        return class_obj(*[data[attr_name] for attr_name in class_obj.attr_names])

