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
__version__ = "1.2"

from .bundle_builder import BundleBuilder  # noqa
from .bundle_builder_toml import Bundle as BundleBuilderTOML

# These are entry points for copying files into
# .dist-info directories of wheels when they are built

def copy_distinfo_file(cmd, basename, filename, binary=''):
    """Entry point to copy files into bundle .dist-info directory.

    File is copied as text if binary is '', and as binary if 'b'.
    """
    encoding = None if binary else 'utf-8'
    try:
        with open(basename, 'r' + binary, encoding=encoding) as fi:
            value = fi.read()
            import logging
            log = logging.getLogger()
            log.info("copying %s" % basename)
            if not cmd.dry_run:
                with open(filename, 'w' + binary, encoding=encoding) as fo:
                    fo.write(value)
    except IOError:
        # Missing file is okay
        pass


def copy_distinfo_binary_file(cmd, basename, filename):
    copy_distinfo_file(cmd, basename, filename, binary='b')
