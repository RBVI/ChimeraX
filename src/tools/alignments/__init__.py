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

def finish(bundle_info, session):
    """De-install alignments manager from existing session"""
    del session.alignments

def get_class(class_name):
    if class_name == "AlignmentsManager":
        from . import manager
        return manager.AlignmentsManager

def initialize(bundle_info, session):
    """Install alignments manager into existing session"""
    from . import settings
    settings.init(session)

    from .manager import AlignmentsManager
    session.alignments = AlignmentsManager(session, bundle_info)

from .parse import open_file
