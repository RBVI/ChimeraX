# vim: set expandtab ts=4 sw=4:

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
    """De-register MAV from alignments manager"""
    session.alignments.deregister_viewer(bundle_info)

def initialize(bundle_info, session):
    """Register MAV with alignments manager"""
    session.alignments.register_viewer(bundle_info, ["mav", "multalign"])
