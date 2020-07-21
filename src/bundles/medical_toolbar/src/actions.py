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

def run_provider(session, name):
    # run shortcut chosen via bundle provider interface
    if name == 'open dicom':
        open_dicom(session)
    else:
        raise ValueError('Medical Toolbar called with unknown operation "%s"' % name)

def open_dicom(session):
    from chimerax.open_command import show_open_folder_dialog
    show_open_folder_dialog(session)
