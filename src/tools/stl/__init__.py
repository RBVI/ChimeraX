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

def initialize(bundle_info, session):
    """Register STL file format."""
    from . import stl
    stl.register()

    # Configure STLModel for session saving
    stl.STLModel.bundle_info = bundle_info

#
# 'get_class' is called by session code to get class saved in a session
#
def get_class(class_name):
    if class_name == 'STLModel':
        from . import stl
        return stl.STLModel
    return None
