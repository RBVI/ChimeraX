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

def get_icon_path(icon_name):
    from os.path import join, dirname
    return join(dirname(__file__), icon_name + '.png')

def get_qt_icon(icon_name):
    from Qt.QtGui import QPixmap, QIcon
    pixmap = QPixmap(get_icon_path(icon_name))
    icon = QIcon(pixmap)
    return icon

