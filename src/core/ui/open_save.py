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
open_save: open/save dialogs
============================

TODO
"""

"""Just use PyQt5.QtWidgets.QFileDialog for opening files"""
from PyQt5.QtWidgets import QFileDialog
class SaveDialog(QFileDialog):
    def __init__(self, parent, *args, **kw):
        default_suffix = kw.pop('add_extension', None)
        name_filter = kw.pop('name_filter', None)
        super().__init__(parent, *args, **kw)
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setOption(QFileDialog.DontUseNativeDialog)
        if default_suffix:
            self.setDefaultSuffix(default_suffix)
        if name_filter:
            self.setNameFilter(name_filter)
        self._custom_area = None

    @property
    def custom_area(self):
        if self._custom_area is None:
            layout = self.layout()
            row = layout.rowCount()
            from PyQt5.QtWidgets import QFrame
            self._custom_area = QFrame(self)
            from PyQt5.QtCore import Qt
            layout.addWidget(self._custom_area, row, 0, 1, -1, Qt.AlignCenter)
        return self._custom_area

    def get_path(self):
        paths = self.selectedFiles()
        if not paths:
            return None
        path = paths[0]
        return path

def export_file_filter(category=None, format_name=None, all=False):
    """Return file name filter suitable for Export File dialog for Qt"""

    result = []
    from .. import io
    for fmt in io.formats(open = (format_name is not None)):
        if format_name and fmt.name != format_name:
            continue
        if category and fmt.category != category:
            continue
        exts = '*' + ' *'.join(fmt.extensions)
        result.append("%s files (%s)" % (fmt.name, exts))
    if all:
        result.append("All files (*)")
    if not result:
        if not category:
            files = "any"
        else:
            files = "\"%s\"" % category
        raise ValueError("No filters for %s files" % files)
    result.sort(key=str.casefold)
    return ';;'.join(result)

def open_file_filter(all=False):
    """Return file name filter suitable for Open File dialog for Qt"""

    combine = {}
    from .. import io
    for fmt in io.formats(export=False):
        exts = combine.setdefault(fmt.category, [])
        exts.extend(fmt.extensions)
    result = []
    for k in combine:
        exts = '*' + ' *'.join(combine[k])
        compression_suffixes = io.compression_suffixes()
        if compression_suffixes:
            for ext in combine[k]:
                exts += ' ' + ' '.join('*%s%s' % (ext, c) for c in compression_suffixes)
        result.append("%s files (%s)" % (k, exts))
    result.sort(key=str.casefold)
    if all:
        result.insert(0, "All files (*)")
    return ';;'.join(result)

