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

from Qt.QtWidgets import QFileDialog, QSizePolicy
from Qt.QtCore import Qt
class SaveDialog(QFileDialog):
    def __init__(self, session, parent = None, *args, data_formats=None, installed_only=True, **kw):
        if data_formats is None:
            data_formats = [fmt for fmt in session.save_command.save_data_formats if fmt.suffixes]
            if installed_only:
                data_formats = [fmt for fmt in data_formats
                    if session.save_command.provider_info(fmt).bundle_info.installed]
        data_formats.sort(key=lambda fmt: fmt.name.casefold())
        # make some things public
        self.data_formats = data_formats
        self.name_filters = [session.data_formats.qt_file_filter(fmt) for fmt in data_formats]
        if len(data_formats) == 1:
            default_suffix = data_formats[0].suffixes[0] if data_formats[0].suffixes else None
            name_filter = self.name_filters[0]
        else:
            default_suffix = name_filter = None
        super().__init__(parent, *args, **kw)
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setOption(QFileDialog.DontUseNativeDialog)
        if self.name_filters:
            self.setNameFilters(self.name_filters)
            if name_filter:
                self.setNameFilter(name_filter)
        if default_suffix:
            self.setDefaultSuffix(default_suffix)
        self._custom_area = None

    @property
    def custom_area(self):
        if self._custom_area is None:
            layout = self.layout()
            row = layout.rowCount()
            from Qt.QtWidgets import QFrame
            self._custom_area = QFrame(self)
            self._custom_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
            self._custom_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(self._custom_area, row, 0, 1, -1)
        return self._custom_area

    def get_path(self):
        paths = self.selectedFiles()
        if not paths:
            return None
        path = paths[0]
        return path

class OpenDialogWithMessage(QFileDialog):
    def __init__(self, parent = None, message = '', caption = 'Open File', starting_directory = None):
        if starting_directory is None:
            import os
            starting_directory = os.getcwd()
        QFileDialog.__init__(self, parent, caption = caption, directory = starting_directory)
        self.setFileMode(QFileDialog.AnyFile)
        self.setOption(QFileDialog.DontUseNativeDialog)

        if message:
            layout = self.layout()
            row = layout.rowCount()
            from Qt.QtWidgets import QLabel
            label = QLabel(message, self)
            layout.addWidget(label, row, 0, 1, -1, Qt.AlignLeft)

    def get_path(self):
        if not self.exec():
            return None
        paths = self.selectedFiles()
        if not paths:
            return None
        path = paths[0]
        return path

# Unless you need to add custom widgets to the dialog, you should use Qt.QtWidgets.QFileDialog
# for opening files, since that will have native look and feel.  The OpenDialog below is for
# those situations where you do need to add widgets.
class OpenDialog(QFileDialog):
    def __init__(self, parent = None, caption = 'Open File', starting_directory = None,
                 widget_alignment = Qt.AlignCenter, filter = ''):
        if starting_directory is None:
            import os
            starting_directory = os.getcwd()
        QFileDialog.__init__(self, parent, caption = caption, directory = starting_directory,
                             filter = filter)
        self.setFileMode(QFileDialog.AnyFile)
        self.setOption(QFileDialog.DontUseNativeDialog)

        from Qt.QtWidgets import QWidget
        self.custom_area = QWidget()
        layout = self.layout()
        row = layout.rowCount()
        layout.addWidget(self.custom_area, row, 0, 1, -1, widget_alignment)

    def get_path(self):
        if not self.exec():
            return None
        paths = self.selectedFiles()
        if not paths:
            return None
        path = paths[0]
        return path

    def get_paths(self):
        if not self.exec():
            return None
        paths = self.selectedFiles()
        if not paths:
            return None
        return paths
