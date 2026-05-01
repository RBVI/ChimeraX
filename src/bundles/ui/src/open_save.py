# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
open_save: open/save dialogs
============================

TODO
"""

from Qt.QtWidgets import (QFileDialog, QSizePolicy, QPushButton, QMenu, QFrame, QHBoxLayout, QLabel,
    QLineEdit)
from Qt.QtCore import Qt
class SaveDialog(QFileDialog):
    use_native = False
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
        if not self.use_native:
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
            self._custom_area = QFrame(self)
            self._custom_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
            self._custom_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            if not self.use_native:
                layout = self.layout()
                row = layout.rowCount()
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

from chimerax.core.settings import Settings
class SaveQGraphicsDialogSettings(Settings):
    AUTO_SAVE = {
        "dpi": None,
        "save_area": "visible",
        "save_format": "PNG",
        #"transparent_background": False,
    }

# Cribbed from chimerax.ui.open_save.SaveDialog, but since we need to save the formats
# ourselves and save some formats that might be unknown to ChimeraX (depending on what
# QImage can save), we provide our own dialog
from Qt.QtWidgets import QFileDialog
class SaveQGraphicsDialog(QFileDialog):
    format_info = {
        "bmp": ("BMP", "Windows bitmap", "bmp"),
        "jfif": ("JFIF", "JPEG File Interchange Format", "jfif"),
        "jp2": ("JP2", "JPEG 2000", "jp2"),
        "jpeg": ("JPEG", "Joint Photographic Experts Group", "jpg *.jpeg"),
        "png": ("PNG", "Portable Network Graphics", "png"),
        "tiff": ("TIFF", "Tagged Image File Format", "tiff"),
        "webp": ("WebP", "WebP", "webp"),
    }
    def __init__(self, session, view, *args, depiction_name="scene", **kw):
        super().__init__(view, *args, **kw)
        self.session = session
        self.view = view
        self.depiction_name = depiction_name
        from Qt.QtGui import QImageWriter, QIntValidator
        available_fmt_info = []
        for fmt in QImageWriter.supportedImageFormats():
            try:
                available_fmt_info.append(self.format_info[bytes(fmt).decode('utf8')])
            except KeyError:
                continue
        name_filters = ["%s [%s] (*.%s)" % fmt_info for fmt_info in available_fmt_info]
        self.filter_to_info = {flt: info for flt, info in zip(name_filters, available_fmt_info)}
        fmt_to_filter = { info[0]: flt for flt, info in self.filter_to_info.items() }
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setOption(QFileDialog.DontUseNativeDialog)
        self.setNameFilters(name_filters)
        if not hasattr(self.__class__, "settings"):
            self.__class__.settings = SaveQGraphicsDialogSettings(session, "QGraphics save dialog")
        try:
            self.selectNameFilter(fmt_to_filter[self.settings.save_format])
        except KeyError:
            self.selectNameFilter(fmt_to_filter["PNG"])

        custom_area = QFrame(self)
        custom_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
        custom_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = self.layout()
        row = layout.rowCount()
        layout.addWidget(custom_area, row, 0, 1, -1)
        custom_layout = QHBoxLayout()
        custom_area.setLayout(custom_layout)
        custom_layout.addStretch(1)
        self.area_descriptions = {
            "all": f"entire {depiction_name}",
            "visible": "visible region"
        }
        save_area_layout = QHBoxLayout()
        custom_layout.addLayout(save_area_layout)
        save_area_layout.addWidget(QLabel("Save"))
        self.save_area_button = QPushButton(self.area_descriptions[self.settings.save_area])
        menu = QMenu(self.save_area_button)
        for area in ["all", "visible"]:
            menu.addAction(self.area_descriptions[area])
        menu.triggered.connect(lambda action, but=self.save_area_button: but.setText(action.text()))
        self.save_area_button.setMenu(menu)
        save_area_layout.addWidget(self.save_area_button)
        '''
        self._transparent_checkbox = QCheckBox("Transparent background")
        self._transparent_checkbox.setChecked(self.settings.transparent_background)
        custom_layout.addWidget(self._transparent_checkbox)
        custom_layout.addStretch(1)
        '''
        custom_layout.addWidget(QLabel("DPI:"))
        self._dpi_entry = QLineEdit()
        self._dpi_entry.setAlignment(Qt.AlignCenter)
        self._dpi_entry.setPlaceholderText("default")
        self._dpi_entry.setMaximumWidth(50)
        validator = QIntValidator()
        validator.setBottom(1)
        self._dpi_entry.setValidator(validator)
        if self.settings.dpi is not None:
            self._dpi_entry.setText(str(self.settings.dpi))
        custom_layout.addWidget(self._dpi_entry)
        custom_layout.addStretch(1)

    @property
    def dpi(self):
        if self._dpi_entry.hasAcceptableInput():
            return int(self._dpi_entry.text())
        return None

    def exec(self):
        ok = super().exec()
        if not ok:
            return False
        path, fmt_name = self.file_info
        if path is None:
            return False
        #self.settings.transparent_background = self.transparent_background
        self.settings.dpi = dpi = self.dpi
        self.settings.save_area = save_area = self.save_area
        from Qt.QtGui import QImage, QPainter
        if save_area == "visible":
            source = self.view
            image_size = self.view.viewport().rect().size()
        else:
            source = self.view.scene()
            image_size = source.sceneRect().toAlignedRect().size()
        #NOTE: investigate if I need to multiply toSize() by device pixel ratio
        image = QImage(image_size, QImage.Format_ARGB32)
        if dpi is not None:
            dpm = round(dpi * 39.3701)
            image.setDotsPerMeterX(dpm)
            image.setDotsPerMeterY(dpm)
        source.render(QPainter(image))
        if image.save(path, fmt_name.lower()):
            self.session.logger.info("Saved %s image to %s" % (self.depiction_name, path))
            return True
        self.session.logger.info("Failed to save %s image to %s" % (self.depiction_name, path))
        return False

    @property
    def file_info(self):
        paths = self.selectedFiles()
        if not paths:
            return None, None
        path = paths[0]
        name_filter = self.selectedNameFilter()
        fmt_name, fmt_desc, suffix_info = self.filter_to_info[name_filter]
        self.settings.save_format = fmt_name
        suffix = '.' + (suffix_info[:suffix_info.index(' ')] if ' ' in suffix_info else suffix_info)
        if path.endswith(suffix):
            return path, fmt_name
        return path + suffix, fmt_name

    @property
    def save_area(self):
        but_text = self.save_area_button.text()
        for key, text in self.area_descriptions.items():
            if but_text in (key, text):
                return key

    '''
    @property
    def transparent_background(self):
        return self._transparent_checkbox.isChecked()
    '''
