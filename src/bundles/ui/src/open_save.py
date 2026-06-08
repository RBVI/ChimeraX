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
    QLineEdit, QGraphicsView, QVBoxLayout, QSpinBox, QSizePolicy)
from Qt.QtCore import Qt, QMargins, QRectF, QSize
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
        "image_pad": 2,
        "save_area": "visible",
        "save_format": "PNG",
        "view_spacing": 2,
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
    def __init__(self, session, view_info, *args, depiction_name="scene", view_names=None, 
            view_fit="shrink", **kw):
        ''' Save an image of one or more QGraphicsViews

        'view_info' is either a single QGraphicsView, or a rectangular array of views that need to
        be "glued together" to form the final image.  If the latter, it is a series of rows (top to
        bottom) that contain QGraphicsViews (left to right).

        'depiction_name' is describes the overall image (e.g. "sequence alignment").

        'view_names' are descriptions of each of the component QGraphicsViews if 'view_info' was an array.

        'view_fit' controls how views are sized when multiple views are being depicted and the views in a
        row have different heights or views in a column have different widths.  If 'view_fit' is 'shrink'
        then the smallest size is used.  If it is "expand" then the largest size is used.

        The exec() method runs the dialog and returns a boolean indicating if the file was successfully
        saved.
        '''
        if isinstance(view_info, QGraphicsView):
            num_views = 1
        else:
            num_views = len(view_info) * len(view_info[0])
            if num_views == 1:
                view_info = view_info[0][0]
            elif view_names and num_views != len(view_names):
                raise ValueError("Number of view names (%d) does not match number of views (%d)"
                    % (len(view_names), num_views))
        super().__init__(view_info if num_views == 1 else view_info[-1][-1], *args, **kw)
        self.num_views = num_views
        self.session = session
        self.view_info = view_info
        self.depiction_name = depiction_name
        self.view_fit = view_fit
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
        custom_layout = QVBoxLayout()
        custom_area.setLayout(custom_layout)
        for i in range(1,3):
            explanation = QLabel()
            explanation.setWordWrap(True)
            explanation.setAlignment(Qt.AlignHCenter)
            custom_layout.addWidget(explanation)
            setattr(self, '_explanation%d' % i, explanation)
        if num_views > 1:
            text = "%d views " % num_views
            if view_names:
                text += '(' + ', '.join(view_names) + ') '
            text += "will be joined together to form the final image."
            text += "  The views will be composited directly adjacent to each other, so"
            text += " so it can be desirable to add an amount of spacing, which can be"
            text += " specified below (in pixels, and can be zero)."
            self._explanation1.setText(text)
        else:
            self._explanation1.setHidden(True)
        self.area_descriptions = {
            "all": f"entire {depiction_name}",
            "visible": "visible region"
        }
        text = 'The "%s" uses the minimum bounding box enclosing the depiction.' \
            % self.area_descriptions["all"]
        text += "  You can specify an amount of padding (in pixels) to add to the edges of the image."
        self._explanation2.setText(text)
        self._explanation2.setHidden(self.settings.save_area == "visible")
        self._widget_layout = widget_layout = QHBoxLayout()
        widget_layout.setSpacing(0)
        custom_layout.addLayout(widget_layout)
        widget_layout.addStretch(1)
        save_area_layout = QHBoxLayout()
        save_area_layout.setSpacing(0)
        widget_layout.addLayout(save_area_layout)
        save_area_layout.addWidget(QLabel("Save "))
        self._save_area_button = QPushButton(self.area_descriptions[self.settings.save_area])
        menu = QMenu(self._save_area_button)
        for area in ["all", "visible"]:
            menu.addAction(self.area_descriptions[area])
        menu.triggered.connect(self._save_area_changed)
        self._save_area_button.setMenu(menu)
        save_area_layout.addWidget(self._save_area_button)
        self._pad_stretch_col = widget_layout.count()
        widget_layout.addStretch(1)
        self._pad_label = QLabel("Padding: ")
        widget_layout.addWidget(self._pad_label)
        self._pad_box = QSpinBox()
        self._pad_box.setRange(0, 999)
        self._pad_box.setValue(self.settings.image_pad)
        widget_layout.addWidget(self._pad_box)
        if self.settings.save_area == "visible":
            widget_layout.setStretch(self._pad_stretch_col, 0)
            self._pad_label.setHidden(True)
            self._pad_box.setHidden(True)
        if num_views > 1:
            widget_layout.addStretch(1)
            widget_layout.addWidget(QLabel("View spacing: "))
            self._spacing_box = QSpinBox()
            self._spacing_box.setRange(0, 999)
            self._spacing_box.setValue(self.settings.view_spacing)
            widget_layout.addWidget(self._spacing_box)

        widget_layout.addStretch(1)
        widget_layout.addWidget(QLabel("DPI: "))
        self._dpi_entry = QLineEdit()
        self._dpi_entry.setAlignment(Qt.AlignCenter)
        self._dpi_entry.setPlaceholderText("default")
        self._dpi_entry.setMaximumWidth(50)
        validator = QIntValidator()
        validator.setBottom(1)
        self._dpi_entry.setValidator(validator)
        if self.settings.dpi is not None:
            self._dpi_entry.setText(str(self.settings.dpi))
        widget_layout.addWidget(self._dpi_entry)
        widget_layout.addStretch(1)

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
        self.settings.dpi = dpi = self.dpi
        self.settings.save_area = save_area = self.save_area
        from Qt.QtGui import QPainter
        if self.num_views == 1:
            render_kw = {}
            if save_area == "visible":
                source = self.view_info
                image_size = self.view_info.viewport().rect().size()
            else:
                source = self.view_info.scene()
                target_rect = source.sceneRect().toAlignedRect()
                image_size = target_rect.size()
                self.settings.image_pad = image_pad = self.image_pad
                if image_pad > 0:
                    render_kw = {
                        'target': QRectF(image_pad, image_pad, image_size.width(), image_size.height())
                    }
                    image_size = image_size.grownBy(QMargins(image_pad, image_pad, image_pad, image_pad))
            image = self._make_image(image_size, dpi)
            if save_area == "all" and image_pad > 0:
                image.fill(source.backgroundBrush().color())
            source.render(QPainter(image), **render_kw)
        else:
            if save_area == "visible":
                get_view_size = lambda item: item.viewport().rect().size()
                get_source = lambda item: item
                image_pad = 0
            else:
                get_view_size = lambda item: item.scene().sceneRect().toAlignedRect().size()
                get_source = lambda item: item.scene()
                self.settings.image_pad = image_pad = self.image_pad
            widths = [None] * len(self.view_info[0])
            heights = [None] * len(self.view_info)
            fit_func = min if self.view_fit == "shrink" else max
            for row, row_items in enumerate(self.view_info):
                for col, item in enumerate(row_items):
                    size = get_view_size(item)
                    w, h = size.width(), size.height()
                    cur_w = widths[col]
                    if cur_w is None:
                        widths[col] = w
                    else:
                        widths[col] = fit_func(cur_w, w)
                    cur_h = heights[row]
                    if cur_h is None:
                        heights[row] = h
                    else:
                        heights[row] = fit_func(cur_h, h)

            self.settings.view_spacing = view_spacing = self.view_spacing
            total_width = view_spacing * (len(widths) - 1) + image_pad * 2 * len(widths) + sum(widths)
            total_height = view_spacing * (len(heights) - 1) + image_pad * 2 * len(heights) + sum(heights)
            from math import ceil
            image_size = QSize(ceil(total_width), ceil(total_height))
            image = self._make_image(image_size, dpi)
            if view_spacing > 0 or image_pad > 0:
                image.fill(self.view_info[-1][-1].scene().backgroundBrush().color())
            cur_y = 0
            for row, row_items in enumerate(self.view_info):
                cur_x = 0
                for col, view in enumerate(row_items):
                    view_size = get_view_size(view)
                    get_source(view).render(QPainter(image), target=
                        QRectF(cur_x + image_pad, cur_y + image_pad, view_size.width(), view_size.height()))
                    cur_x += widths[col] + 2 * image_pad + view_spacing
                cur_y += heights[row] + 2 * image_pad + view_spacing
        if image.save(path, fmt_name.lower()):
            self.session.logger.info("Saved %s image to %s" % (self.depiction_name, path))
            return True
        self.session.logger.info("Failed to save %s image to %s" % (self.depiction_name, path))
        return False

    @property
    def image_pad(self):
        return self._pad_box.value()

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
        but_text = self._save_area_button.text()
        for key, text in self.area_descriptions.items():
            if but_text in (key, text):
                return key

    @property
    def view_spacing(self):
        return self._spacing_box.value()

    def _make_image(self, image_size, dpi):
        from Qt.QtGui import QImage
        image = QImage(image_size, QImage.Format_ARGB32)
        if dpi is not None:
            dpm = round(dpi * 39.3701)
            image.setDotsPerMeterX(dpm)
            image.setDotsPerMeterY(dpm)
        return image

    def _save_area_changed(self, action):
        self._save_area_button.setText(action.text())
        if self.save_area == "visible":
            self._explanation2.setHidden(True)
            self._widget_layout.setStretch(self._pad_stretch_col, 0)
            self._pad_label.setHidden(True)
            self._pad_box.setHidden(True)
        else:
            self._explanation2.setHidden(False)
            self._widget_layout.setStretch(self._pad_stretch_col, 1)
            self._pad_label.setHidden(False)
            self._pad_box.setHidden(False)
