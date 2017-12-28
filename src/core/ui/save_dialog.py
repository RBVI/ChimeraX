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

class _SaveFormat:

    def __init__(self, name, wildcard, make_ui, update, save):
        self.name = name
        self._wildcard = wildcard
        self._make_ui = make_ui
        self._update = update
        self._save = save
        self._window = None

    @property
    def wildcard(self):
        return self._wildcard()

    def window(self, parent):
        if self._make_ui is None:
            return None
        if self._window is None:
            self._window = self._make_ui(parent)
        return self._window

    def update(self, session, save_dialog):
        if self._update:
            self._update(session, save_dialog)

    def save(self, session, filename):
        return self._save(session, filename)

def _add_missing_suffix(filename, fmt):
    import os.path
    ext = os.path.splitext(filename)[1]
    exts = fmt.extensions
    if exts and ext not in exts:
        filename += exts[0]
    return filename

class MainSaveDialogBase:

    DEFAULT_FORMAT = "ChimeraX Session"

    def __init__(self, ui):
        self.file_dialog = None
        self._registered_formats = {}
        self._format_selector = None
        self.register(self.DEFAULT_FORMAT, _session_wildcard, None, None, _session_save)
        from ..toolshed import SESSION
        from ..io import formats
        from .open_save import export_file_filter
        for fmt in formats(open=False):
            if fmt.category not in (SESSION, "Image"):
                self.register(fmt.name, lambda fmt=fmt: export_file_filter(format_name=fmt.name),
                    None, None, lambda ses, fn, fmt=fmt:
                    fmt.export(ses, _add_missing_suffix(fn, fmt), fmt.name))

    def register(self, format_name, wildcard, make_ui, update, save):
        self._registered_formats[format_name] = _SaveFormat(format_name, wildcard, make_ui,
                                                            update, save)
        if self._format_selector:
            self._update_format_selector()

    def deregister(self, format_name):
        del self._registered_formats[format_name]


def _session_wildcard():
    from .open_save import export_file_filter
    from .. import toolshed
    return export_file_filter(toolshed.SESSION)


def _session_save(session, filename):
    import os.path
    ext = os.path.splitext(filename)[1]
    from .. import io
    fmt = io.format_from_name("ChimeraX session")
    exts = fmt.extensions
    if exts and ext not in exts:
        filename += exts[0]
    from ..commands import run, quote_if_necessary
    run(session, "save session %s" % quote_if_necessary(filename))


class ImageSaverBase:

    DEFAULT_FORMAT = "png"
    DEFAULT_EXT = "png"
    SUPERSAMPLE_OPTIONS = (("None", None),
                           ("2x2", 2),
                           ("3x3", 3),
                           ("4x4", 4))

    def __init__(self, save_dialog):
        import weakref
        self._save_dialog = weakref.ref(save_dialog)

    def _file_dialog(self):
        d = self._save_dialog()
        if d:
            return d.file_dialog
        else:
            return None

    def _select_format(self, *args):
        # TODO: enable options that apply to this graphics format
        pass

    def register(self):
        self._save_dialog().register("Image File", self.wildcard, self.make_ui,
                                     self.update, self.save)

class MainSaveDialog(MainSaveDialogBase):
    def display(self, parent, session):
        self.session = session
        if self.file_dialog is None:
            from .open_save import SaveDialog
            self.file_dialog = SaveDialog(parent, "Save File")
            self._customize_file_dialog()
        else:
            fmt = self.current_format()
            fmt.update(session, self)
        try:
            if not self.file_dialog.exec():
                return
        finally:
            del self.session
        fmt = self.current_format()
        filename = self.file_dialog.selectedFiles()[0]
        fmt.save(session, filename)

    def current_format(self):
        if self._format_selector is None:
            format_name = self.DEFAULT_FORMAT
        else:
            format_name = self._format_selector.currentText()
        return self._registered_formats[format_name]

    def set_wildcard(self, format):
        fmt = self.current_format()
        self.file_dialog.setNameFilters(fmt.wildcard.split(';;'))

    def _customize_file_dialog(self):
        from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QFrame
        self._options_panel = options_panel = self.file_dialog.custom_area
        label = QLabel(options_panel)
        label.setText("Format:")
        self._format_selector = selector = QComboBox(options_panel)
        selector.currentIndexChanged.connect(self._select_format)
        self._no_options_label = no_opt_label = QLabel(options_panel)
        no_opt_label.setText("No user-settable options")
        no_opt_label.setFrameStyle(QFrame.Box)
        self._known_options = set([no_opt_label])
        self._current_option = no_opt_label
        self._update_format_selector()
        selector.setCurrentIndex(selector.findText(self.DEFAULT_FORMAT))
        self._options_layout = options_layout = QHBoxLayout(options_panel)
        options_layout.addWidget(label)
        options_layout.addWidget(selector)
        options_layout.addWidget(no_opt_label)
        options_panel.setLayout(options_layout)
        self._select_format(self.DEFAULT_FORMAT)
        return options_panel

    def _select_format(self, *args, **kw):
        fmt = self.current_format()
        self.file_dialog.setNameFilters(fmt.wildcard.split(';;'))
        w = fmt.window(self._options_panel) or self._no_options_label
        fmt.update(self.session, self)
        if w is self._current_option:
            return
        self._current_option.hide()
        if w not in self._known_options:
            from PyQt5.QtWidgets import QFrame
            w.setFrameStyle(QFrame.Box)
            self._options_layout.addWidget(w)
            self._known_options.add(w)
        w.show()
        self._current_option = w

    def _update_format_selector(self):
        choices = list(self._registered_formats.keys())
        choices.sort()
        self._format_selector.clear()
        self._format_selector.addItems(choices)

class ImageSaver(ImageSaverBase):
    def make_ui(self, parent):
        from PyQt5.QtWidgets import QFrame, QGridLayout, QComboBox, QLabel, QHBoxLayout, \
            QLineEdit
        container = QFrame(parent)
        container.setFrameStyle(QFrame.Box)
        layout = QGridLayout(container)
        layout.setContentsMargins(2, 0, 0, 0)

        from ..image import image_formats
        selector = QComboBox(container)
        selector.addItems(list(f.name for f in image_formats))
        selector.currentIndexChanged.connect(self._select_format)
        selector.setCurrentIndex(selector.findText(self.DEFAULT_FORMAT))
        format_label = QLabel(container)
        format_label.setText("Format:")
        from PyQt5.QtCore import Qt
        layout.addWidget(format_label, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(selector, 0, 1, Qt.AlignLeft)
        self._format_selector = selector

        size_frame = QFrame(container)
        size_layout = QHBoxLayout(size_frame)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self._width = QLineEdit(size_frame)
        new_width = int(0.4 * self._width.sizeHint().width())
        self._width.setFixedWidth(new_width)
        x = QLabel(size_frame)
        x.setText("x")
        self._height = QLineEdit(size_frame)
        self._height.setFixedWidth(new_width)
        size_layout.addWidget(self._width, Qt.AlignRight)
        size_layout.addWidget(x, Qt.AlignHCenter)
        size_layout.addWidget(self._height, Qt.AlignLeft)
        size_frame.setLayout(size_layout)
        size_label = QLabel(container)
        size_label.setText("Size:")
        layout.addWidget(size_label, 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(size_frame, 1, 1, Qt.AlignLeft)

        ss_label = QLabel(container)
        ss_label.setText("Supersample:")
        supersamples = QComboBox(container)
        supersamples.addItems([o[0] for o in self.SUPERSAMPLE_OPTIONS])
        layout.addWidget(ss_label, 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(supersamples, 2, 1, Qt.AlignLeft)
        self._supersample = supersamples

        container.setLayout(layout)
        return container

    def save(self, session, filename):
        import os.path
        ext = os.path.splitext(filename)[1]
        e = '.' + self._get_current_extension()
        if ext != e:
            filename += e
        try:
            w = int(self._width.text())
            h = int(self._height.text())
        except ValueError:
            from ..errors import UserError
            raise UserError("width/height must be integers")
        if w <= 0 or h <= 0:
            from ..errors import UserError
            raise UserError("width/height must be positive integers")
        ss = self.SUPERSAMPLE_OPTIONS[self._supersample.currentIndex()][1]
        from ..commands import run, quote_if_necessary
        cmd = "save image %s width %g height %g" % (quote_if_necessary(filename), w, h)
        if ss is not None:
            cmd += " supersample %g" % ss
        run(session, cmd)

    def update(self, session, save_dialog):
        gw = session.ui.main_window.graphics_window
        w, h = gw.width(), gw.height()
        self._width.setText(str(w))
        self._height.setText(str(h))

    def wildcard(self):
        from ..image import image_formats
        exts = sum((list(f.suffixes) for f in image_formats), [])
        exts.remove(self.DEFAULT_EXT)
        exts.insert(0, self.DEFAULT_EXT)
        fmts = ' '.join("*.%s" % e for e in exts)
        wildcard = "Image file (%s)" % fmts
        return wildcard

    def _get_current_extension(self):
        format_name = self._format_selector.currentText()
        from ..image import image_formats
        for f in image_formats:
            if f.name == format_name:
                return f.suffixes[0]
        else:
            raise RuntimeError("unsupported graphics format: %s" % format_name)
