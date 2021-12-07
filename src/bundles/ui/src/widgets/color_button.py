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
ColorButton is a derived class from Qt.QtWidgets.QPushButton
that simplifies showing and editing colors

ColorButton may be instantiated just like QPushButton, but handles
these extra keyword arguments:

    size_hint:   a QSize compatible value, typically (width, height),
                 specifying the preferred initial size for the view.
    interceptor: a callback function taking one argument, an instance
                 of QWebEngineUrlRequestInfo, invoked to handle navigation
                 requests.
    schemes:     an iterable of custom schemes that will be used in the
                 view.  If schemes is specified, then interceptor will
                 be called when custom URLs are clicked.
    download:    a callback function taking one argument, an instance
                 of QWebEngineDownloadItem, invoked when download is
                 requested.
"""

from Qt.QtCore import Signal
from Qt.QtGui import QColor
from Qt.QtWidgets import QPushButton
from numpy import array, uint8, ndarray

# some hackery to attempt to make it one color-chooser dialog for the
# entire app, rather than one per color button

_color_dialog = None
_color_callback = None
_color_setter_id = None
def _make_color_callback(*args):
    if _color_callback is not None:
        _color_callback(*args)

def _color_dialog_destroyed(*args):
    global _color_dialog
    _color_dialog = None

def _check_color_chooser(dead_button_id):
    global _color_setter_id, _color_callback
    if _color_setter_id == dead_button_id:
        _color_callback = _color_setter_id = None
        if _color_dialog:
            _color_dialog.hide()


class ColorButton(QPushButton):

    color_changed = Signal(ndarray)
    color_pause = Signal(ndarray)

    def __init__(self, *args, max_size=None, has_alpha_channel=False, pause_delay=None, **kw):
        super().__init__(*args)
        if max_size is not None:
            self.setMaximumSize(*max_size)
        from Qt.QtCore import Qt
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        self._has_alpha_channel = has_alpha_channel
        self.clicked.connect(self.show_color_chooser)
        self._color = None
        self._pause_timer = None
        self._pause_delay = pause_delay 	# Seconds before color_pause signal is issued.

    def get_color(self):
        return self._color

    def set_color(self, color):
        rgba = color_to_numpy_rgba8(color)
        if (rgba == self._color).all():
            return
        self.setStyleSheet('background-color: %s' % hex_color_name(color))
        self._color = rgba

    color = property(get_color, set_color)

    def show_color_chooser(self):
        global _color_dialog, _color_callback, _color_dialog_destroyed, _color_setter_id
        _color_setter_id = id(self)
        _color_callback = None
        if _color_dialog is None:
            from Qt.QtWidgets import QColorDialog
            _color_dialog = cd = QColorDialog(self.window())
            cd.setOption(cd.NoButtons, True)
            cd.currentColorChanged.connect(_make_color_callback)
            cd.destroyed.connect(_color_dialog_destroyed)
        else:
            cd = _color_dialog
            # On Mac, Qt doesn't realize it when the color dialog has been hidden
            # with the red 'X' button, so "hide" it now so that Qt doesn't believe
            # that the later show() is a no op.  Whereas on Windows doing a hide
            # followed by a show causes the chooser to jump back to it's original
            # position, so do the hide _only_ on Mac
            import sys
            if sys.platform == 'darwin':
                cd.hide()
        cd.setOption(cd.ShowAlphaChannel, self._has_alpha_channel)
        if self._color is not None:
            cd.setCurrentColor(QColor(*tuple(self._color)))
        _color_callback = self._color_changed_cb
        cd.show()

    def changeEvent(self, event):
        if event.type() == event.EnabledChange:
            if self.isEnabled():
                color = self._color
            else:
                color = [int((c + 218)/2) for c in self._color]
            self.setStyleSheet('background-color: %s' % hex_color_name(color))

    def _color_changed_cb(self, color):
        try:
            self.set_color(color)
        except RuntimeError:
            # C++ has been destroyed (don't seem to get a destroyed() signal)
            global _color_callback
            _color_callback = None
        else:
            self.color_changed.emit(self._color)
            self._set_pause_timer()

    def _set_pause_timer(self):
        delay = self._pause_delay
        if delay is None:
            return
        t = self._pause_timer
        if t is not None:
            t.stop()
        from Qt.QtCore import QTimer
        self._pause_timer = t = QTimer()
        t.setSingleShot(True)
        t.timeout.connect(lambda *, p=self.color_pause, c=self._color: p.emit(c))
        t.start(int(1000*self._pause_delay))

def color_to_numpy_rgba8(color):
    if isinstance(color, QColor):
        return array([color.red(), color.green(), color.blue(), color.alpha()], dtype=uint8)
    from chimerax.core.colors import Color, BuiltinColors
    if isinstance(color, str):
        try:
            color = BuiltinColors[color]
        except KeyError:
            raise ValueError("'%s' is not a built-in color name" % color)
    if isinstance(color, Color):
        return color.uint8x4()
    import numbers
    if isinstance(color[0], numbers.Integral):
        if len(color) == 3:
            color = list(color) + [255]
        return array(color, dtype=uint8)
    if isinstance(color[0], numbers.Real):
        if len(color) == 3:
            color = list(color) + [1.0]
        return array([min(255, max(0, int(ch*255.0 + 0.5))) for ch in color], dtype=uint8)
    raise ValueError("Don't know how to convert %s to integral numpy array" % repr(color))

def hex_color_name(color):
    return "#%02x%02x%02x" % tuple(color_to_numpy_rgba8(color)[:3])

class MultiColorButton(ColorButton):
    """Like ColorButton, but can be used when multiple different colors are present.
       Typically used in option panels when the objects involved have multiple different
       colors for an attribute.
    """

    def set_color(self, color):
        """Like ColorButton.set_color, but None means multiple colors"""
        if color is None:
            from chimerax.ui.icons import get_icon_path
            icon_file = get_icon_path("multi")
            max_size = self.maximumSize()
            import os
            if max_size.width() == max_size.height():
                test_icon = get_icon_path("multi%d" % max_size.width())
                if os.path.exists(test_icon):
                    icon_file = test_icon
            if os.sep != '/':
                icon_file = '/'.join(icon_file.split(os.sep))
            self.setStyleSheet("background-image: url(%s);" % icon_file)
        else:
            ColorButton.set_color(self, color)

    color = property(ColorButton.get_color, set_color)

