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
ColorButton is a derived class from PyQt5.QtWidgets.QPushButton
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

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QPushButton
from numpy import array, uint8, ndarray

class ColorButton(QPushButton):

    color_changed = pyqtSignal(ndarray)

    def __init__(self, parent=None, *, max_size=None, has_alpha_channel=False, **kw):
        super().__init__(parent)
        if max_size is not None:
            self.setMaximumSize(*max_size)
        from PyQt5.QtCore import Qt
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        self._has_alpha_channel = has_alpha_channel
        self.clicked.connect(self.show_color_chooser)
        self._color = None

    def set_color(self, color):
        self.setStyleSheet('background-color: %s' % hex_color_name(color))
        self._color = color_to_numpy_rgba8(color)

    def show_color_chooser(self):
        from PyQt5.QtWidgets import QColorDialog
        cd = QColorDialog(self)
        cd.setOption(cd.ShowAlphaChannel, self._has_alpha_channel)
        cd.setOption(cd.NoButtons, True)
        if self._color is not None:
            cd.setCurrentColor(QColor(*tuple(self._color)))
        cd.currentColorChanged.connect(self._color_changed_cb)
        cd.show()

    def _color_changed_cb(self, color):
        self.set_color(color)
        self.color_changed.emit(self._color)

def color_to_numpy_rgba8(color):
    if isinstance(color, QColor):
        return array([color.red(), color.green(), color.blue(), color.alpha()], dtype=uint8)
    if isinstance(color[0], int):
        if len(color) == 3:
            color = list(color) + [255]
        return array(color, dtype=uint8)
    if isinstance(color[0], float):
        if len(color) == 3:
            color = list(color) + [1.0]
        return array([min(255, max(0, int(ch*255.0 + 0.5))) for ch in color], dtype=uint8)
    return color

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
            import os
            this_dir, fname = os.path.split(__file__)
            self.setStyleSheet("background-image: url(%s);" % os.path.join(this_dir, "multi.png"))
            self.setText(">1")
        else:
            self.setText("")
            ColorButton.set_color(self, color)
