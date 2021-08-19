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

SUPERSAMPLE_OPTIONS = (("None", 1),
                       ("2x", 2),
                       ("3x", 3),
                       ("4x", 4))

from Qt.QtWidgets import QFrame, QHBoxLayout, QComboBox, QLabel, QLineEdit, QCheckBox
from Qt.QtCore import Qt
class SaveOptionsWidget(QFrame):
    def __init__(self, session, fmt):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 0, 0, 0)

        size_layout = QHBoxLayout()
        size_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(size_layout, stretch=1)
        size_layout.addWidget(QLabel("Size:"), alignment=Qt.AlignRight | Qt.AlignVCenter)
        self._width = w = QLineEdit()
        size_layout.addWidget(self._width, alignment=Qt.AlignRight)
        new_width = int(0.4 * w.sizeHint().width())
        w.setFixedWidth(new_width)
        w.textEdited.connect(self._width_changed)
        size_layout.addWidget(QLabel("x"), alignment=Qt.AlignHCenter)
        self._height = h = QLineEdit()
        size_layout.addWidget(self._height, alignment=Qt.AlignLeft)
        h.setFixedWidth(new_width)
        h.textEdited.connect(self._height_changed)

        self._keep_aspect = ka = QCheckBox('preserve aspect')
        ka.setChecked(True)
        ka.stateChanged.connect(self._aspect_changed)
        size_layout.addWidget(ka, alignment=Qt.AlignLeft)

        supersample_layout = QHBoxLayout()
        supersample_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(supersample_layout, stretch=1)
        supersample_layout.addWidget(QLabel("Supersample:"), alignment=Qt.AlignRight)

        supersamples = QComboBox()
        supersamples.addItems([o[0] for o in SUPERSAMPLE_OPTIONS])
        supersamples.setCurrentIndex(2)
        supersample_layout.addWidget(supersamples, alignment=Qt.AlignLeft)
        self._supersample = supersamples

        if fmt in ["PNG", "TIFF"]:
            self._transparent = trans = QCheckBox('Transparent\nbackground')
            trans.setChecked(False)
            layout.addWidget(trans)
        else:
            self._transparent = None

        self._session = session
        gw = session.ui.main_window.graphics_window
        w, h = gw.width(), gw.height()
        self._width.setText(str(w))
        self._height.setText(str(h))

        self.setLayout(layout)

    def options_string(self):
        # Get image width and height
        try:
            w = int(self._width.text())
            h = int(self._height.text())
        except ValueError:
            from chimerax.core.errors import UserError
            raise UserError("width/height must be integers")
        if w <= 0 or h <= 0:
            from chimerax.core.errors import UserError
            raise UserError("width/height must be positive integers")

        # Get supersampling
        ss = SUPERSAMPLE_OPTIONS[self._supersample.currentIndex()][1]

        cmd = "width %g height %g" % (w, h)
        if ss is not None:
            cmd += " supersample %g" % ss
        if self._transparent is not None and self._transparent.isChecked():
            cmd += " transparentBackground true"
        return cmd

    def _width_changed(self):
        if self._keep_aspect.isChecked():
            w,h,iw,ih = self._sizes()
            if w > 0 and iw is not None:
                self._height.setText('%.0f' % ((iw/w) * h))
    
    def _height_changed(self):
        if self._keep_aspect.isChecked():
            w,h,iw,ih = self._sizes()
            if h > 0 and ih is not None:
                self._width.setText('%.0f' % ((ih/h) * w))

    def _sizes(self):
        gw = self._session.ui.main_window.graphics_window
        w, h = gw.width(), gw.height()
        try:
            iw = int(self._width.text())
        except ValueError:
            iw = None
        try:
            ih = int(self._height.text())
        except ValueError:
            ih = None
        return w, h, iw, ih

    def _aspect_changed(self, state):
        if self._keep_aspect.isChecked():
            w,h,iw,ih = self._sizes()
            if iw != w:
                self._width_changed()
            else:
                self._height_changed()
    
