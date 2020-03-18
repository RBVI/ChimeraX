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

from PyQt5.QtWidgets import QFrame, QGridLayout, QComboBox, QLabel, QHBoxLayout, QLineEdit, QCheckBox
class SaveOptionsWidget(QFrame):
    def __init__(self, session):
        super().__init__()
        layout = QGridLayout()
        layout.setContentsMargins(2, 0, 0, 0)
        row = 0

        size_frame = QFrame()
        size_layout = QHBoxLayout(size_frame)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self._width = w = QLineEdit(size_frame)
        new_width = int(0.4 * w.sizeHint().width())
        w.setFixedWidth(new_width)
        w.textEdited.connect(self._width_changed)
        x = QLabel(size_frame)
        x.setText("x")
        self._height = h = QLineEdit(size_frame)
        h.setFixedWidth(new_width)
        h.textEdited.connect(self._height_changed)
        
        from PyQt5.QtCore import Qt
        size_layout.addWidget(self._width, Qt.AlignRight)
        size_layout.addWidget(x, Qt.AlignHCenter)
        size_layout.addWidget(self._height, Qt.AlignLeft)
        size_frame.setLayout(size_layout)
        size_label = QLabel()
        size_label.setText("Size:")
        layout.addWidget(size_label, row, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(size_frame, row, 1, Qt.AlignLeft)
        row += 1

        self._keep_aspect = ka = QCheckBox('preserve aspect')
        ka.setChecked(True)
        ka.stateChanged.connect(self._aspect_changed)
        layout.addWidget(ka, row, 1, Qt.AlignLeft)
        row += 1

        ss_label = QLabel()
        ss_label.setText("Supersample:")
        supersamples = QComboBox()
        supersamples.addItems([o[0] for o in SUPERSAMPLE_OPTIONS])
        supersamples.setCurrentIndex(2)
        layout.addWidget(ss_label, row, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(supersamples, row, 1, Qt.AlignLeft)
        self._supersample = supersamples

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

    def _aspect_changed(self, state):
        if self._keep_aspect.isChecked():
            w,h,iw,ih = self._sizes()
            if iw != w:
                self._width_changed()
            else:
                self._height_changed()
    
