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


def button_row(parent, title, name_and_callback_list, hspacing = 3):
    from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton
    f = QFrame(parent)
#    f.setStyleSheet('QFrame { background-color: pink; padding-top: 0px; padding-bottom: 0px;}')
    parent.layout().addWidget(f)

    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(hspacing)

    l = QLabel(title, f)
    layout.addWidget(l)
#    l.setStyleSheet('QLabel { background-color: pink;}')
    
    from PyQt5.QtCore import Qt
    for name, callback in name_and_callback_list:
        b = QPushButton(name, f)
#        b.setMaximumSize(100,25)
#        b.setStyleSheet('QPushButton { background-color: pink;}')
#        b.setAttribute(Qt.WA_LayoutUsesWidgetRect) # Avoid extra padding on Mac
        b.setStyleSheet('QPushButton { padding-left: 15px; padding-right: 15px; padding-top: 5px; padding-bottom: 2px;}')
        if callback is None:
            b.setEnabled(False)
        else:
            b.clicked.connect(callback)
        layout.addWidget(b)

    layout.addStretch(1)

    return f

def _row_frame(parent, spacing = 5):
    from PyQt5.QtWidgets import QFrame, QHBoxLayout
    f = QFrame(parent)
    parent.layout().addWidget(f)

    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(spacing)
    return f, layout

class EntriesRow:
    def __init__(self, parent, *args, spacing = 5):
        f, layout = _row_frame(parent, spacing)
        self.frame = f
        
        from PyQt5.QtWidgets import QLabel, QPushButton
        values = []
        for a in args:
            if isinstance(a, str):
                if a == '':
                    s = StringEntry(f)
                    layout.addWidget(s.widget)
                    values.append(s)
                else:
                    newline = a.endswith('\n')
                    if newline:
                        a = a[:-1]
                    l = QLabel(a, f)
                    layout.addWidget(l)
                    if newline:
                        # String ends in newline so make new row.
                        layout.addStretch(1)
                        f, layout = _row_frame(parent)
            elif isinstance(a, bool):
                cb = BooleanEntry(f, a)
                layout.addWidget(cb.widget)
                values.append(cb)
            elif isinstance(a, int):
                ie = IntegerEntry(f, a)
                layout.addWidget(ie.widget)
                values.append(ie)
            elif isinstance(a, float):
                fe = FloatEntry(f, a)
                layout.addWidget(fe.widget)
                values.append(fe)
            elif isinstance(a, tuple):
                b = QPushButton(a[0], f)
# TODO: QPushButton has extra vertical space (about 5 pixels top and bottom) on macOS 10.14 (Mojave)
#       with Qt 5.12.4.  Couldn't find any thing to fix this, although below are some attempts
#                b.setMaximumSize(100,25)
#                from PyQt5.QtWidgets import QToolButton
#                b = QToolButton(f)
#                b.setText(a[0])
#                from PyQt5.QtCore import Qt
#                b.setContentsMargins(0,0,0,0)
#                b.setAttribute(Qt.WA_LayoutUsesWidgetRect)
#                b.setStyleSheet('padding-left: 15px; padding-right: 15px; padding-top: 4px;  padding-bottom: 4px;')
#                b.setStyleSheet('QPushButton { border: none;}')
#                b.setStyleSheet('QPushButton { background-color: pink;}')
#                layout.setAlignment(b, Qt.AlignTop)
                layout.addWidget(b)
                b.clicked.connect(a[1])

        layout.addStretch(1)    # Extra space at end

        self.values = values

def radio_buttons(*check_boxes):
    for cb in check_boxes:
        cb.widget.stateChanged.connect(lambda state, cb=cb, others=check_boxes: _uncheck_others(cb, others))

def _uncheck_others(check_box, other_check_boxes):
    if check_box.enabled:
        for ocb in other_check_boxes:
            if ocb is not check_box:
                ocb.enabled = False

class StringEntry:
    def __init__(self, parent, value = '', pixel_width = 100):
        from PyQt5.QtWidgets import QLineEdit
        self._line_edit = le = QLineEdit(value, parent)
        le.setMaximumWidth(pixel_width)
        self.return_pressed = le.returnPressed
    def _get_value(self):
        return self._line_edit.text()
    def _set_value(self, value):
        self._line_edit.setText(value)
    value = property(_get_value, _set_value)
    @property
    def widget(self):
        return self._line_edit

class NumberEntry:
    format = '%d'
    string_to_value = int
    def __init__(self, parent, value, pixel_width = 50):
        from PyQt5.QtWidgets import QLineEdit
        self._line_edit = le = QLineEdit(self.format % value, parent)
        le.setMaximumWidth(pixel_width)
        self.return_pressed = le.returnPressed
    def _get_value(self):
        return self.string_to_value(self._line_edit.text())
    def _set_value(self, value):
        v = self.format % value if value is not None else ''
        self._line_edit.setText(v)
    value = property(_get_value, _set_value)
    @property
    def widget(self):
        return self._line_edit
                
class IntegerEntry(NumberEntry):
    pass

class FloatEntry(NumberEntry):
    format = '%.4g'
    string_to_value = float

class BooleanEntry:
    def __init__(self, parent, value):
        from PyQt5.QtWidgets import QCheckBox
        self._check_box = cb = QCheckBox(parent)
        cb.setChecked(value)
        self.changed = cb.stateChanged
    def _get_value(self):
        return self._check_box.isChecked()
    def _set_value(self, value):
        self._check_box.setChecked(value)
    value = property(_get_value, _set_value)
    enabled = value
    @property
    def widget(self):
        return self._check_box

from PyQt5.QtWidgets import QWidget
class CollapsiblePanel(QWidget):
    def __init__(self, parent=None, title=''):
        QWidget.__init__(self, parent=parent)

        from PyQt5.QtWidgets import QFrame, QToolButton, QGridLayout, QSizePolicy
        self.content_area = c = QFrame()
        self.toggle_button = b = QToolButton()
        self.main_layout = layout = QGridLayout()

        from PyQt5.QtCore import Qt
        b.setStyleSheet("QToolButton { border: none; font: 14px; }")
        # TODO: Could not figure out a way to reduce left padding on disclosure arrow on Mac Qt 5.9
#        b.setAttribute(Qt.WA_LayoutUsesWidgetRect)  # Avoid extra padding on Mac
#        b.setStyleSheet("QToolButton { margin: 0; padding-left: 0px; border: none; font: 14px; }")
#        b.setMaximumSize(20,10)
#        b.setContentsMargins(0,0,0,0)
        b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        b.setArrowType(Qt.RightArrow)
        b.setText(str(title))
        b.setCheckable(True)
        b.setChecked(False)

#        c.setStyleSheet("QFrame { background-color: white; border: none; }")
        c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # start out collapsed
        c.setMaximumHeight(0)
        c.setMinimumHeight(0)

        # don't waste space
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        row = 0
        layout.addWidget(b, row, 0, 1, 1, Qt.AlignLeft)
        row += 1
        layout.addWidget(c, row, 0, 1, 3)
        self.setLayout(layout)


        b.clicked.connect(self.toggle_panel_display)

    def toggle_panel_display(self, checked):
        from PyQt5.QtCore import Qt
        arrow_type = Qt.DownArrow if checked else Qt.RightArrow
        self.toggle_button.setArrowType(arrow_type)
        c = self.content_area
        h = c.sizeHint().height() if checked else 0
        c.setMaximumHeight(h)
        c.setMinimumHeight(h)
        if not checked:
            # Resize dock widget to reclaim spaced used by popup.
            _resize_dock_widget(self)

def _resize_dock_widget(child_widget):
    # Qt 5.12.4 is pretty screwed up in allowing QDockWidget to resize in a simple way.
    child_widget.adjustSize()
    child_widget.resize(child_widget.sizeHint())
    p = _dock_widget_parent(child_widget)
    if p:
        p.adjustSize()
        from PyQt5.QtCore import Qt
        p.window().resizeDocks([p], [p.widget().sizeHint().height()], Qt.Vertical)

def _dock_widget_parent(widget):
    from PyQt5.QtWidgets import QDockWidget
    if isinstance(widget, QDockWidget):
        return widget
    p = widget.parent()
    if p is None:
        return p
    return _dock_widget_parent(p)
