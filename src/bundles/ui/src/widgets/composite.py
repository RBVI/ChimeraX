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


def button_row(parent, name_and_callback_list,
               label = '', spacing = 3, margins = None, button_list = False):
    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton
    f = QFrame(parent)
#    f.setStyleSheet('QFrame { background-color: pink; padding-top: 0px; padding-bottom: 0px;}')
    parent.layout().addWidget(f)

    layout = QHBoxLayout(f)
    if margins is None:
        margins = (0,0,0,0)
        import sys
        if sys.platform == 'darwin':
            margins = (0,0,0,15)  # Qt 5.15 on macOS 10.15.7 needs more space
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)

    if label:
        l = QLabel(label, f)
        layout.addWidget(l)
#    l.setStyleSheet('QLabel { background-color: pink;}')
    
#    from Qt.QtCore import Qt
    buttons = []
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
        buttons.append(b)

    layout.addStretch(1)

    if button_list:
        return f, buttons

    return f

def row_frame(parent, spacing = 5):
    from Qt.QtWidgets import QFrame, QHBoxLayout
    f = QFrame(parent)
    parent.layout().addWidget(f)

    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(spacing)
    return f, layout

class EntriesRow:
    def __init__(self, parent, *args, spacing = 5):
        f, layout = row_frame(parent, spacing)
        self.frame = f

        from .color_button import ColorButton
        from Qt.QtWidgets import QLabel, QPushButton
        self.values = values = []
        self.labels = labels = []
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
                    labels.append(l)
                    if newline:
                        # String ends in newline so make new row.
                        layout.addStretch(1)
                        f, layout = row_frame(parent)
            elif isinstance(a, bool):
                cb = BooleanEntry(f, a)
                layout.addWidget(cb.widget)
                import sys
                if sys.platform == 'darwin':
                    layout.addSpacing(5)  # Fix checkbuttons spacing problem macOS 10.15.7
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
                if len(a) >= 2 and len([a for s in a if isinstance(s,str)]) == len(a):
                    # Menu with fixed set of string values
                    me = MenuEntry(f, a)
                    layout.addWidget(me.widget)
                    values.append(me)
                else:
                    # Button with callback function
                    b = QPushButton(a[0], f)
                    layout.addWidget(b)
                    b.clicked.connect(a[1])
# TODO: QPushButton has extra vertical space (about 5 pixels top and bottom) on macOS 10.14 (Mojave)
#       with Qt 5.12.4.  Couldn't find any thing to fix this, although below are some attempts
#                b.setMaximumSize(100,25)
#                from Qt.QtWidgets import QToolButton
#                b = QToolButton(f)
#                b.setText(a[0])
#                from Qt.QtCore import Qt
#                b.setContentsMargins(0,0,0,0)
#                b.setAttribute(Qt.WA_LayoutUsesWidgetRect)
#                b.setStyleSheet('padding-left: 15px; padding-right: 15px; padding-top: 4px;  padding-bottom: 4px;')
#                b.setStyleSheet('QPushButton { border: none;}')
#                b.setStyleSheet('QPushButton { background-color: pink;}')
#                layout.setAlignment(b, Qt.AlignTop)
            elif a is ColorButton:
                cb = ColorButton(f, max_size = (20,20))
                layout.addWidget(cb)
                values.append(cb)

        layout.addStretch(1)    # Extra space at end

def radio_buttons(*check_boxes):
    for cb in check_boxes:
        cb.widget.toggled.connect(lambda state, cb=cb, others=check_boxes: _uncheck_others(cb, others))

def _uncheck_others(check_box, other_check_boxes):
    if check_box.enabled:
        for ocb in other_check_boxes:
            if ocb is not check_box and ocb.enabled:
                ocb.enabled = False
    else:
        # If all checkboxes are off, then turn this one back on.
        # This prevents turning all radio buttons off.
        if len([ocb for ocb in other_check_boxes if ocb.enabled]) == 0:
            check_box.enabled = True

class StringEntry:
    def __init__(self, parent, value = '', pixel_width = 100):
        from Qt.QtWidgets import QLineEdit
        self._line_edit = le = QLineEdit(value, parent)
        le.setMaximumWidth(pixel_width)
        self.return_pressed = le.returnPressed
    def _get_value(self):
        return self._line_edit.text()
    def _set_value(self, value):
        self._line_edit.setText(value)
    value = property(_get_value, _set_value)
    def _get_pixel_width(self):
        return self._line_edit.maximumWidth()
    def _set_pixel_width(self, width):
        self._line_edit.setMaximumWidth(width)
        self._line_edit.setMinimumWidth(width)
    pixel_width = property(_get_pixel_width, _set_pixel_width)
    @property
    def widget(self):
        return self._line_edit

class NumberEntry:
    format = '%d'
    string_to_value = int
    def __init__(self, parent, value, pixel_width = 50):
        from Qt.QtWidgets import QLineEdit
        self._line_edit = le = QLineEdit(self.format % value, parent)
        le.setMaximumWidth(pixel_width)
        self.return_pressed = le.returnPressed
    def _get_value(self):
        return self.string_to_value(self._line_edit.text())
    def _set_value(self, value):
        v = self.format % value if value is not None else ''
        self._line_edit.setText(v)
    value = property(_get_value, _set_value)
    def _get_pixel_width(self):
        return self._line_edit.maximumWidth()
    def _set_pixel_width(self, width):
        self._line_edit.setMaximumWidth(width)
    pixel_width = property(_get_pixel_width, _set_pixel_width)
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
        from Qt.QtWidgets import QCheckBox
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

class MenuEntry:
    def __init__(self, parent, values):
        from Qt.QtWidgets import QPushButton, QMenu
        self._button = b = QPushButton(parent)
        b.setText(values[0])
        m = QMenu(b)
        for value in values:
            m.addAction(value)
        b.setMenu(m)
        m.triggered.connect(self._menu_selection_cb)
    def _menu_selection_cb(self, action):
        self.value = action.text()
    def _get_value(self):
        return self._button.text()
    def _set_value(self, value):
        self._button.setText(value)
    value = property(_get_value, _set_value)
    @property
    def widget(self):
        return self._button

from Qt.QtWidgets import QWidget
class CollapsiblePanel(QWidget):
    def __init__(self, parent=None, title='', margins = None):
        QWidget.__init__(self, parent=parent)

        from Qt.QtWidgets import QFrame, QToolButton, QGridLayout, QSizePolicy

        # Setup vertical layout for content area.
        self.content_area = c = QFrame(self)
        from Qt.QtWidgets import QVBoxLayout
        clayout = QVBoxLayout(c)
        if margins is None:
            margins = (0,0,0,0) if title is None else (30,0,0,0)
        clayout.setContentsMargins(*margins)
        import sys
        if sys.platform == 'darwin':
            clayout.setSpacing(0)  # Avoid very large spacing Qt 5.15.2, macOS 10.15.7

        # Use grid layout for disclosure button and content.
        self.main_layout = layout = QGridLayout(self)

        if title is None:
            b = None
        else:
            b = QToolButton(self)
            from Qt.QtCore import Qt
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
            b.clicked.connect(self.toggle_panel_display)
        self.toggle_button = b
        
#        c.setStyleSheet("QFrame { background-color: white; border: none; }")
        c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # start out collapsed
        c.setMaximumHeight(0)
        c.setMinimumHeight(0)

        # don't waste space
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        row = 0
        if b:
            layout.addWidget(b, row, 0, 1, 1, Qt.AlignLeft)
            row += 1
        layout.addWidget(c, row, 0, 1, 3)

    @property
    def shown(self):
        return self.content_area.maximumHeight() > 0
    
    def toggle_panel_display(self, checked = None):
        if checked is None:
            checked = not self.shown
        tb = self.toggle_button
        if tb:
            from Qt.QtCore import Qt
            arrow_type = Qt.DownArrow if checked else Qt.RightArrow
            tb.setArrowType(arrow_type)
        self.resize_panel(checked)

    def resize_panel(self, shown = None):
        if shown is None:
            shown = self.shown
        c = self.content_area
        h = c.sizeHint().height() if shown else 0
        c.setMaximumHeight(h)
        c.setMinimumHeight(h)
        if not shown:
            # Resize dock widget to reclaim spaced used by popup.
            _resize_dock_widget(self)

def _resize_dock_widget(child_widget):
    # Qt 5.12.4 is pretty screwed up in allowing QDockWidget to resize in a simple way.
    child_widget.adjustSize()
    child_widget.resize(child_widget.sizeHint())
    p = _dock_widget_parent(child_widget)
    if p:
        p.adjustSize()
        main_win = p.window()
        # For undocked dock widgets this will not be a QMainWindow,
        # no need to resize all dock widgets.
        from Qt.QtWidgets import QMainWindow
        if isinstance(main_win, QMainWindow):
            from Qt.QtCore import Qt
            main_win.resizeDocks([p], [p.widget().sizeHint().height()], Qt.Vertical)

def _dock_widget_parent(widget):
    from Qt.QtWidgets import QDockWidget
    if isinstance(widget, QDockWidget):
        return widget
    p = widget.parent()
    if p is None:
        return p
    return _dock_widget_parent(p)

class ModelMenu:
    '''Menu of session models prefixed with a text label.'''
    def __init__(self, session, parent, label = None,
                 model_types = None, model_filter = None,
                 model_chosen_cb = None, special_items = []):

        from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
        self.frame = f = QFrame(parent)
        layout = QHBoxLayout(f)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(10)

        if label:
            fl = QLabel(label, f)
            layout.addWidget(fl)

        class_filter = None if model_types is None else tuple(model_types)
        filter_func = (lambda model: True) if model_filter is None else model_filter
        from chimerax.ui.widgets import ModelMenuButton
        sm = ModelMenuButton(session, class_filter = class_filter,
                             filter_func = filter_func,
                             special_items = special_items, parent = f)
        self._menu = sm
        
        mlist = [m for m in session.models.list(type = class_filter) if filter_func(m)]
        mdisp = [m for m in mlist if m.visible]
        if mdisp:
            sm.value = mdisp[0]
        elif mlist:
            sm.value = mlist[0]

        if model_chosen_cb:
            sm.value_changed.connect(model_chosen_cb)

        layout.addWidget(sm)

        layout.addStretch(1)    # Extra space at end

    def _get_value(self):
        return self._menu.value
    def _set_value(self, value):
        self._menu.value = value
    value = property(_get_value, _set_value)

def vertical_layout(frame, margins = (0,0,0,0), spacing = 0):
    from Qt.QtWidgets import QVBoxLayout
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    frame.setLayout(layout)
    return layout

def horizontal_layout(frame, margins = (0,0,0,0), spacing = 0):
    from Qt.QtWidgets import QHBoxLayout
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    frame.setLayout(layout)
    return layout
