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

from abc import ABCMeta, abstractmethod

class Option(metaclass=ABCMeta):
    """Supported API. Base class (and common API) for all options"""

    multiple_value = "-- multiple --"
    read_only = False

    def __init__(self, name, default, callback, *, balloon=None, attr_name=None, settings=None,
            auto_set_attr=True, **kw):
        """'callback' can be None"""

        # non-empty name overrides any default name
        if name or not hasattr(self, 'name'):
            self.name = name

        if attr_name:
            self.attr_name = attr_name
        elif not hasattr(self, 'attr_name'):
            if self.name:
                self.attr_name = self.name
            else:
                self.attr_name = None

        if settings is None:
            self.settings_handler = self.settings = None
        else:
            if self.attr_name is None:
                raise ValueError("'settings' specified but not 'attr_name' (required for 'settings')")
            # weakrefs are unhashable, which causes a problem in the container code that
            # tries to organize options by settings before saving, so just use a strong
            # reference to settings for now; if that proves problematic then revisit.
            self.settings = settings
            from weakref import proxy
            self.settings_handler = self.settings.triggers.add_handler('setting changed',
                lambda trig_name, data, *, pself=proxy(self):
                data[0] == pself.attr_name and (setattr(pself, "value", pself.get_attribute())
                or (pself._callback and pself._callback(pself))))
        self.auto_set_attr = auto_set_attr

        if default is None and attr_name and settings:
            self.default = getattr(settings, attr_name)
        if default is not None or not hasattr(self, 'default'):
            self.default = default

        self._make_widget(**kw)

        if balloon or not hasattr(self, 'balloon'):
            self.balloon = balloon
        command_line = self.attr_name and not self.settings
        if self.balloon or command_line:
            balloon = self.balloon if self.balloon else ""
            if command_line:
                attr_balloon = "attribute name: " + self.attr_name
                attr_vals = None
                if hasattr(self, "attr_values_balloon"):
                    attr_balloon += '\n' + self.attr_values_balloon
                elif hasattr(self, 'mapping'):
                    attr_vals = self.mapping.items()
                elif hasattr(self, 'labels') and hasattr(self, 'values'):
                    if not isinstance(self, BooleanOption) or self.labels != BooleanOption.labels:
                        attr_vals = zip(self.values, self.labels)
                if attr_vals:
                    attr_balloon += '\n'
                    vals_text = "values: "
                    for val, vtext in attr_vals:
                        if vals_text != "values: ":
                            vals_text += ", "
                        vals_text += str(val)
                        vals_text += " " + vtext
                        if len(vals_text) > 32:
                            attr_balloon += vals_text
                            attr_balloon += '\n'
                            vals_text = ""
                    if vals_text:
                        attr_balloon += vals_text
                    else:
                        attr_balloon = attr_balloon[:-1]
                if balloon:
                    balloon += "\n\n"
                balloon += attr_balloon
                self.balloon = balloon

        # prevent showing the default from making a callback...
        self._callback = None
        if self.default is not None:
            self.value = self.default
        self._callback = callback

        if self.read_only:
            self.enabled = False

    def __del__(self):
        if self.settings_handler:
            self.settings_handler.remove()
            self.settings_handler = None

    def display_for_items(self, items):
        """Supported API.  Use the option's 'attr_name' attribute to survey the given items for
           their value or values for that attribute and display the value or values in the option.
           The 'items' can be a chimerax.atomic.Collection or a normal Python sequence. If a Collection,
           the "plural form" of attr_name will be used to check the Collection.
        """
        values = self.values_for_items(items)
        if len(values) == 0:  # 'not values' fails when it's a numpy array
            return
        from numpy import array_equal
        value = values[0]
        for val in values[1:]:
            if not array_equal(val, value):
                self.set_multiple()
                return
        self.value = value

    def values_for_items(self, items):
        """Supported API.  Convenience function to get values for the 'attr_name' attribute from the
           given items.  Used by display_for_items() and by subclasses overriding display_for_items().
        """
        if not items:
            return []
        from chimerax.atomic import Collection
        if isinstance(items, Collection):
            from chimerax.core.commands import plural_of
            values = getattr(items, plural_of(self.attr_name))
        else:
            values = [getattr(i, self.attr_name) for i in items]
        return values

    @property
    def enabled(self):
        """Supported API. Enable/disable the option"""
        return self.widget.isEnabled()

    @enabled.setter
    def enabled(self, enable):
        # usually no need to override, since enabling/disabling
        # a Qt widget implicitly does the same for its children
        self.widget.setEnabled(enable)

    def get_attribute(self):
        """Supported API. Get the attribute associated with this option ('attr_name' in constructor)
        from the option's settings attribute"""
        if not self.attr_name:
            raise ValueError("No attribute associated with %s" % repr(self))
        if self.settings is None:
            raise ValueError("No settings/object known to fetch attribute %s from, via option %s"
                    % (self.attr_name, repr(self)))
        return getattr(self.settings, self.attr_name)

    def set_attribute(self):
        """Supported API. Set the attribute associated with this option ('attr_name' in constructor)
        from the option's settings attribute"""
        if not self.attr_name:
            raise ValueError("No attribute associated with %s" % repr(self))
        if self.settings is None:
            raise ValueError("No settings/object known to set attribute %s in, via option %s"
                    % (self.attr_name, repr(self)))
        setattr(self.settings, self.attr_name, self.value)

    @abstractmethod
    def set_multiple(self):
        """Supported API. Indicate that the items the option covers have multiple different values"""
        pass

    # no "shown" property because the option is in a QFormLayout and there is no way to hide a row,
    # not to mention that hiding our widget doesn't hide the corresponding label

    # In Python 3.7, abstract properties where the getter/setter funcs have the same name don't
    # work as expected in derived classes; use old-style property definition

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def set_value(self, val):
        pass

    value = property(get_value, set_value,
        doc="Supported API. Get/set the option's value; when set it should NOT invoke the callback")

    def make_callback(self):
        """Supported API. Called (usually by GUI) to propagate changes back to program"""
        if self.attr_name and self.settings and self.auto_set_attr:
            # the attr-set callback will call _callback()
            self.set_attribute()
        elif self._callback:
            self._callback(self)

    @abstractmethod
    def _make_widget(self):
        # Create (as self.widget) the widget to display the option value.
        # The "widget" can actually be a layout, if several widgets are needed
        # to compose/display the value of the option
        pass

def make_optional(cls):
    def get_value(self):
        if self._check_box.isChecked():
            # substitute the original widget back before asking for the value
            self.widget, self._orig_widget = self._orig_widget, self.widget
            val = self._super_class.value.fget(self)
            self.widget, self._orig_widget = self._orig_widget, self.widget
            return val
        return None

    def set_value(self, value):
        if value is None:
            self._check_box.setChecked(False)
            self.widget, self._orig_widget = self._orig_widget, self.widget
            self._super_class.enabled.fset(self, False)
            self.widget, self._orig_widget = self._orig_widget, self.widget
        else:
            self._check_box.setChecked(True)
            self.widget, self._orig_widget = self._orig_widget, self.widget
            self._super_class.enabled.fset(self, True)
            self._super_class.value.fset(self, value)
            self.widget, self._orig_widget = self._orig_widget, self.widget

    def set_multiple(self):
        self._check_box.setChecked(True)
        self.widget, self._orig_widget = self._orig_widget, self.widget
        self._super_class.set_multiple(self)
        self.widget, self._orig_widget = self._orig_widget, self.widget

    def _make_widget(self, **kw):
        self._super_class._make_widget(self, **kw)
        self._orig_widget = self.widget
        from Qt.QtWidgets import QCheckBox, QHBoxLayout, QLayout
        self.widget = layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        from Qt.QtCore import Qt
        self._check_box = cb = QCheckBox()
        cb.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        def enable_and_call(s=self):
            s.widget, s._orig_widget = s._orig_widget, s.widget
            s._super_class.enabled.fset(s, s._check_box.isChecked())
            s.widget, s._orig_widget = s._orig_widget, s.widget
            s.make_callback()
        cb.clicked.connect(lambda *args, s=self: enable_and_call(s))
        layout.addWidget(cb, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        if isinstance(self._orig_widget, QLayout):
            layout.addLayout(self._orig_widget, stretch=1)
        else:
            layout.addWidget(self._orig_widget, stretch=1, alignment=Qt.AlignLeft | Qt.AlignVCenter)

    attr_dict = {
        'value': property(get_value, set_value),
        'set_multiple': set_multiple,
        '_make_widget': _make_widget,
        '_super_class': cls,
    }
    opt_class = type('Optional' + cls.__name__, (cls,), attr_dict)
    return opt_class


class NumericOption(Option):
    """base class for options that display single numbers (e.g. IntOption, FloatOption)"""

    def display_for_items(self, items):
        """Supported API.  Use the option's 'attr_name' attribute to survey the given items for
           their value or values for that attribute and display the value or values in the option.
           The 'items' can be a chimerax.atomic.Collection or a normal Python sequence. If a Collection,
           the "plural form" of attr_name will be used to check the Collection.  Ranges will be
           shown appropriately.
        """
        values = self.values_for_items(items)
        if len(values) == 0:  # 'not values' fails when it's a numpy array
            return
        from numbers import Real
        num_vals = [val for val in values if isinstance(val, Real)]
        if not num_vals:
            self.show_text("N/A")
        else:
            min_val = min(num_vals)
            max_val = max(num_vals)
            if ("%g" % max_val) == ("%g" % min_val):
                self.value = max_val
            else:
                decimal_places = getattr(self, 'decimal_places', None)
                decimal_format = "%g" if not decimal_places else "%%.%df" % decimal_places
                self.show_text((decimal_format + " \N{LEFT RIGHT ARROW} " + decimal_format)
                    % (min_val, max_val))

    @abstractmethod
    def show_text(self, text):
        """So that option can show text such as 'N/A' and number ranges"""
        pass

class BooleanOption(Option):
    """Supported API. Option for true/false values"""

    def get_value(self):
        return self.widget.isChecked()

    def set_value(self, value):
        self.widget.setChecked(value)

    value = property(get_value, set_value)

    def set_multiple(self):
        from Qt.QtCore import Qt
        self.widget.setCheckState(Qt.PartiallyChecked)

    def _make_widget(self, as_group=False, **kw):
        from Qt.QtWidgets import QCheckBox, QGroupBox
        if as_group:
            self.widget = QGroupBox(self.name)
            self.name = ""
            self.widget.setCheckable(True)
        else:
            self.widget = QCheckBox(**kw)
        self.widget.clicked.connect(self.make_callback)

class EnumBase(Option):
    values = ()
    def get_value(self):
        if self.__as_radio_buttons:
            return self.values[self.__button_group.checkedId()]
        button_text = self.widget.text()
        if isinstance(self, SymbolicEnumOption):
            return self.values[self.labels.index(button_text)]
        return button_text

    def set_value(self, value):
        if self.__as_radio_buttons:
            self.__button_group.button(self.values.index(value)).setChecked(True)
        elif isinstance(self, SymbolicEnumOption):
            self.widget.setText(self.labels[self.values.index(value)])
        else:
            self.widget.setText(value)

    value = property(get_value, set_value)

    def set_multiple(self):
        if not self.__as_radio_buttons:
            self.widget.setText(self.multiple_value)

    def remake_menu(self, *, make_callback=True):
        from Qt.QtWidgets import QRadioButton
        from Qt.QtGui import QAction
        from Qt.QtCore import Qt
        if isinstance(self, SymbolicEnumOption):
            labels = self.labels
        else:
            labels = self.values
        if self.__as_radio_buttons:
            for b in self.__button_group.buttons():
                self.__button_group.removeButton(b)
                b.hide()
                b.destroy()
            layout = self.widget.layout()
            for i, l in enumerate(labels):
                b = QRadioButton(l)
                layout.addWidget(b, alignment=Qt.AlignLeft)
                self.__button_group.addButton(b, id=i)
        else:
            menu = self.widget.menu()
            menu.clear()
            for label, value in zip(labels, self.values):
                menu_label = label.replace('&', '&&')
                action = QAction(menu_label, self.widget)
                action.triggered.connect(lambda *, s=self, val=value: s._menu_cb(val))
                menu.addAction(action)
            if self.values and self.value not in self.values and self.value != self.multiple_value:
                self.value = labels[0]
                if make_callback:
                    self.make_callback()
    remake_buttons = remake_menu

    def _make_widget(self, *, as_radio_buttons=False, display_value=None, **kw):
        from Qt.QtWidgets import QPushButton, QMenu, QWidget, QButtonGroup, QVBoxLayout
        self.__as_radio_buttons = as_radio_buttons
        if as_radio_buttons:
            self.widget = QWidget()
            layout = QVBoxLayout()
            self.widget.setLayout(layout)
            self.__button_group = QButtonGroup()
            self.remake_buttons()
            self.__button_group.button(self.values.index(self.default)).setChecked(True)
            from Qt import using_pyqt6
            if using_pyqt6:
                self.__button_group.idClicked.connect(self.make_callback)
            else:
                self.__button_group.buttonClicked[int].connect(self.make_callback)
        else:
            if display_value is not None:
                button_label = display_value
            elif isinstance(self, SymbolicEnumOption):
                button_label = self.labels[self.values.index(self.default)]
            else:
                button_label = self.default
            self.widget = QPushButton(button_label, **kw)
            self.widget.setAutoDefault(False)
            menu = QMenu(self.widget)
            self.widget.setMenu(menu)
            self.remake_menu(make_callback=False)
        return self.widget

    def _menu_cb(self, value):
        self.value = value
        self.make_callback()

class EnumOption(EnumBase):
    """Supported API. Option for enumerated values.
       The given values will be displayed in the interface and returned by the 'value' attribute.
       If you want to display different text in the interface than the literal value, use the
       SymbolicEnumOption.  You can specify values either by subclassing and overriding the 'values'
       class attribute, or by supplying the 'values' keyword to the constructor.
    """
    def __init__(self, *args, values=None, **kw):
        if values is not None:
            self.values = values
        super().__init__(*args, **kw)

OptionalEnumOption = make_optional(EnumOption)

class FloatOption(NumericOption):
    """Supported API. Option for floating-point values.
       Constructor takes option min/max keywords to specify lower/upper bound values.
       Besides being numeric values, those keyords can also be 'positive' or 'negative'
       respectively, in which case the allowed value can be arbitrarily close to zero but
       cannot be equal to zero.

       'decimal_places' indicates allowable number of digits after the decimal point
       (default: 3).  Values with more digits will be rounded.  If the widget provides
       a means to increment the value (e.g. up/down arrow) then 'step' is how much the
       value will be incremented (default: 10x the smallest value implied by 'decimal_places').
       
       if 'as_slider' is True, then a slider widget will be used instead of an entry widget.
       If using a slider, it is recommended to set 'min' and 'max' values; otherwise the
       widget will cover a very large numeric range.  When using a slider, you can specify
       'continuous_callback' as True/False (default False) to control whether the option's
       callback happens as the slider is dragged, or just when the slider is released.

       Supports 'left_text' and 'right_text' keywords for putting text before
       and after the entry widget on the right side of the form, or below left/right
       of the slider if using a slider."""

    def get_value(self):
        return self._float_widget.value()

    def set_value(self, value):
        self._float_widget.set_text("")
        self._float_widget.blockSignals(True)
        self._float_widget.setValue(value)
        self._float_widget.blockSignals(False)

    value = property(get_value, set_value)

    def set_multiple(self):
        self.show_text(self.multiple_value)

    def show_text(self, text):
        self._float_widget.set_text(text)

    def _possible_callback(self):
        if self._value_has_changed:
            self._value_has_changed = False
            self.make_callback()

    def _make_widget(self, *, min=None, max=None, left_text=None, right_text=None,
            decimal_places=3, step=None, as_slider=False, **kw):
        self.decimal_places = decimal_places
        self._float_widget = _make_float_widget(min, max, step, decimal_places, as_slider=as_slider, **kw)
        self._value_has_changed = False
        if as_slider:
            self._float_widget.valueChanged.connect(lambda val, s=self: s.make_callback())
        else:
            self._float_widget.valueChanged.connect(lambda val, s=self:
                setattr(s, '_value_has_changed', True))
            self._float_widget.editingFinished.connect(lambda *, s=self: s._possible_callback())
        if (not left_text and not right_text) or as_slider:
            if left_text:
                self._float_widget.set_left_text(left_text)
            if right_text:
                self._float_widget.set_right_text(right_text)
            self.widget = self._float_widget
            return
        from Qt.QtWidgets import QWidget, QHBoxLayout, QLabel
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        if left_text:
            layout.addWidget(QLabel(left_text))
            l = 0
        layout.addWidget(self._float_widget)
        if right_text:
            layout.addWidget(QLabel(right_text))
            r = 0
        self.widget.setLayout(layout)

class FloatEnumOption(EnumBase):
    """Supported API. Option for a floating-point number and an enum (as a 2-tuple), for something
       such as size and units"""

    def get_value(self):
        return (self._float_widget.value(), self._enum.text())

    def set_value(self, value):
        float, text = value
        self._float_widget.set_text("")
        self._float_widget.setValue(float)
        self._enum.setText(text)

    value = property(get_value, set_value)

    def set_multiple(self):
        self._float_widget.set_text(self.mulitple_value)
        self._enum.setText(self.multiple_value)

    def _make_widget(self, min=None, max=None, float_label=None, enum_label=None,
            decimal_places=3, step=None, display_value=None, as_slider=False, **kw):
        self.decimal_places = decimal_places
        self._float_widget = _make_float_widget(min, max, step, decimal_places, as_slider=as_slider)
        self._float_widget.valueChanged.connect(lambda val, s=self: s.make_callback())
        self._enum = EnumBase._make_widget(self, display_value=display_value, **kw)
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        if float_label:
            layout.addWidget(QLabel(float_label))
        layout.addWidget(self._float_widget)
        if enum_label:
            layout.addWidget(QLabel(enum_label))
        layout.addWidget(self._enum)
        self.widget.setLayout(layout)

class FontOption(EnumOption):
    # setting 'values' delayed until (first) constructor, so avoid the expense of querying the font database
    # until someone actually uses this option
    values = None
    def __init__(self, *args, **kw):
        if self.values is None:
            from Qt.QtGui import QFontDatabase
            from Qt import using_qt5
            fdb = QFontDatabase() if using_qt5 else QFontDatabase
            self.values = sorted(list(fdb.families()))
            super().__init__(*args, **kw)

class FileOption(Option):
    """base class for specifying a file """

    @classmethod
    def browse_func(cls, *args, **kw):
        from Qt.QtWidgets import QFileDialog
        if cls == InputFileOption:
            return QFileDialog.getOpenFileName(*args, **kw)
        return QFileDialog.getSaveFileName(*args, **kw)

    def get_value(self):
        return self.line_edit.text()

    def set_value(self, value):
        self.line_edit.setText(value)

    value = property(get_value, set_value)

    def set_multiple(self):
        self.line_edit.setText(self.multiple_value)

    def _make_widget(self, initial_text_width="10em", start_folder=None, browser_title="Choose File", **kw):
        """initial_text_width should be a string holding a "stylesheet-friendly"
           value, (e.g. '10em' or '7ch') or None"""
        from Qt.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton
        self.widget = QWidget()
        self.widget.setContentsMargins(0,0,0,0)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.widget.setLayout(layout)
        self.line_edit = QLineEdit()
        self.line_edit.returnPressed.connect(self.make_callback)
        if initial_text_width:
            self.line_edit.setStyleSheet("* { width: %s }" % initial_text_width)
        layout.addWidget(self.line_edit, stretch=1)
        self.start_folder = start_folder
        self.browser_title = browser_title
        button = QPushButton("Browse")
        button.clicked.connect(self._launch_browser)
        layout.addWidget(button)

    def _launch_browser(self, *args):
        import os
        if self.start_folder is None or not os.path.exists(self.start_folder):
            start_folder = os.getcwd()
        else:
            start_folder = self.start_folder
        file, filter = self.browse_func(self.widget, self.browser_title, start_folder)
        if file:
            self.line_edit.setText(file)
            self.line_edit.returnPressed.emit()

class InputFileOption(FileOption):
    pass
class OutputFileOption(FileOption):
    pass

class InputFolderOption(Option):
    """Option for specifying an existing folder for input"""

    def get_value(self):
        return self.line_edit.text()

    def set_value(self, value):
        self.line_edit.setText(value)

    value = property(get_value, set_value)

    def set_multiple(self):
        self.line_edit.setText(self.multiple_value)

    def _make_widget(self, initial_text_width="10em", start_folder=None, browser_title="Choose Folder", **kw):
        """initial_text_width should be a string holding a "stylesheet-friendly"
           value, (e.g. '10em' or '7ch') or None"""
        from Qt.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton
        self.widget = QWidget()
        self.widget.setContentsMargins(0,0,0,0)
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.widget.setLayout(layout)
        self.line_edit = QLineEdit()
        self.line_edit.returnPressed.connect(self.make_callback)
        if initial_text_width:
            self.line_edit.setStyleSheet("* { width: %s }" % initial_text_width)
        layout.addWidget(self.line_edit, stretch=1)
        self.start_folder = start_folder
        self.browser_title = browser_title
        button = QPushButton("Browse")
        button.clicked.connect(self._launch_browser)
        layout.addWidget(button)

    def _launch_browser(self, *args):
        from Qt.QtWidgets import QFileDialog
        import os
        if self.start_folder is None or not os.path.exists(self.start_folder):
            start_folder = os.getcwd()
        else:
            start_folder = self.start_folder
        folder = QFileDialog.getExistingDirectory(self.widget, self.browser_title, start_folder,
            QFileDialog.ShowDirsOnly)
        if folder:
            self.line_edit.setText(folder)
            self.line_edit.returnPressed.emit()

OutputFolderOption = InputFolderOption

class IntOption(NumericOption):
    """Supported API. Option for integer values.
       Constructor takes option min/max keywords to specify lower/upper bound values.
       
       Supports 'left_text' and 'right_text' keywords for putting text before
       and after the entry widget on the right side of the form"""

    def get_value(self):
        return self._spin_box.value()

    def set_value(self, value):
        self._spin_box.setSpecialValueText("")
        self._spin_box.blockSignals(True)
        self._spin_box.setValue(value)
        self._spin_box.blockSignals(False)

    value = property(get_value, set_value)

    def set_multiple(self):
        self.show_text(self.multiple_value)

    def show_text(self, text):
        self._spin_box.blockSignals(True)
        self._spin_box.setValue(self._spin_box.minimum())
        self._spin_box.blockSignals(False)
        self._spin_box.setSpecialValueText(text)

    def _possible_callback(self):
        if self._value_has_changed:
            self._value_has_changed = False
            self.make_callback()

    def _make_widget(self, min=None, max=None, left_text=None, right_text=None, **kw):
        self._spin_box = _make_int_spinbox(min, max, **kw)
        self._value_has_changed = False
        self._spin_box.valueChanged.connect(lambda val, s=self: setattr(s, '_value_has_changed', True))
        self._spin_box.editingFinished.connect(lambda *, s=self: s._possible_callback())
        if not left_text and not right_text:
            self.widget = self._spin_box
            return
        from Qt.QtWidgets import QWidget, QHBoxLayout, QLabel
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        if left_text:
            layout.addWidget(QLabel(left_text))
        layout.addWidget(self._spin_box)
        if right_text:
            layout.addWidget(QLabel(right_text))
        self.widget.setLayout(layout)

class RGBA8Option(Option):
    """Option for rgba colors, returns 8-bit (0-255) rgba values"""

    def get_value(self):
        return self.widget.color

    def set_value(self, value):
        """Accepts a wide variety of values, not just rgba"""
        self.widget.color = value

    value = property(get_value, set_value)

    def set_multiple(self):
        self.widget.color = None

    def _make_widget(self, **kw):
        from ..widgets import MultiColorButton
        self.widget = MultiColorButton(max_size=(16,16), has_alpha_channel=True)
        initial_color = kw.get('initial_color', getattr(self, 'default_initial_color', None))
        if initial_color is not None:
            self.widget.color = initial_color
        self.widget.color_changed.connect(lambda c, s=self: s.make_callback())

class RGBAOption(RGBA8Option):
    """Option for rgba colors, returns floating-point (0-1) rgba values"""

    def get_value(self):
        return [x/255.0 for x in super().value]

    value = property(get_value, RGBA8Option.set_value)

class ColorOption(RGBA8Option):
    """Option for rgba colors"""

    def get_value(self):
        from chimerax.core.colors import Color
        return Color(rgba=super().value)

    value = property(get_value, RGBA8Option.set_value)

OptionalColorOption = make_optional(ColorOption)
OptionalColorOption.default_initial_color = [0.75, 0.75, 0.75, 1.0]
OptionalRGBA8Option = make_optional(RGBA8Option)
OptionalRGBA8Option.default_initial_color = [0.75, 0.75, 0.75, 1.0]

class OptionalRGBAOption(OptionalRGBA8Option):
    """Option for floating-point (0-1) rgba colors, with possibility of None.

    Supports 'initial_color' constructor arg for initializing the color button even when
    the starting value of the option is None (checkbox will be unchecked)
    """

    def get_value(self):
        rgba8 = super().value
        if rgba8 is None:
            return None
        return [x/255.0 for x in rgba8]

    value = property(get_value, OptionalRGBA8Option.value.fset)

class OptionalRGBA8PairOption(Option):
    """Like OptionalRGBA8Option, but two checkboxes/colors

    Supports 'initial_colors' constructor arg (2-tuple of colors) for initializing the color buttons
    even when the starting value of the option is (None, None) (checkboxes will be unchecked)
    """

    def get_value(self):
        return (self._color_button[i].color if self._check_box[i].isChecked() else None
            for i in range(2) )

    def set_value(self, value):
        """2-tuple.  Accepts a wide variety of values, not just rgba"""
        for i, val in enumerate(value):
            if val is None:
                self._check_box[i].setChecked(False)
            else:
                self._check_box[i].setChecked(True)
                self._color_button[i].color = val

    value = property(get_value, set_value)

    def set_multiple(self):
        for i in range(2):
            self._check_box[i].setChecked(True)
            self._color_button[i].color = None

    def _make_widget(self, **kw):
        from ..widgets import MultiColorButton
        from Qt.QtWidgets import QWidget, QCheckBox, QHBoxLayout, QLabel
        labels = kw.pop('labels', (None, "  "))
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self._check_box = []
        self._color_button = []
        for i in range(2):
            label = labels[i]
            if label:
                layout.addWidget(QLabel(label))
            cb = QCheckBox()
            self._check_box.append(cb)
            cb.clicked.connect(self.make_callback)
            layout.addWidget(cb)
            mcb = MultiColorButton(max_size=(16,16), has_alpha_channel=True)
            self._color_button.append(mcb)
            default_color = OptionalRGBA8Option.default_initial_color
            mcb.color = kw.get('initial_colors', (default_color, default_color))[i]
            mcb.color_changed.connect(lambda c, s=self: s.make_callback())
            layout.addWidget(mcb)
        self.widget.setLayout(layout)

class OptionalRGBAPairOption(OptionalRGBA8PairOption):
    """Like OptionalRGBAOption, but two checkboxes/colors

    Supports 'initial_colors' constructor arg (2-tuple of colors) for initializing the color buttons
    even when the starting value of the option is (None, None) (checkboxes will be unchecked)
    """

    def get_value(self):
        rgba8s = super().value
        return tuple(None if i is None else [c/255.0 for c in i] for i in rgba8s)

    value = property(get_value, OptionalRGBA8PairOption.set_value)

class StringOption(Option):
    """Supported API. Option for text strings"""

    def get_value(self):
        return self.widget.text()

    def set_value(self, value):
        self.widget.setText(value)

    value = property(get_value, set_value)

    def set_multiple(self):
        self.widget.setText(self.multiple_value)

    def _make_widget(self, **kw):
        from Qt.QtWidgets import QLineEdit
        self.widget = QLineEdit(**kw)
        self.widget.editingFinished.connect(lambda *, s=self: s.make_callback())

class PasswordOption(StringOption):
    """Supported API. Option for entering a password"""

    def _make_widget(self, **kw):
        super()._make_widget(**kw)
        self.widget.setEchoMode(self.widget.PasswordEchoOnEdit)

class PhysicalSizeOption(FloatEnumOption):
    values = ("cm", "inches")

    def _make_widget(self, min=0, **kw):
        super()._make_widget(min=min, **kw)

class StringIntOption(Option):
    """Supported API. Option for a string and an int (as a 2-tuple), for something such as host and port"""

    def get_value(self):
        return (self._line_edit.text(), self._spin_box.value())

    def set_value(self, value):
        text, integer = value
        self._line_edit.setText(text)
        self._spin_box.setSpecialValueText("")
        self._spin_box.setValue(integer)

    value = property(get_value, set_value)

    def set_multiple(self):
        self._line_edit.setText(self.multiple_value)
        self._spin_box.setSpecialValueText(self.multiple_value)
        self._spin_box.setValue(self._spin_box.minimum())

    def _make_widget(self, min=None, max=None, string_label=None, int_label=None,
            initial_text_width="10em", **kw):
        """initial_text_width should be a string holding a "stylesheet-friendly"
           value, (e.g. '10em' or '7ch') or None"""
        from Qt.QtWidgets import QLineEdit
        self._line_edit = QLineEdit()
        self._line_edit.editingFinished.connect(lambda *, s=self: s.make_callback())
        if initial_text_width:
            self._line_edit.setStyleSheet("* { width: %s }" % initial_text_width)
        self._spin_box = _make_int_spinbox(min, max, **kw)
        self._spin_box.valueChanged.connect(lambda val, s=self: s.make_callback())
        from Qt.QtWidgets import QWidget, QHBoxLayout, QLabel
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        if string_label:
            layout.addWidget(QLabel(string_label))
        layout.addWidget(self._line_edit)
        if int_label:
            layout.addWidget(QLabel(int_label))
        layout.addWidget(self._spin_box)
        self.widget.setLayout(layout)

class StringsOption(Option):
    """Supported API. Option for list of plain text strings
       There is no builtin way for the user to indicate that they are done editing the text,
       so no callback will occur.  If such an indication is needed, another widget would have to
       provide it."""

    def get_value(self):
        return self.widget.toPlainText().split('\n')

    def set_value(self, value):
        self.widget.setText('\n'.join(value))

    value = property(get_value, set_value)

    def set_multiple(self):
        self.widget.setText(self.multiple_value)

    def _make_widget(self, initial_text_width="10em", initial_text_height="4em", **kw):
        """initial_text_width/height should be a string holding a "stylesheet-friendly"
           value, (e.g. '10em' or '7ch') or None"""
        from Qt.QtWidgets import QTextEdit
        self.widget = QTextEdit(**kw)
        self.widget.setAcceptRichText(False)
        self.widget.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        sheet_info = ""
        if initial_text_width:
            sheet_info = "width: %s" % initial_text_width
        else:
            sheet_info = ""
        if initial_text_height:
            if sheet_info:
                sheet_info += "; "
            sheet_info += "height: %s" % initial_text_height
        if sheet_info:
            self.widget.setStyleSheet("* { " + sheet_info + " }")

class HostPortOption(StringIntOption):
    """Supported API. Option for a host name or address and a TCP port number (as a 2-tuple)"""
    def _make_widget(self, **kw):
        StringIntOption._make_widget(self, min=0, max=65535, string_label="host", int_label="port",             **kw)


class SymbolicEnumOption(EnumOption):
    """Supported API. Option for enumerated values with symbolic names
       The given values will be returned by the 'value' attribute and the corresponding symbolic names
       will be displayed in the user interface.  If your values and symbolic names are the same, just
       use EnumOption.  You can specify values and symbolic names either by subclassing and overriding
       the 'values' and 'labels' class attributes, or by supplying the 'values' and 'labels' keywords
       to the constructor.
    """
    values = ()
    labels = ()
    def __init__(self, *args, labels=None, **kw):
        if labels is not None:
            self.labels = labels
        super().__init__(*args, **kw)

OptionalSymbolicEnumOption = make_optional(SymbolicEnumOption)

def _make_float_widget(min, max, step, decimal_places, *, as_slider=False, continuous_callback=False, **kw):
    def compute_bound(bound, default_bound):
        if bound is None:
            return default_bound
        if bound in ('positive', 'negative'):
            return 0.0
        return bound
    default_minimum = -(2**31)
    default_maximum = 2**31 - 1
    minimum = compute_bound(min, default_minimum)
    maximum = compute_bound(max, default_maximum)
    if step is None:
        step = 10 ** (0 - (decimal_places-1))

    if as_slider:
        from chimerax.ui.widgets import FloatSlider
        return FloatSlider(minimum, maximum, step, decimal_places, continuous_callback, **kw)
    # as spinbox...
    from Qt.QtWidgets import QDoubleSpinBox
    class NZDoubleSpinBox(QDoubleSpinBox):
        def value(self):
            val = super().value()
            if val == 0.0 and self.non_zero:
                step = self.singleStep()
                if self.minimum() == 0.0:
                    val = step
                else:
                    val = -step
            return val

        def event(self, event):
            ret = super().event(event)
            if event.type() in [event.Type.KeyPress, event.Type.KeyRelease]:
                event.accept()
                return True
            return ret

        def eventFilter(self, source, event):
            # prevent scroll wheel from changing value (usually accidentally)
            if event.type() == event.Type.Wheel and source is self:
                event.ignore()
                return True
            return super().eventFilter(source, event)

        def set_text(self, text):
            self.blockSignals(True)
            self.setValue(self.minimum())
            self.blockSignals(False)
            self.setSpecialValueText(text)

        def special_value_shown(self):
            return self.specialValueText() != ""

        def stepBy(self, *args, **kw):
            super().stepBy(*args, **kw)
            self.editingFinished.emit()

    spin_box = NZDoubleSpinBox(**kw)
    spin_box.non_zero = (max == 'negative' or min == 'positive')
    spin_box.setDecimals(decimal_places)
    spin_box.setMinimum(minimum)
    spin_box.setMaximum(maximum)
    spin_box.setSingleStep(step)
    from Qt.QtCore import Qt
    spin_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    spin_box.installEventFilter(spin_box)
    return spin_box

def _make_int_spinbox(min, max, **kw):
    from Qt.QtWidgets import QSpinBox
    class NoScrollSpinBox(QSpinBox):
        def event(self, event):
            ret = super().event(event)
            if event.type() in [event.Type.KeyPress, event.Type.KeyRelease]:
                event.accept()
                return True
            return ret

        def eventFilter(self, source, event):
            # prevent scroll wheel from changing value (usually accidentally)
            if event.type() == event.Type.Wheel and source is self:
                event.ignore()
                return True
            return super().eventFilter(source, event)

        def stepBy(self, *args, **kw):
            super().stepBy(*args, **kw)
            self.editingFinished.emit()

    spin_box = NoScrollSpinBox(**kw)
    default_minimum = -(2**31)
    default_maximum = 2**31 - 1
    spin_box.setMinimum(default_minimum if min is None else min)
    spin_box.setMaximum(default_maximum if max is None else max)
    from Qt.QtCore import Qt
    spin_box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    spin_box.installEventFilter(spin_box)
    return spin_box
