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
    """Base class (and common API) for all options"""

    multiple_value = "-- multiple --"
    read_only = False

    def __init__(self, name, default, callback, *, balloon=None, attr_name=None, **kw):
        """'callback' can be None"""

        # non-empty name overrides any default name
        if name or not hasattr(self, 'name'):
            self.name = name

        if attr_name:
            self.attr_name = attr_name
        else:
            if self.name:
                self.attr_name = self.name
            else:
                self.attr_name = None

        self._callback = callback
        if default != None or not hasattr(self, 'default'):
            self.default = default

        self._make_widget(**kw)

        if balloon or not hasattr(self, 'balloon'):
            self.balloon = balloon
        command_line = False
        if hasattr(self, 'in_class'):
            command_line = self.attr_name and '.' not in self.attr_name
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
                    if not isinstance(BooleanOption) or self.labels != BooleanOption.labels:
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

        if self.default is not None:
            self.value = self.default

        self._enabled = True
        if self.read_only:
            self.disable()

    @abstractmethod
    def get(self):
        # return the option's value
        pass

    @abstractmethod
    def set(self, value):
        # set the option's value
        pass

    def get_attribute(self, obj):
        if not self.attr_name:
            raise ValueError("No attribute associated with %s" % repr(self))
        fetcher = recurse_getattr if '.' in self.attr_name else getattr
        return fetcher(obj, self.attr_name)

    def set_attribute(self, obj):
        if not self.attr_name:
            raise ValueError("No attribute associated with %s" % repr(self))
        setter = recurse_setattr if '.' in self.attr_name else setattr
        setter(obj, self.attr_name, self.get())

    @abstractmethod
    def set_multiple(self):
        # indicate that the items the option covers have multiple different values
        pass

    def _get_value(self):
        return self.get()

    def _set_value(self, val):
        self.set(val)

    value = property(_get_value, _set_value)

    def enable(self):
        # put widget in 'enabled' (active) state
        self.widget.setDisabled(False)

    def disable(self):
        # put widget in 'disabled' (inactive) state
        self.widget.setDisabled(True)

    def make_callback(self):
        # Called (usually by GUI) to propagate changes back to program
        if self._callback:
            self._callback(self)

    @abstractmethod
    def _make_widget(self):
        # create (as self.widget) the widget to display the option value
        pass

def recurse_getattr(obj, attr_name):
    attrs = attr_name.split('.')
    for a in attrs:
        obj = getattr(obj, a)
    return obj

def recurse_setattr(obj, attr_name, val):
    attrs = attr_name.split('.')
    for a in attrs[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attrs[-1], val)

class EnumOption(Option):
    """Option for enumerated values"""
    values = ()

    def get(self):
        return self.widget.text()

    def remake_menu(self, labels=None):
        from PyQt5.QtWidgets import QAction
        if labels is None:
            labels = self.values
        menu = self.widget.menu()
        menu.clear()
        for label in labels:
            action = QAction(label, self.widget)
            action.triggered.connect(lambda arg, s=self, lab=label: s._menu_cb(lab))
            menu.addAction(action)

    def set(self, value):
        self.widget.setText(value)

    def set_multiple(self):
        self.widget.setText(self.multiple_value)

    def _make_widget(self, *, display_value=None, **kw):
        from PyQt5.QtWidgets import QPushButton, QMenu
        if display_value is None:
            display_value = self.default
        self.widget = QPushButton(display_value, **kw)
        menu = QMenu()
        self.widget.setMenu(menu)
        self.remake_menu()

    def _menu_cb(self, label):
        self.set(label)
        self.make_callback()

class RgbaOption(Option):
    """Option for rgba colors"""

    def get(self):
        return self.widget.color

    def set(self, value):
        """Accepts a wide variety of values, not just rgba"""
        self.widget.color = value

    def set_multiple(self):
        self.widget.color = None

    def _make_widget(self, **kw):
        from ..widgets import MultiColorButton
        self.widget = MultiColorButton(max_size=(16,16), has_alpha_channel=True)
        self.widget.color_changed.connect(lambda c, s=self: s.make_callback())

class ColorOption(RgbaOption):
    """Option for rgba colors"""

    def get(self):
        from ...colors import Color
        return Color(rgba=RgbaOption.get(self))

class SymbolicEnumOption(EnumOption):
    """Option for enumerated values with symbolic names"""
    values = ()
    labels = ()

    def get(self):
        return self._value

    def remake_menu(self):
        EnumOption.remake_menu(self, labels=self.labels)

    def set(self, value):
        self._value = value
        self.widget.setText(self.labels[list(self.values).index(value)])

    def set_multiple(self):
        self._value = None
        EnumOption.set_multiple(self)

    def make_callback(self):
        label = self.widget.text()
        i = list(self.labels).index(label)
        self._value = self.values[i]
        EnumOption.make_callback(self)

    def _make_widget(self, **kw):
        self._value = self.default
        EnumOption._make_widget(self,
            display_value=self.labels[list(self.values).index(self.default)], **kw)

    def _menu_cb(self, label):
        self.set(self.values[self.labels.index(label)])
        self.make_callback()
