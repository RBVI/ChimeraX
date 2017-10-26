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

from PyQt5.QtWidgets import QWidget, QFormLayout

class OptionsPanel(QWidget):
    """OptionsPanel is a container for Options"""

    def __init__(self, parent=None, *, sorting=None, **kw):
        """sorting:
             None; options shown in order added
             True: options sorted alphabetically by name
             func: options sorted based on the provided key function
        """
        QWidget.__init__(self, parent, **kw)
        self._sorting = sorting
        self._options = []
        self.setLayout(QFormLayout())

    def add_option(self, option):
        if self._sorting is None:
            insert_row = len(self._options)
        else:
            if self._sorting is True:
                test = lambda o1, o2: o1.name < o2.name
            else:
                test = lambda o1, o2: self._sorting(o1) < self._sorting(o2)
            for insert_row in range(len(self._options)):
                if test(option, self._options[insert_row]):
                    break
            else:
                insert_row = len(self._options)
        self.layout().insertRow(insert_row, option.name, option.widget)
        if option.balloon:
            self.layout().itemAt(insert_row,
                QFormLayout.LabelRole).widget().setToolTip(option.balloon)

def recurse_getattr(obj, attr_name):
    attrs = attr_name.split('.')
    for a in attrs:
        obj = getattr(obj, a)
    return obj

from abc import ABCMeta, abstractmethod

class Option(metaclass=ABCMeta):
    """Base class (and common API) for all options"""

    multiple_value = "-- mulitple --"
    read_only = False

    get_attribute = getattr
    set_attribute = setattr

    def __init__(self, name, default, callback, *, balloon=None, attr_name=None, **kw):
        # non-empty name overrides any default name
        if name or not hasattr(self, 'name'):
            self.name = name

        if attr_name:
            self.attr_name = attr_name
        else:
            if not hasattr(self, 'attr_name'):
                if self.name:
                    self.attr_name = self.name
                else:
                    self.attr_name = ""

        self.multi_attribute = '.' in self.attr_name
        if self.multi_attribute:
            self.get_attribute = recurse_getattr

        self._callback = callback
        if default != None or not hasattr(self, 'default'):
            self.default = default

        self._make_widget(**kw)

        if balloon or not hasattr(self, 'balloon'):
            self.balloon = balloon
        command_line = hasattr(self, 'in_class') and '.' not in self.attr_name
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

    def _make_callback(self):
        # Called by GUI to propagate changes back to program
        if self._callback:
            self._callback(self)

    @abstractmethod
    def _make_widget(self):
        # create (as self.widget) the widget to display the option value
        pass

class ColorOption(Option):
    """Option for colors"""


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
        self._make_callback()

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

    def _make_callback(self):
        label = self.widget.text()
        i = list(self.labels).index(label)
        self._value = self.values[i]
        EnumOption._make_callback(self)

    def _make_widget(self, **kw):
        self._value = self.default
        EnumOption._make_widget(self,
            display_value=self.labels[list(self.values).index(self.default)], **kw)

    def _menu_cb(self, label):
        self.set(self.values[self.labels.index(label)])
        self._make_callback()
