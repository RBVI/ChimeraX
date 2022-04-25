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

from chimerax.ui.options import Option

class AtomPairRestrictOption(Option):
    restrict_kw_vals = ("any", "cross", "both")
    atom_spec_menu_text = "between selection and atom spec..."

    def __init__(self, *args, atom_word="atom", **kw):
        from chimerax.core.commands import plural_of
        self.fixed_kw_menu_texts = (
            "with at least one %s selected" % atom_word,
            "with exactly one %s selected" % atom_word,
            "with both %s selected" % plural_of(atom_word)
        )
        if args[-1] != None:
            raise AssertionError(self.__class__.__name + " does not support callbacks")
        super().__init__(*args, **kw)

    def get_value(self):
        text = self.__push_button.text()
        for val, val_text in zip(self.restrict_kw_vals, self.fixed_kw_menu_texts):
            if text == val_text:
                return val
        atom_spec = self.__line_edit.text().strip()
        if not atom_spec:
            from chimerax.core.errors import UserError
            raise UserError("Restriction atom specifier must not be blank")
        return atom_spec

    def set_value(self, value):
        if value in self.restrict_kw_vals:
            self.__push_button.setText(self.fixed_kw_menu_texts[self.restrict_kw_vals.index(value)])
            self.__line_edit.hide()
        else:
            self.__push_button.setText(self.atom_spec_menu_text)
            self.__line_edit.setText(value)
            self.__line_edit.show()

    value = property(get_value, set_value)

    def set_multiple(self):
        self.__push_button.setText(self.multiple_value)

    def _make_widget(self, *, display_value=None, **kw):
        if display_value is None:
            display_value = self.fixed_kw_menu_texts[0]
        from Qt.QtWidgets import QHBoxLayout, QPushButton, QMenu, QLineEdit
        from Qt.QtGui import QAction
        from Qt.QtCore import Qt
        self.widget = layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        self.__push_button = QPushButton(display_value, **kw)
        self.__push_button.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        menu = QMenu(self.__push_button)
        self.__push_button.setMenu(menu)
        for label in self.fixed_kw_menu_texts + (self.atom_spec_menu_text,):
            action = QAction(label, self.__push_button)
            action.triggered.connect(lambda *, s=self, lab=label: self._menu_cb(lab))
            menu.addAction(action)
        layout.addWidget(self.__push_button, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        self.__line_edit = QLineEdit()
        self.__line_edit.setMinimumWidth(72)
        if display_value in self.fixed_kw_menu_texts:
            self.__line_edit.hide()
        layout.addWidget(self.__line_edit, alignment=Qt.AlignCenter)

    def _menu_cb(self, label):
        if label in self.fixed_kw_menu_texts:
            self.value = self.restrict_kw_vals[self.fixed_kw_menu_texts.index(label)]
        else:
            self.value = self.__line_edit.text()
        # No callback, because user may need to fill in atom spec field first

class StructureOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    value = property(get_value, set_value)

    def set_multiple(self):
        self.widget.setText(self.multiple_value)

    def _make_widget(self, **kw):
        from .widgets import StructureMenuButton
        self.widget = StructureMenuButton(self.session, **kw)
        self.widget.value_changed.connect(self.make_callback)

class StructuresOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    value = property(get_value, set_value)

    def set_multiple(self):
        pass

    def _make_widget(self, **kw):
        from .widgets import StructureListWidget
        self.widget = StructureListWidget(self.session, **kw)
        self.widget.value_changed.connect(self.make_callback)

class AtomicStructureOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    value = property(get_value, set_value)

    def set_multiple(self):
        self.widget.setText(self.multiple_value)

    def _make_widget(self, **kw):
        from .widgets import AtomicStructureMenuButton
        self.widget = AtomicStructureMenuButton(self.session, **kw)
        self.widget.value_changed.connect(self.make_callback)

class AtomicStructuresOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    value = property(get_value, set_value)

    def set_multiple(self):
        pass

    def _make_widget(self, **kw):
        from .widgets import AtomicStructureListWidget
        self.widget = AtomicStructureListWidget(self.session, **kw)
        self.widget.value_changed.connect(self.make_callback)

class ChainOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    value = property(get_value, set_value)

    def set_multiple(self):
        self.widget.setText(self.multiple_value)

    def _make_widget(self, **kw):
        from .widgets import ChainMenuButton
        self.widget = ChainMenuButton(self.session, **kw)
        self.widget.value_changed.connect(self.make_callback)

class ChainsOption(Option):
    def __init__(self, session, *args, **kw):
        self.session = session
        super().__init__(*args, **kw)

    def get_value(self):
        return self.widget.value

    def set_value(self, value):
        self.widget.value = value

    value = property(get_value, set_value)

    def set_multiple(self):
        pass

    def _make_widget(self, **kw):
        from .widgets import ChainListWidget
        self.widget = ChainListWidget(self.session, **kw)
        self.widget.value_changed.connect(self.make_callback)
