# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
from chimerax.core.settings import Settings
from Qt.QtWidgets import QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QButtonGroup, QRadioButton, QWidget
from Qt.QtWidgets import QPushButton, QScrollArea, QMenu, QCheckBox, QLineEdit, QSpacerItem, QSizePolicy
from Qt.QtWidgets import QGroupBox, QInputDialog
from Qt.QtGui import QDoubleValidator, QIntValidator
from Qt.QtCore import Qt
from chimerax.core.commands import run
from chimerax.ui import tool_user_error
from chimerax.ui.widgets import ColorButton
from .cmd import builtin_presets

style_attrs = list(builtin_presets["simple"].keys())

class AnisoTool(ToolInstance):

    help = "help:user/tools/thermalellipsoids.html"

    NO_PRESET_TEXT = "no preset"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from .settings import get_settings
        self.settings = get_settings(session)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        parent.setLayout(main_layout)


        preset_model_layout = QHBoxLayout()
        main_layout.addLayout(preset_model_layout)
        preset_model_layout.addStretch(1)
        preset_model_layout.addWidget(QLabel("Preset:"), alignment=Qt.AlignRight)

        self.preset_menu_button = pmb = QPushButton()
        preset_menu = QMenu(pmb)
        preset_menu.aboutToShow.connect(self._populate_preset_menu)
        pmb.setMenu(preset_menu)
        preset_model_layout.addWidget(pmb, alignment=Qt.AlignLeft)
        preset_model_layout.addStretch(1)

        from chimerax.atomic.widgets import AtomicStructureMenuButton as ASMB
        self.structure_button = sb = ASMB(session, no_value_button_text="No relevant structures",
            filter_func=lambda s: s.atoms.has_aniso_u.any())
        self.structure_button.value_changed.connect(self._update_widgets)
        preset_model_layout.addWidget(sb)
        preset_model_layout.addStretch(1)

        from .cmd import builtin_presets
        defaults = builtin_presets["simple"]

        scale_smoothing_layout = QHBoxLayout()
        main_layout.addLayout(scale_smoothing_layout)
        scale_smoothing_layout.addStretch(1)
        scale_smoothing_layout.addWidget(QLabel("Scale factor:"))
        self.scale = sf = DoubleEntry(str(defaults["scale"]))
        scale_smoothing_layout.addWidget(sf)
        spacer = QSpacerItem(10, 0, QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        scale_smoothing_layout.addSpacerItem(spacer)
        scale_smoothing_layout.addWidget(QLabel("Smoothing level:"))
        self.smoothing = sl = QLineEdit(str(defaults["smoothing"]))
        sl.setAlignment(Qt.AlignCenter)
        sl.setMaximumWidth(3 * sl.fontMetrics().averageCharWidth())
        validator = QIntValidator()
        validator.setBottom(1)
        sl.setValidator(validator)
        scale_smoothing_layout.addWidget(sl)
        scale_smoothing_layout.addStretch(1)

        set_scale_layout = QHBoxLayout()
        main_layout.addLayout(set_scale_layout)
        set_scale_layout.addStretch(1)
        set_scale_layout.addWidget(QLabel("Set scale factor for probability (%):"))
        self.set_scale = ss = DoubleEntry("")
        ss.editingFinished.connect(self._set_scale_cb)
        set_scale_layout.addWidget(ss)
        set_scale_layout.addStretch(1)

        self.show_ellipsoid = QGroupBox("Depict ellipsoids")
        main_layout.addWidget(self.show_ellipsoid)
        self.show_ellipsoid.setCheckable(True)
        eb_layout = QGridLayout()
        eb_layout.setSpacing(0)
        eb_layout.setContentsMargins(0,0,0,0)
        self.show_ellipsoid.setLayout(eb_layout)
        eb_layout.addWidget(QLabel("Color:"), 0, 0, alignment=Qt.AlignRight)
        self.color = ColorWidget("use atom color", max_size=(16,16), has_alpha_channel=True)
        eb_layout.addWidget(self.color, 0, 1, alignment=Qt.AlignLeft)
        eb_layout.addWidget(QLabel("Transparency:"), 1, 0, alignment=Qt.AlignRight)
        self.transparency = etb = QPushButton()
        t_menu = QMenu(etb)
        t_menu.triggered.connect(lambda act, but=etb: but.setText(act.text()))
        t_menu.addAction("same as color")
        for pct in range(0, 101, 10):
            t_menu.addAction(f"{pct}%")
        etb.setMenu(t_menu)
        eb_layout.addWidget(etb, 1, 1, alignment=Qt.AlignLeft)

        from .cmd import builtin_presets
        defaults = builtin_presets["simple"]
        for label, factor_prefix, attr_prefix in [
                ("principal axes", "Length", "axis"),
                ("principal ellipses", "Size", "ellipse")
        ]:
            gbox = QGroupBox("Depict " + label)
            main_layout.addWidget(gbox)
            gbox.setCheckable(True)
            box_layout = QGridLayout()
            box_layout.setSpacing(0)
            box_layout.setContentsMargins(0,0,0,0)
            gbox.setLayout(box_layout)

            box_layout.addWidget(QLabel("Color:"), 0, 0, alignment=Qt.AlignRight)
            cw = ColorWidget("use atom color", max_size=(16,16), has_alpha_channel=True)
            box_layout.addWidget(cw, 0, 1, alignment=Qt.AlignLeft)
            setattr(self, attr_prefix + '_color', cw)

            box_layout.addWidget(QLabel(factor_prefix + " factor:"), 1, 0, alignment=Qt.AlignRight)
            factor = DoubleEntry("1.0")
            box_layout.addWidget(factor, 1, 1, alignment=Qt.AlignLeft)
            factor.setAlignment(Qt.AlignCenter)
            setattr(self, attr_prefix + "_factor", (gbox, factor))

            box_layout.addWidget(QLabel("Thickness:"), 2, 0, alignment=Qt.AlignRight)
            attr_name = attr_prefix + "_thickness"
            thickness = DoubleEntry(str(defaults[attr_name]))
            box_layout.addWidget(thickness, 2, 1, alignment=Qt.AlignLeft)
            setattr(self, attr_name, thickness)

        self._update_widgets()

        hide_show_layout = QHBoxLayout()
        main_layout.addLayout(hide_show_layout)
        hide_show_layout.addStretch(1)
        show_button = QPushButton("Show")
        show_button.clicked.connect(lambda *args, f=self._show_hide_cb: f())
        hide_show_layout.addWidget(show_button)
        hide_show_layout.addWidget(QLabel("/"))
        hide_button = QPushButton("Hide")
        hide_button.clicked.connect(lambda *args, f=self._show_hide_cb: f(hide=True))
        hide_show_layout.addWidget(hide_button)
        hide_show_layout.addWidget(QLabel("depictions"))
        hide_show_layout.addStretch(1)
        sel_restrict_layout = QHBoxLayout()
        main_layout.addLayout(sel_restrict_layout)
        sel_restrict_layout.addStretch(1)
        self.sel_restrict_check_box = QCheckBox("Restrict Show/Hide to current selection, if any")
        sel_restrict_layout.addWidget(self.sel_restrict_check_box)
        sel_restrict_layout.addStretch(1)

        from .mgr import triggers
        self.handlers = [triggers.add_handler("style changed", self._style_changed_cb)]

        tw.manage(placement=None)

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()
        self.structure_button.destroy()
        super().delete()

    def _delete_preset(self):
        preset_name, okayed = QInputDialog.getItem(self.preset_menu_button,
            "Delete User Preset", "Preset:", sorted(list(self.settings.custom_presets.keys())), 0, False)
        if not okayed:
            return
        run(self.session, "aniso preset delete " + preset_name)
        if self.preset_menu_button.text() == preset_name:
            self.preset_menu_button.setText(self.NO_PRESET_TEXT)

    def _gather_diffs(self, s):
        from chimerax.core.commands import camel_case
        from chimerax.core.colors import color_name
        from .mgr import manager_for_structure
        mgr = manager_for_structure(self.session, s)
        diffs = []
        for attr_name in style_attrs:
            widget = getattr(self, attr_name)
            arg_name = camel_case(attr_name)
            if isinstance(widget, QLineEdit):
                str_val = widget.text()
                if not widget.hasAcceptableInput():
                    return tool_user_error("Unacceptable value (%s) for '%s' argument"
                        % (str_val, arg_name))
                if isinstance(widget.validator(), QDoubleValidator):
                    val = float(str_val)
                else:
                    val = int(str_val)
            elif isinstance(widget, QGroupBox):
                # "the blah_factor attributes, that are composed of a QGroupBox and QLineEdit,
                # are represented as a tuple of widgets
                val = widget.isChecked()
                str_val = str(val).lower()
            elif isinstance(widget, ColorWidget):
                val = widget.value
                if val is None:
                    str_val = "none"
                else:
                    str_val = color_name(val)
            elif isinstance(widget, QPushButton):
                text = widget.text()
                if text == "same as color":
                    val = None
                    str_val = "none"
                else:
                    str_val = text[:-1]
                    val = int(str_val)
            elif isinstance(widget, tuple):
                gbox, factor = widget
                if gbox.isChecked():
                    str_val = factor.text()
                    if not factor.hasAcceptableInput():
                        return tool_user_error("Unacceptable value (%s) for '%s' argument"
                            % (str_val, arg_name))
                    if isinstance(factor.validator(), QDoubleValidator):
                        val = float(str_val)
                    else:
                        val = int(str_val)
                else:
                    str_val = "none"
                    val = None
            else:
                raise AssertionError("Unhandled type of input widget")
            # Since numpy has non-Pythonic equality operators, can't use simple equality test
            from numpy import array_equal
            if not array_equal(val, mgr.drawing_params[attr_name]):
                diffs.extend([arg_name, str_val])
        return diffs

    def _populate_preset_menu(self):
        menu = self.preset_menu_button.menu()
        menu.clear()
        for entry in sorted(list(builtin_presets.keys()) + list(self.settings.custom_presets.keys()),
                key=lambda x: x.casefold()):
            act = menu.addAction(entry)
            act.triggered.connect(lambda *args, act=act: self._preset_menu_cb(act))
        menu.addSeparator()
        act = menu.addAction("Preset from current settings...")
        act.triggered.connect(lambda *args: self._preset_from_current())
        s = self.structure_button.value
        # disable if changes have not been applied to structure
        act.setEnabled(s and not self._gather_diffs(s))
        act = menu.addAction("Delete user preset...")
        act.triggered.connect(lambda *args: self._delete_preset())
        act.setEnabled(bool(self.settings.custom_presets))

    def _preset_from_current(self):
        s = self.structure_button.value
        if not s:
            return tool_user_error("No structure chosen")
        preset_name, okayed = QInputDialog.getText(self.preset_menu_button,
            "Save Preset", "Name:", QLineEdit.Normal, "")
        if not okayed:
            return
        preset_name = preset_name.strip()
        if not preset_name:
            return tool_user_error("Preset name must not be blank")
        if preset_name in builtin_presets.keys():
            return tool_user_error("Cannot use built-in preset name")
        run(self.session, "aniso preset save " + s.atomspec + " " + preset_name)
        self.preset_menu_button.setText(preset_name)

    def _preset_menu_cb(self, action):
        s = self.structure_button.value
        if not s:
            return tool_user_error("No structure chosen")
        self._show_hide_cb(apply_widgets=False)
        run(self.session, "aniso preset " + s.atomspec + " " + action.text())

    def _set_preset_button_text(self, mgr):
        s = self.structure_button.value
        if not s:
            self.preset_menu_button.setText(self.NO_PRESET_TEXT)
            return None
        if mgr is None:
            from .mgr import manager_for_structure
            mgr = manager_for_structure(self.session, s)
        for preset_dict in [builtin_presets, self.settings.custom_presets]:
            for name, settings in preset_dict.items():
                if settings == mgr.drawing_params:
                    self.preset_menu_button.setText(name)
                    return mgr
        self.preset_menu_button.setText(self.NO_PRESET_TEXT)
        return mgr

    def _set_scale_cb(self):
        scale_text = self.set_scale.text()
        if not scale_text:
            return
        if not self.set_scale.hasAcceptableInput():
            return tool_user_error("Probability must be a percentage")
        prob = float(scale_text) / 100.0
        self.scale.setText("%g" % _prob_to_scale(prob))

    def _show_hide_cb(self, *, apply_widgets=True, hide=False):
        s = self.structure_button.value
        if not s:
            return tool_user_error("No structure chosen")

        spec = s.atomspec

        if apply_widgets and not hide:
            diffs = self._gather_diffs(s)
            if diffs:
                run(self.session, "aniso style " + " ".join(diffs)),

        if self.sel_restrict_check_box.isChecked() and s.atoms.selecteds.any():
            spec += " & sel"

        run(self.session, "aniso" + (" hide" if hide else '') + ' ' + spec)

    def _style_changed_cb(self, trig_name, mgr):
        s = self.structure_button.value
        if not s or mgr.structure != s:
            return
        self._update_widgets(mgr=mgr)

    def _update_widgets(self, *, mgr=None):
        mgr = self._set_preset_button_text(mgr)
        if mgr is None:
            return

        for attr_name in style_attrs:
            widget = getattr(self, attr_name)
            param_value = mgr.drawing_params[attr_name]
            if isinstance(widget, QLineEdit):
                widget.setText(str(param_value))
            elif isinstance(widget, QGroupBox):
                # "the blah_factor attributes, that are composed of a QGroupBox and QLineEdit,
                # are represented as a tuple of widgets
                widget.setChecked(param_value)
            elif isinstance(widget, ColorWidget):
                widget.value = param_value
            elif isinstance(widget, QPushButton):
                if param_value is None:
                    text = "same as color"
                else:
                    text = f"{round(param_value)}%"
                widget.setText(text)
            elif isinstance(widget, tuple):
                gbox, factor = widget
                if param_value is None:
                    gbox.setChecked(False)
                else:
                    gbox.setChecked(True)
                    factor.setText(str(param_value))
            else:
                raise AssertionError("Unhandled type of input widget")

class ColorWidget(QWidget):
    def __init__(self, non_color_text, *, choice_changed_cb=None, color_changed_cb=None, **kw):
        super().__init__()
        layout = QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.non_color_but = QRadioButton('')
        layout.addWidget(self.non_color_but, 0, 0, alignment=Qt.AlignRight)
        if choice_changed_cb:
            self.non_color_but.toggled.connect(choice_changed_cb)
        layout.addWidget(QLabel(non_color_text), 0, 1, alignment=Qt.AlignLeft)
        self.color_choice_but = QRadioButton('')
        layout.addWidget(self.color_choice_but, 1, 0, alignment=Qt.AlignRight)
        self.color_button = ColorButton(**kw)
        self.color_button.color = "light gray"
        if color_changed_cb:
            self.color_button.color_changed.connect(lambda clr, *args, cb=color_changed_cb:
                self._editor_change(clr, cb))
        layout.addWidget(self.color_button, 1, 1, alignment=Qt.AlignLeft)

    @property
    def value(self):
        if self.non_color_but.isChecked():
            return None
        return self.color_button.color

    @value.setter
    def value(self, val):
        if val is None:
            self.non_color_but.setChecked(True)
        else:
            self.color_choice_but.setChecked(True)
            self.color_button.color = val

    def _editor_change(self, color, cb):
        if not self.non_color_but.isChecked():
            cb(color)

class DoubleEntry(QLineEdit):
    def __init__(self, init_text, **kw):
        super().__init__(init_text)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximumWidth(7 * self.fontMetrics().averageCharWidth())
        self.setValidator(QDoubleValidator(0.0001, 1000000, -1))

def simpson(f, a, b, n=2):
    """Approximate the definite integral of f from a to b
       by Composite Simpson's rule, dividing the interval in n parts.

       Cribbed from numerical integration wikipedia page.
    """
    assert n > 0
    # forces n to be even
    n = n + (n % 2)
    dx = (b - a) / n
    ans = f(a) + f(b)
    x = a + dx
    m = 4
    for i in range(1, n):
        ans += m * f(x)
        m = 6 - m
        x = x + dx

    return dx * ans / 3

from math import pi, sqrt, exp
def integrand(val):
    val2 = val * val
    return val2 * exp(-val2/2.0)

def integral(val, constant=sqrt(2.0/pi)):
    return constant * simpson(integrand, 0, val, n=70)

def _prob_to_scale(target_prob, low_bound=None, high_bound=None, converge=0.0001):
    if low_bound is None and high_bound is None:
        val = 1.0
    elif low_bound is None:
        val = high_bound / 2.0
    elif high_bound is None:
        val = low_bound * 2.0
    else:
        val = (low_bound + high_bound) / 2.0
    prob = integral(val)
    if prob < target_prob:
        low_bound = val
    else:
        high_bound = val
    if low_bound and high_bound and high_bound - low_bound < converge:
        return (high_bound + low_bound) / 2.0
    return _prob_to_scale(target_prob, low_bound, high_bound, converge)

