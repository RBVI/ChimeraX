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
from Qt.QtGui import QDoubleValidator, QIntValidator
from Qt.QtCore import Qt
from chimerax.core.commands import run
from chimerax.ui import tool_user_error
from .cmd import builtin_presets

style_attrs = list(builtin_presets["simple"].keys())

class AnisoTool(ToolInstance):

    #help = "help:user/tools/thermalellipsoids.html"

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
        preset_menu.triggered.connect(self._preset_menu_cb)
        pmb.setMenu(preset_menu)
        self._populate_preset_menu()
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
        self.scale = sf = QLineEdit(str(defaults["scale"]))
        sf.setAlignment(Qt.AlignCenter)
        sf.setMaximumWidth(7 * sf.fontMetrics().averageCharWidth())
        sf.setValidator(QDoubleValidator(0.0001, 1000000, -1))
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
        self.set_scale = ss = QLineEdit("")
        ss.setAlignment(Qt.AlignCenter)
        ss.setMaximumWidth(5 * ss.fontMetrics().averageCharWidth())
        ss.setValidator(QDoubleValidator(0.0001, 100, -1))
        ss.editingFinished.connect(self._set_scale_cb)
        set_scale_layout.addWidget(ss)
        set_scale_layout.addStretch(1)

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

    def _populate_preset_menu(self):
        menu = self.preset_menu_button.menu()
        menu.clear()
        for entry in sorted(list(builtin_presets.keys()) + list(self.settings.custom_presets.keys()),
                key=lambda x: x.casefold()):
            menu.addAction(entry)
        #TODO: entries for saving a preset and deleting a custom preset

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
            from chimerax.core.commands import camel_case
            from .mgr import manager_for_structure
            mgr = manager_for_structure(self.session, s)
            diffs = []
            for attr_name in style_attrs:
                try:
                    widget = getattr(self, attr_name)
                except AttributeError:
                    #TODO: remove try/accept once all attribute widgets implemented
                    continue
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
                    if val != mgr.drawing_params[attr_name]:
                        diffs.extend([arg_name, str_val])
            if diffs:
                run(self.session, "aniso style " + " ".join(diffs)),

        if self.sel_restrict_check_box.isChecked():
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
            try:
                widget = getattr(self, attr_name)
            except AttributeError:
                #TODO: remove try/accept once all attribute widgets implemented
                continue
            if isinstance(widget, QLineEdit):
                widget.setText(str(mgr.drawing_params[attr_name]))

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

