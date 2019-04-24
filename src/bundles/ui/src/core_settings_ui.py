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

"""
core_settings_ui: GUI to control core settings
==============================================

TODO
"""

from chimerax.core.core_settings import set_proxies, settings as core_settings
from .options import SymbolicEnumOption, ColorOption, BooleanOption, IntOption, FloatOption
from .options import StringOption, HostPortOption, Option, EnumOption, StringsOption
from .widgets import hex_color_name

class AtomSpecOption(SymbolicEnumOption):
    values = ("command", "serial", "simple")
    labels = ("command line", "serial number", "simple")

class ToolSideOption(EnumOption):
    values = ("left", "right")

class UpdateIntervalOption(SymbolicEnumOption):
    values = ("day", "week", "month")
    labels = ("every day", "every week", "every month")

class InitWindowSizeOption(Option):

    def __init__(self, *args, session=None, **kw):
        self.session = session
        Option.__init__(self, *args, **kw)

    def get_value(self):
        size_scheme = self.push_button.text()
        if size_scheme == "last used":
            data = None
        elif size_scheme == "proportional":
            data = (self.w_proportional_spin_box.value()/100,
                self.h_proportional_spin_box.value()/100)
        else:
            data = (self.w_fixed_spin_box.value(), self.h_fixed_spin_box.value())
        return (size_scheme, data)

    def set_value(self, value):
        size_scheme, size_data = value
        self.push_button.setText(size_scheme)
        if size_scheme == "proportional":
            w, h = size_data
            data = (self.w_proportional_spin_box.setValue(w*100),
                self.h_proportional_spin_box.setValue(h*100))
        elif size_scheme == "fixed":
            w, h = size_data
            self.w_fixed_spin_box.setValue(w)
            self.h_fixed_spin_box.setValue(h)
        self._show_appropriate_widgets()

    value = property(get_value, set_value)

    def set_multiple(self):
        self.push_button.setText(self.multiple_value)

    def _make_widget(self, **kw):
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        self.widget.setLayout(layout)

        from PyQt5.QtWidgets import QPushButton, QMenu
        size_scheme, size_data = self.default
        self.push_button = QPushButton(size_scheme)
        menu = QMenu()
        self.push_button.setMenu(menu)
        from PyQt5.QtWidgets import QAction
        menu = self.push_button.menu()
        for label in ("last used", "proportional", "fixed"):
            action = QAction(label, self.push_button)
            action.triggered.connect(lambda arg, s=self, lab=label: s._menu_cb(lab))
            menu.addAction(action)
        from PyQt5.QtCore import Qt
        layout.addWidget(self.push_button, 0, Qt.AlignLeft)

        self.fixed_widgets = []
        self.proportional_widgets = []
        w_pr_val, h_pr_val = 67, 67
        w_px_val, h_px_val = 1200, 750
        if size_scheme == "proportional":
            w_pr_val, h_pr_val = size_data
        elif size_scheme == "fixed":
            w_px_val, h_px_val = size_data
        from PyQt5.QtWidgets import QSpinBox, QWidget, QLabel
        self.nonmenu_widgets = QWidget()
        layout.addWidget(self.nonmenu_widgets)
        nonmenu_layout = QVBoxLayout()
        nonmenu_layout.setContentsMargins(0,0,0,0)
        nonmenu_layout.setSpacing(2)
        self.nonmenu_widgets.setLayout(nonmenu_layout)
        w_widgets = QWidget()
        nonmenu_layout.addWidget(w_widgets)
        w_layout = QHBoxLayout()
        w_widgets.setLayout(w_layout)
        w_layout.setContentsMargins(0,0,0,0)
        w_layout.setSpacing(2)
        self.w_proportional_spin_box = QSpinBox()
        self.w_proportional_spin_box.setMinimum(1)
        self.w_proportional_spin_box.setMaximum(100)
        self.w_proportional_spin_box.setValue(w_pr_val)
        self.w_proportional_spin_box.valueChanged.connect(lambda val, s=self: s.make_callback())
        w_layout.addWidget(self.w_proportional_spin_box)
        self.proportional_widgets.append(self.w_proportional_spin_box)
        self.w_fixed_spin_box = QSpinBox()
        self.w_fixed_spin_box.setMinimum(1)
        self.w_fixed_spin_box.setMaximum(1000000)
        self.w_fixed_spin_box.setValue(w_px_val)
        self.w_fixed_spin_box.valueChanged.connect(lambda val, s=self: s.make_callback())
        w_layout.addWidget(self.w_fixed_spin_box)
        self.fixed_widgets.append(self.w_fixed_spin_box)
        w_proportional_label = QLabel("% of screen width")
        w_layout.addWidget(w_proportional_label)
        self.proportional_widgets.append(w_proportional_label)
        w_fixed_label = QLabel("pixels wide")
        w_layout.addWidget(w_fixed_label)
        self.fixed_widgets.append(w_fixed_label)
        h_widgets = QWidget()
        nonmenu_layout.addWidget(h_widgets)
        h_layout = QHBoxLayout()
        h_widgets.setLayout(h_layout)
        h_layout.setContentsMargins(0,0,0,0)
        h_layout.setSpacing(2)
        self.h_proportional_spin_box = QSpinBox()
        self.h_proportional_spin_box.setMinimum(1)
        self.h_proportional_spin_box.setMaximum(100)
        self.h_proportional_spin_box.setValue(h_pr_val)
        self.h_proportional_spin_box.valueChanged.connect(lambda val, s=self: s.make_callback())
        h_layout.addWidget(self.h_proportional_spin_box)
        self.proportional_widgets.append(self.h_proportional_spin_box)
        self.h_fixed_spin_box = QSpinBox()
        self.h_fixed_spin_box.setMinimum(1)
        self.h_fixed_spin_box.setMaximum(1000000)
        self.h_fixed_spin_box.setValue(h_px_val)
        self.h_fixed_spin_box.valueChanged.connect(lambda val, s=self: s.make_callback())
        h_layout.addWidget(self.h_fixed_spin_box)
        self.fixed_widgets.append(self.h_fixed_spin_box)
        h_proportional_label = QLabel("% of screen height")
        h_layout.addWidget(h_proportional_label)
        self.proportional_widgets.append(h_proportional_label)
        h_fixed_label = QLabel("pixels high")
        h_layout.addWidget(h_fixed_label)
        self.fixed_widgets.append(h_fixed_label)

        self.current_fixed_size_label = QLabel()
        self.current_proportional_size_label = QLabel()
        nonmenu_layout.addWidget(self.current_fixed_size_label)
        nonmenu_layout.addWidget(self.current_proportional_size_label)
        self._update_current_size()

        self._show_appropriate_widgets()

    def _menu_cb(self, label):
        self.push_button.setText(label)
        self._show_appropriate_widgets()
        self.make_callback()

    def _show_appropriate_widgets(self):
        for w in self.proportional_widgets + self.fixed_widgets:
            w.hide()
        self.current_fixed_size_label.hide()
        self.current_proportional_size_label.hide()
        self.nonmenu_widgets.hide()
        size_scheme = self.push_button.text()
        if size_scheme == "proportional":
            self.nonmenu_widgets.show()
            for w in self.proportional_widgets:
                w.show()
            self.current_proportional_size_label.show()
        elif size_scheme == "fixed":
            self.nonmenu_widgets.show()
            for w in self.fixed_widgets:
                w.show()
            self.current_fixed_size_label.show()

    def _update_current_size(self, trig_name=None, wh=None):
        mw = getattr(self.session.ui, "main_window", None)
        if not mw:
            self.session.ui.triggers.add_handler('ready', self._update_current_size)
            return

        if wh is None:
            # this should only happen once...
            mw.triggers.add_handler('resized', self._update_current_size)
            window_width, window_height = mw.width(), mw.height()
        else:
            window_width, window_height = wh

        from PyQt5.QtWidgets import QDesktopWidget
        dw = QDesktopWidget()
        screen_geom = self.session.ui.primaryScreen().availableGeometry()
        screen_width, screen_height = screen_geom.width(), screen_geom.height()
        self.current_fixed_size_label.setText(
            "Current: %d wide, %d high" % (window_width, window_height))
        self.current_proportional_size_label.setText("Current: %d%% wide, %d%% high" % (
                int(100.0 * window_width / screen_width),
                int(100.0 * window_height / screen_height)))

def _enable_trackpad_multitouch(session, enable):
    session.ui.mouse_modes.trackpad.enable_multitouch(enable)

def _set_trackpad_sensitivity(session, value):
    session.ui.mouse_modes.trackpad.trackpad_speed = value

class CoreSettingsPanel:

    # settings_info is keyed on setting name, and value is a tuple composed of:
    #
    # 1) Description to display in the gui
    # 2) Category (also for gui)
    # 3) Option class to use, or (Option-class, {additional __init__ keywords}) tuple
    # 4) Updater to use when option changed.  One of:
    #     a) None, if no update necessary
    #     b) string, a command to run (and see next tuple component)
    #     c) a function to call (also see next tuple component)
    # 5) Converter to use with updater.  Either None (don't provide arg to updater), or
    #     a function to convert the option's value to a form usable with the updater.
    #     If the updater is a command, then the converted value will be as the right
    #     side of the '%' string-formatting operator; otherwise it will be the second arg
    #     provided to the function call (session will be the first).
    # 6) Balloon help for option.  Can be None.
    # 7) Whether to automatically set the core setting.  If True, the setting will be changed
    #     before any updater is called.  Otherwise, the updater is in charge of setting the
    #     setting.  Usually only set to False if the updater needs to examine the old value.
    settings_info = {
        'atomspec_contents': (
            "Atomspec display style",
            "Labels",
            AtomSpecOption,
            None,
            None,
            """How to format display of atomic data<br>
            <table>
            <tr><td>simple</td><td>&nbsp;</td><td>Simple readable form</td></tr>
            <tr><td>command line</td><td>&nbsp;</td><td>Form used in commands</td></tr>
            <tr><td>serial number</td><td>&nbsp;</td><td>Atom serial number</td></tr>
            </table>""",
            True),
        'background_color': (
            "Background color",
            "Background",
            ColorOption,
            "set bgColor %s",
            hex_color_name,
            "Background color of main graphics window",
            True),
        'clipping_surface_caps': (
            'Surface caps',
            'Clipping',
            BooleanOption,
            'surface cap %s',
            None,
            'Whether to cap surface holes created by clipping',
            False),
        'default_tool_window_side': (
            "Default tool side",
            "Window",
            ToolSideOption,
            None,
            None,
            "Which side of main window that new tool windows appear on by default.",
            True),
        'errors_raise_dialog': (
            'Errors shown in dialog',
            'Log',
            BooleanOption,
            None,
            None,
            'Should error messages be shown in a separate dialog as well as being logged',
            True),
        'http_proxy': (
            'HTTP proxy',
            'Web Access',
            HostPortOption,
            lambda ses, val: set_proxies(),
            None,
            'HTTP proxy for ChimeraX to use when trying to reach web sites',
            True),
        'https_proxy': (
            'HTTPS proxy',
            'Web Access',
            HostPortOption,
            lambda ses, val: set_proxies(),
            None,
            'HTTPS proxy for ChimeraX to use when trying to reach web sites',
            True),
        'initial_window_size': (
            "Initial overall window size",
            "Window",
            (InitWindowSizeOption, {'session': None}),
            None,
            None,
            """Initial overall size of ChimeraX window""",
            True),
        'resize_window_on_session_restore': (
            'Resize graphics window on session restore',
            'Window',
            BooleanOption,
            None,
            None,
            'Whether to resize main window when restoring a session to the size it had when the session was saved.',
            True),
        'startup_commands': (
            "Execute these commands at startup",
            "Startup",
            StringsOption,
            None,
            None,
            "List of commands to execute when ChimeraX command-line tool starts",
            True),
        'toolshed_update_interval': (
            "Toolshed update interval",
            "Toolshed",
            UpdateIntervalOption,
            None,
            None,
            'How frequently to check toolshed for new updates<br>',
            True),
        'trackpad_multitouch': (
            'Trackpad gestures to rotate and move',
            'Trackpad',
            BooleanOption,
            _enable_trackpad_multitouch,
            None,
            'Whether to enable 2 and 3 finger trackpad drags to rotate and move.',
            True),
        'trackpad_sensitivity': (
            'Trackpad sensitivity',
            'Trackpad',
            (FloatOption, {'decimal_places': 2 }),
            _set_trackpad_sensitivity,
            None,
            'How fast models move in response to multitouch trackpad gestures',
            True),
        'warnings_raise_dialog': (
            'Warnings shown in dialog',
            'Log',
            BooleanOption,
            None,
            None,
            'Should warning messages be shown in a separate dialog as well as being logged',
            True),
    }

    def __init__(self, session, ui_area):
        from PyQt5.QtWidgets import QBoxLayout
        self.session = session
        from chimerax.core.commands import run
        from .options import CategorizedSettingsPanel
        self.options_widget = CategorizedSettingsPanel(help_cb=lambda *, category=None, ses=session, run=run:
            run(ses, "help help:user/preferences.html"
            + ("" if category is None else "#" + category.replace(' ', '').lower())))
        self.options = {}

        for setting, setting_info in self.settings_info.items():
            opt_name, category, opt_class, updater, converter, balloon, set_setting = setting_info
            if isinstance(opt_class, tuple):
                opt_class, kw = opt_class
                if 'session' in kw:
                    kw['session'] = self.session
            else:
                kw = {}
            opt = opt_class(opt_name, getattr(core_settings, setting), self._opt_cb,
                attr_name=setting, settings=core_settings, balloon=balloon, auto_set_attr=set_setting, **kw)
            self.options_widget.add_option(category, opt)
            self.options[setting] = opt

        core_settings.triggers.add_handler('setting changed', self._core_setting_changed)
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        layout.setSpacing(5)
        layout.addWidget(self.options_widget, 1)
        layout.setContentsMargins(0, 0, 0, 0)

        ui_area.setLayout(layout)

    def _core_setting_changed(self, trig_name, info):
        setting_name, old_val, new_val = info
        if setting_name in self.options:
            self.options[setting_name].value = new_val

    def _opt_cb(self, opt):

        setting = opt.attr_name
        opt_name, category, opt_class, updater, converter, balloon, set_setting = self.settings_info[setting]
        if updater is None:
            return

        if isinstance(updater, str):
            # command to run
            val = opt.value
            if converter:
                val = converter(val)
            from chimerax.core.commands import run
            run(self.session, updater % val)
        else:
            updater(self.session, opt.value)
