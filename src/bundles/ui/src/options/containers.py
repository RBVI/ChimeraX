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

"""OptionsPanel and CategorizedOptionsPanel are for interfaces that want to present a set of
options that aren't remembered in any way between different sessions.  The corresponding
Settings classes offer buttons to Save/Reset/Restore the option values to/from a Settings
instance (found in chimerax.core.settings).

The Categorized classes organize the presented options into categories, which the user can
switch between.
"""

from Qt.QtWidgets import QWidget, QFormLayout, QTabWidget, QVBoxLayout, QGridLayout, \
    QPushButton, QCheckBox, QScrollArea, QGroupBox
from Qt.QtCore import Qt

class OptionsPanel(QWidget):
    """Supported API. OptionsPanel is a container for single-use (not savable) Options"""

    def __init__(self, parent=None, *, sorting=True, scrolled=True, contents_margins=None):
        """sorting:
            False; options shown in order added
            True: options sorted alphabetically by name
            func: options sorted based on the provided key function
        """
        QWidget.__init__(self, parent)
        self._layout = QVBoxLayout()
        if contents_margins is not None:
            self._layout.setContentsMargins(*contents_margins)
        if scrolled:
            sublayout = QVBoxLayout()
            sublayout.setContentsMargins(3,0,3,2)
            self.setLayout(sublayout)
            scroller = QScrollArea()
            scroller.setWidgetResizable(True)
            sublayout.addWidget(scroller)
            scrolled_area = QWidget()
            scroller.setWidget(scrolled_area)
            scrolled_area.setLayout(self._layout)
        else:
            self.setLayout(self._layout)
        self._sorting = sorting
        self._options = []
        self._option_groups = []
        self._layout.setSizeConstraint(self._layout.SetMinAndMaxSize)
        self._form = QFormLayout()
        self._form.setSizeConstraint(self._form.SetMinAndMaxSize)
        self._form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._form.setVerticalSpacing(1)
        # None of the below seem to have an effect on the Mac...
        #self._form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        #self._form.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # if we wanted to force the form contents to upper left...
        #self._form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._layout.addLayout(self._form)

    def add_option(self, option):
        """Supported API. Add an option (instance of chimerax.ui.options.Option)."""
        if self._sorting is False:
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
        self._form.insertRow(insert_row, option.name, option.widget)
        self._options.insert(insert_row, option)
        label_item = self._form.itemAt(insert_row, QFormLayout.LabelRole)
        if label_item:
            label_widget = label_item.widget()
            label_widget.setTextInteractionFlags(Qt.TextBrowserInteraction)
            label_widget.setOpenExternalLinks(True)
            if option.balloon:
                label_widget.setToolTip(option.balloon)

    def add_option_group(self, group_label=None, checked=None, group_alignment=None, **kw):
        if group_label is None:
            grouping_widget = QWidget()
        else:
            grouping_widget = QGroupBox(group_label)
            grouping_widget.setContentsMargins(1,grouping_widget.contentsMargins().top()//2,1,1)
            if checked is not None:
                grouping_widget.setCheckable(True)
                grouping_widget.setChecked(checked)
        add_kw = {} if group_alignment is None else { 'alignment': group_alignment }
        self._layout.addWidget(grouping_widget, **add_kw)
        suboptions = OptionsPanel(scrolled=False, **kw)
        self._option_groups.append(suboptions)
        return grouping_widget, suboptions

    def change_label_for_option(self, option, new_label):
        self._form.labelForField(option.widget).setText(new_label)

    def options(self):
        all_options = self._options[:]
        for grp in self._option_groups:
            # an option group can have further subgroups, so call options()
            all_options.extend(grp.options())
        return all_options

    def sizeHint(self):
        from Qt.QtCore import QSize
        form_size = self._form.minimumSize()
        return QSize(min(form_size.width(), 800), min(form_size.height(), 800))

class CategorizedOptionsPanel(QTabWidget):
    """Supported API. CategorizedOptionsPanel is a container for single-use (not savable) Options sorted by category"""

    def __init__(self, parent=None, *, category_sorting=True, option_sorting=True,
            category_scrolled={}, **kw):
        """sorting:
            False: categories/options shown in order added
            True: categories/options sorted alphabetically by name
            func: categories/options sorted based on the provided key function

            If category not found in category_scrolled, defaults to True
        """
        self._contents_margins = kw.pop('contents_margins', None)
        QTabWidget.__init__(self, parent, **kw)
        self._category_sorting = category_sorting
        self._option_sorting = option_sorting
        self._category_to_panel = {}
        self._category_scrolled = category_scrolled

    def add_option(self, category, option):
        """Supported API. Add option (instance of chimerax.ui.options.Option) to given category"""
        try:
            panel = self._category_to_panel[category]
        except KeyError:
            panel = OptionsPanel(sorting=self._option_sorting, contents_margins=self._contents_margins,
                scrolled=self._category_scrolled.get(category, True))
            self.add_tab(category, panel)
        panel.add_option(option)

    def add_option_group(self, category, **kw):
        try:
            panel = self._category_to_panel[category]
        except KeyError:
            panel = OptionsPanel(sorting=self._option_sorting, contents_margins=self._contents_margins)
            self.add_tab(category, panel)
        return panel.add_option_group(**kw)

    def add_tab(self, category, panel):
        """Supported API. Only used if a tab needs to present a custom interface.

           The panel needs to offer a .options() method that returns its options."""
        self._category_to_panel[category] = panel
        if len(self._category_to_panel) == 1 or self._category_sorting == False:
            self.addTab(panel, category)
        else:
            cats = list(self._category_to_panel.keys()) + [category]
            if self._category_sorting is True:
                cats.sort()
            else:
                cats.sort(key=lambda cat: self._category_sorting(cat))
            self.insertTab(cats.index(category), panel, category)

    def categories(self):
        return self._category_to_panel.keys()

    def current_category(self):
        return self.tabText(self.currentIndex())

    def set_current_category(self, category):
        category = category.casefold()
        for index in range(self.count()):
            if category == self.tabText(index).casefold():
                self.setCurrentIndex(index)
                break
        else:
            raise ValueError("category not found")

    def options(self, category):
        return self._category_to_panel[category].options()

class SettingsPanelBase(QWidget):
    def __init__(self, parent, option_sorting, multicategory,
            *, category_sorting=None, buttons=True, help_cb=None, **kw):
        QWidget.__init__(self, parent)
        self.multicategory = multicategory
        if multicategory:
            self.options_panel = CategorizedOptionsPanel(option_sorting=option_sorting,
                    category_sorting=category_sorting, **kw)
        else:
            self.options_panel = OptionsPanel(sorting=option_sorting, **kw)

        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.addWidget(self.options_panel, 1)
        layout.setContentsMargins(0,0,0,0)

        if buttons:
            button_container = QWidget()
            bc_layout = QGridLayout()
            bc_layout.setContentsMargins(0, 0, 0, 0)
            bc_layout.setVerticalSpacing(5)
            if multicategory:
                self.current_check = QCheckBox("Buttons below apply to current section only")
                self.current_check.setToolTip("If checked, buttons only affect current section")
                self.current_check.setChecked(True)
                from .. import shrink_font
                shrink_font(self.current_check)
                from Qt.QtCore import Qt
                bc_layout.addWidget(self.current_check, 0, 0, 1, 4, Qt.AlignRight)
            save_button = QPushButton("Save")
            save_button.clicked.connect(self._save)
            save_button.setToolTip("Save as startup defaults")
            bc_layout.addWidget(save_button, 1, 0)
            reset_button = QPushButton("Reset")
            reset_button.clicked.connect(self._reset)
            reset_button.setToolTip("Reset to initial-installation defaults")
            bc_layout.addWidget(reset_button, 1, 1)
            restore_button = QPushButton("Restore")
            restore_button.clicked.connect(self._restore)
            restore_button.setToolTip("Restore from saved defaults")
            bc_layout.addWidget(restore_button, 1, 2)
            if help_cb is not None:
                self._help_cb = help_cb
                help_button = QPushButton("Help")
                from chimerax.core.commands import run
                help_button.clicked.connect(self._help)
                help_button.setToolTip("Show help")
                bc_layout.addWidget(help_button, 1, 3)

            button_container.setLayout(bc_layout)
            layout.addWidget(button_container, 0)

        self.setLayout(layout)

    def show_category(self, category):
        self.options_panel.set_current_category(category)

    def _get_actionable_options(self):
        if self.multicategory:
            if self.current_check.isChecked():
                options = self.options_panel.options(self.options_panel.current_category())
            else:
                options = []
                for cat in self.options_panel.categories():
                    options.extend(self.options_panel.options(cat))
        else:
            options = self.options_panel.options()
        return options

    def _help(self):
        if self.multicategory and self.current_check.isChecked():
            self._help_cb(category=self.options_panel.current_category())
        else:
            self._help_cb()

    def _reset(self):
        from chimerax.core.configfile import Value
        for opt in self._get_actionable_options():
            default_val = opt.settings.PROPERTY_INFO[opt.attr_name]
            if isinstance(default_val, Value):
                default_val = default_val.default
            # '==' on numpy objects doesn't return a boolean
            import numpy
            if not numpy.array_equal(opt.value, default_val):
                opt.value = default_val
                opt.make_callback()

    def _restore(self):
        for opt in self._get_actionable_options():
            restore_val = opt.settings.saved_value(opt.attr_name)
            # '==' on numpy objects doesn't return a boolean
            import numpy
            if not numpy.array_equal(opt.value, restore_val):
                opt.value = restore_val
                opt.make_callback()

    def _save(self):
        save_info = {}
        for opt in self._get_actionable_options():
            # need to ensure "current value" is up to date before saving...
            setattr(opt.settings, opt.attr_name, opt.value)
            save_info.setdefault(opt.settings, []).append(opt.attr_name)
        # We don't simply use settings.save() when all options are being saved
        # since there may be settings that aren't presented in the GUI
        for settings, save_settings in save_info.items():
            settings.save(settings=save_settings)

class SettingsPanel(SettingsPanelBase):
    """Supported API. SettingsPanel is a container for remember-able Options that work in conjunction with
       Options that have Settings instances (found in chimerax.core.settings) specified via their
       'settings' constructor arg.
    """

    def __init__(self, parent=None, *, sorting=True, **kw):
        """'settings' is a Settings instance.  The remaining arguments are the same as
            for OptionsPanel
        """
        SettingsPanelBase.__init__(self, parent, sorting, multicategory=False, **kw)

    def add_option(self, option):
        """Supported API. Add an option (instance of chimerax.ui.options.Option)."""
        self.options_panel.add_option(option)

    def add_option_group(self, **kw):
        """Returns a container widget and an OptionsPanel; caller is responsible
           for creating a layout for the container widget and placing the
           OptionsPanel in it, along with any other desired widgets"""
        return self.options_panel.add_option_group(**kw)

class CategorizedSettingsPanel(SettingsPanelBase):
    """Supported API. CategorizedSettingsPanel is a container for remember-able Options
       (i.e. Options that have Settings instances (found in chimerax.core.settings) specified via their
       'settings' constructor arg) and that are presented in categories.
    """

    def __init__(self, parent=None, *, category_sorting=True, option_sorting=True, **kw):
        """'settings' is a Settings instance.  The remaining arguments are the same as
            for CategorizedOptionsPanel
        """
        SettingsPanelBase.__init__(self, parent, option_sorting, multicategory=True,
            category_sorting=category_sorting, **kw)

    def add_option(self, category, option):
        """Supported API. Add option (instance of chimerax.ui.options.Option) to given category"""
        self.options_panel.add_option(category, option)

    def add_option_group(self, category, **kw):
        """Returns a container widget and an OptionsPanel; caller is responsible
           for creating a layout for the container widget and placing the
           OptionsPanel in it, along with any other desired widgets"""
        try:
            panel = self.options_panel._category_to_panel[category]
        except KeyError:
            panel = OptionsPanel(sorting=self.options_panel._option_sorting,
                contents_margins=self.options_panel._contents_margins)
            self.options_panel.add_tab(category, panel)
        return panel.add_option_group(**kw)

    def add_tab(self, category, panel):
        """Supported API. Same as CategorizedOptionsPanel.add_tab(...)"""
        self.options_panel.add_tab(category, panel)
