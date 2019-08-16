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

from PyQt5.QtWidgets import QWidget

class ItemTable(QWidget):
    """ Typical usage is to add_column()s, set_data(), and launch() (see doc strings for those).
        If you saved the table's state (via session_info() call), then provide the 'session_data'
        keyword to the launch() call with the saved state as the value.
    """

    PREF_SUBKEY_COL_DISP = "default col display"
    PREF_SUBKEY_WRAP = "col wrap width"

    def __init__(self, *, auto_multiline_headers=True, column_control_info=None, allow_user_sorting=True,
            settings_attr="item_table_info"):
        """ 'auto_multiline_headers' controls whether header titles can be split into multiple
            lines on word boundaries.

            'allow_user_sorting' controls whether mouse clicks on column headers will sort the
            columns.

            'column_control_info', if provided, is used to populate either a menu or widget with
            checkbutton entries or checkbuttons (respectively) to control which columns are displayed.
            For a menu the value should be:
                (QMenu instance, chimerax.core.settings.Settings instance, defaults dictionary,
                  fallback default [, optional display callback])
            For a widget the value is:
                (QWidget instance, chimerax.core.settings.Settings instance, defaults dictionary,
                  fallback default, display callback, wrap length, number of checkbutton columns)
            The Settings instance will be used to remember the displayed column preferences (as
            an attribute given by 'settings_attr').  The defaults dictionary controls whether the
            column is shown by default, which column titles as keys and booleans as values (True =
            displayed).  The fallback default (a boolean) is for columns missing from the defaults
            dictionary.  The display callback, if not None, is called when a column is configured
            in/out of the table.  It is called with a single argument: the ItemColumn instance (whose
            'display' attribute corresponds to the state _after_ the change).  Widget-style control
            areas have two additional options.  If wrap length is not None, then the control area
            will include an entry field for controlling the maximum widths of columns before they
            wrap.  The value for wrap length should be a two-tuple of the floating point width and
            a string indicating the units of the width -- either "inches" or "cm".  The value is only
            used as an initial default; any changes the user makes wil be remembered via settings.
            The number of checkbutton columns option controls, well, the number of checkbutton
            columns.  If None, the number will be determined automatically (approx. square root of
            number of check buttons).
        """
        self._allow_user_sorting = allow_user_sorting
        self._auto_multiline_headers = auto_multiline_headers
        self._column_control_info = column_control_info
        self._settings_attr = settings_attr
        if column_control_info:
            settings = column_control_info[1]
            prefs = getattr(settings, settings_attr, None)
            if prefs is None:
                prefs = { self.PREF_SUBKEY_COL_DISP: column_control_info[2] }
                setattr(settings, settings_attr, prefs)
            if isinstance(column_control_info[0], QWidget):
                from PyQt5.QtWidgets import QVBoxLyout, QGridLayout, QHBoxLayout, QWidget, QLabel
                from PyQt5.QtCore import Qt
                main_layout = QVBoxLayout()
                column_control_info[0].setLayout(main_layout)
                self._col_checkbutton_layout = QGridLayout()
                main_layout.addLayout(self._col_checkbutton_layout)
                if column_control_info[-2] is not None:
                    wrap_val = prefs.get(self.PREF_SUBKEY_WRAP, column_control_info[-2])
                    from chimerax.ui.options import PhysicalSizeOption, OptionsPanel
                    panel = OptionsPanel()
                    panel.add_option(PhysicalSizeOption("Maximum column width", wrap_val, self._col_wrap))
                    main_layout.addWidget(panel, alignment=Qt.AlignLeft)
                from PyQt5.QtWidgets import QDialogButtonBox as qbbox
                buttons_widget = QWidget()
                main_layout.addWidget(buttons_widget, alignment=Qt.AlignLeft)
                buttons_layout = QHBoxLayout()
                buttons_widget.setLayout(buttons_layout)
                buttons_layout.addWidget(QLabel("Show columns"))
                bbox = qbbox()
                buttons_layout.addWidget(bbox)
                bbox.addButton("All", qbbox.ActionRole).clicked.connect(self._show_all_columns)
                bbox.addButton("Default", qbbox.ActionRole).clicked.connect(self._show_default)
                bbox.addButton("Standard", qbbox.ActionRole).clicked.connect(self._show_standard)
                bbox.addButton("Set Default", qbbox.ActionRole).clicked.connect(self._set_default)
        self._highlighted = []

    def destroy(self):
        #TODO
        pass

class _ItemColumn:
    def __init__(self, title, data_fetch, display_format, title_display, alignment, wrap_length,
            font, color, header_alignment):
        
