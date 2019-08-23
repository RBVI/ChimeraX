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

    COL_FORMAT_BOOLEAN = "boolean"
    COL_FORMAT_TRANSPARENT_COLOR = "alpha"
    COL_FORMAT_OPAQUE_COLOR = "no alpha"

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
                  fallback default, display callback, number of checkbutton columns)
            The Settings instance will be used to remember the displayed column preferences (as
            an attribute given by 'settings_attr').  The defaults dictionary controls whether the
            column is shown by default, which column titles as keys and booleans as values (True =
            displayed).  The fallback default (a boolean) is for columns missing from the defaults
            dictionary.  The display callback, if not None, is called when a column is configured
            in/out of the table.  It is called with a single argument: the ItemColumn instance (whose
            'display' attribute corresponds to the state _after_ the change).  Widget-style control
            areas have an additional field, which is the number of checkbutton columns.  If None,
            the number will be determined automatically (approx. square root of number of check buttons).
        """
        self._columns = []
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

    def add_column(self, title, data_fetch, *, format="%s", display=None, title_display=True,
            justification="center", balloon=None, font=None, refresh=True, color=None,
            header_justifcation=None):
        """ Add a column who's header text is 'title'.  It is allowable to add a column with the
            same title multiple times.  The duplicative additions will be ignored.

            'data_fetch' is how to fetch the value to display from a data item.  If it's a string,
            then it is assumed to be an attribute (or subattribute, e.g. "color.rgba") of the data
            item (see set_data()).  Otherwise it is a function which, when applied to a data item,
            returns the value that should be displayed in the corresponding cell (but see 'format').

            'format' describes how to show that data item's value.  If 'format' is COL_FORMAT_BOOLEAN,
            use a checkbutton.  If it is COL_FORMAT_TRANSPARENT_COLOR or COL_FORMAT_OPAQUE_COLOR then
            use a color button with the appropraite transparency option.  Otherwise it should
            ether be a callable or a text string.  If it's a callable, the callable will be invoked
            with the value returned by 'data_fetch' and should return a textual representation of the
            value.  If it's a text string, it should be a format string such that (format %
            data_fetch(data_item)) returns the text to display in the cell.

            'display' is whether the column should be initially displayed or not.  If None, then
            any user preferences will be used to choose the value, and if there are none then the
            column will default to being displayed.

            'title_display' controls whether the title of the column will be displayed as a header.

            'justification' controls how data items are aligned in the cells.  It can be "left",
            "right", "center", or "decimal".  If "decimal" then a fixed-width font will be employed
            (unless overridden with 'font') and right justification.  This means you should provide
            a 'format' with a fixed number of decimal places, possibly with trailing spaces to
            mitigate the right justification.

            'balloon' is explantory text that will be shown in a tooltip if the user hover the mouse
            over the column header.

            'font' is the font employed to display column data items, and should be a QFont instance.
            If not provided, it defaults to a proportional font unless 'jusification' is "decimal",
            in which case a fixed-width font will be used.

            'refresh' is whether to refresh the table contents once the column is added.  It defaults
            to True, but if you are adding multiple columns to an existing large table, you might
            want to set it to False except for the last added column.

            'color' is the _foreground_ (text) color of the column header.

            'header_justification' is the text justification of the header text.  Same values as
            'justification' except no "decimal". Default to the same justification as 'justification'
            (but "right" if 'justification' is "decimal").
        """
        if title in [c.title for c in self.columns]:
            return

        if display is None:
            if self._column_control_info:
                widget, settings, defaults, fallback = self._column_control_info[:4]
                lookup = getattr(settings, self._settings_attr)[self.PREF_SUBKEY_COL_DISP]
                display = lookup.get(title, fallback)
            else:
                display = True
        if header_justification is None:
            header_justification = justification if justification != "decimal" else "right"

        c = ItemColumn(title, data_fetch, format, title_display, justification, font, color,
            header_justification, balloon)
        #TODO


    def destroy(self):
        #TODO
        pass


class _ItemColumn:
    def __init__(self, title, data_fetch, display_format, title_display, alignment, font, color,
            header_alignment, balloon):
        # set all args to corresponding 'self' attributes...
        import inspect
        args, varargs, keywords, locals = inspect.getargvalues(inspect.currentframe())
        for name in args:
            if name == 'self':
                continue
            setattr(self, name, locals[name])
        self.display = True

    def display_value(self, instance):
        val = self.value(instance)
        if self.display_format in (ItemTable.COL_FORMAT_BOOLEAN, ItemTable.COL_FORMAT_OPAQUE_COLOR,
                ItemTable.COL_FORMAT_TRANSPARENT_COLOR):
            return val
        if callable(self.display_format):
            return self.display_format(val)
        if val is None:
            return ""
        return self.display_format % val

    def value(self, instance):
        if callable(self.data_fetch):
            return self.data_fetch(instance)
        fetched = instance
        try:
            for fetch in self.data_fetch.split('.'):
                fetched = getattr(fetched, fetch)
        except AttributeError:
            return None
        return fetched

    def set_value(self, instance, val):
        if callable(self.data_fetch):
            raise ValueError("Don't know how to set values for column %s" % self.title)
        fields = self.data_fetch.split('.')
        for fetch in fields[:-1]:
            instance = getattr(instance, fetch)
        setattr(instance, fields[-1], val)
