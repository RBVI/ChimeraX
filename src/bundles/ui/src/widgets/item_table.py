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

from Qt.QtWidgets import QWidget, QAction, QCheckBox, QTableView, QMenu, QAbstractItemView
from Qt.QtCore import QAbstractTableModel, Qt, QModelIndex, Signal, QSortFilterProxyModel
# Qt has no QVariant; None can be used in place of an invalid QVariant
# from Qt.QtCore import QVariant
from Qt.QtGui import QFontDatabase, QBrush, QColor

class QCxTableModel(QAbstractTableModel):
    def __init__(self, item_table, **kw):
        self._item_table = item_table
        super().__init__(**kw)

    def columnCount(self, parent=None):
        return len(self._item_table._columns)

    def data(self, index, role=None):
        col = self._item_table._columns[index.column()]
        item = self._item_table._data[index.row()]
        if role is None or role == Qt.DisplayRole:
            val = col.display_value(item)
            from chimerax.core.colors import Color
            if isinstance(val, bool):
                cell = self._item_table.item(index.row(), index.column())
                if not cell.isCheckable():
                    cell.toggled.connect(lambda chk, c=col, i=item: c.set_value(i, chk))
                cell.setCheckState(Qt.Checked if val else Qt.Unchecked)
            elif isinstance(val, Color) or isinstance(val, tuple) and 3 <= len(val) <= 4:
                widget = self._item_table.indexWidget(index)
                if not widget:
                    has_alpha = len(val.rgba) == 4 if isinstance(val, Color) else len(val) == 4
                    from .color_button import ColorButton
                    widget = ColorButton(has_alpha=has_alpha)
                    widget.color_changed.connect(lambda clr, c=col, i=item: c.set_value(i, clr))
                    self._item_table.setIndexWidget(index, widget)
                widget.color = val
            return str(val)
        if role == Qt.FontRole and (item in self._item_table._highlighted or col.justification == "decimal"
                or col.font is not None):
            if col.justification == "decimal":
                font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            elif col.font is None:
                if self._item_table._table_model is None:
                    return None
                font = self._item_table.font()
            else:
                font = col.font
            if item in self._item_table._highlighted:
                font = QFont(font)
                font.setBold(True)
            return font
        if role == Qt.TextAlignmentRole:
            return self._convert_justification(col.justification)
        return None

    def headerData(self, section, orientation, role=None):
        if orientation == Qt.Vertical:
            return None

        col = self._item_table._columns[section]
        if role is None or role == Qt.DisplayRole:
            if self._item_table._auto_multiline_headers:
                title = self._make_multiline(col.title)
            else:
                title = col.title
            return title

        elif role == Qt.TextAlignmentRole:
            return self._convert_justification(col.header_justification)

        elif role == Qt.ForegroundRole:
            if col.color is not None:
                from chimerax.core.colors import Color
                if isinstance(col.color, Color):
                    color = col.color
                else:
                    color = Color(col.color)
                return QBrush(QColor(*color.uint8x4()))

        elif role == Qt.ToolTipRole and col.balloon:
            return col.balloon

        return None

    def rowCount(self, parent=None):
        return len(self._item_table._data)

    def _convert_justification(self, justification):
        if justification == "left":
            return Qt.AlignLeft | Qt.AlignVCenter
        if justification == "center":
            return Qt.AlignHCenter | Qt.AlignVCenter
        return Qt.AlignRight | Qt.AlignVCenter

    def _make_multiline(self, title):
        words = title.strip().split()
        if len(words) < 2:
            return title
        longest = max([len(w) for w in words])
        while True:
            best_diff = best_index = None
            for i in range(len(words)-1):
                w1, w2 = words[i:i+2]
                cur_diff = max(abs(longest - len(w1)), abs(longest - len(w2)))
                diff = abs(longest - len(w1) - len(w2) - 1)
                if diff >= cur_diff:
                    continue
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_index = i
            if best_diff is None:
                break
            words[best_index:best_index+2] = [" ".join(words[best_index:best_index+2])]
        #TODO: might need to be '<br>'.join(words)
        return '\n'.join(words)

class NumSortingProxyModel(QSortFilterProxyModel):
    def lessThan(self, left_index, right_index):
        left_data = self.sourceModel().data(left_index)
        right_data = self.sourceModel().data(right_index)
        try:
            left_num = float(left_data)
            right_num = float(right_data)
        except ValueError:
            return left_data.casefold() < right_data.casefold()
        return left_num < right_num

class ItemTable(QTableView):
    """ Typical usage is to add_column()s, set the 'data' attribute, and then launch() (see doc
        strings for those).  If you saved the table's state (via session_info() call), then provide
        the 'session_data' keyword to the launch() call with the saved state as the value.

        Do not do anything Qt related with the ItemTable until launch() has been called, because
        that's when the underlying QTableView gets initialized.

        ItemTable provides a selection_changed signal that delivers a list of the selected data
        items to the connected function.
    """

    selection_changed = Signal(list, list)

    DEFAULT_SETTINGS_ATTR = "item_table_info"

    COL_FORMAT_BOOLEAN = "boolean"
    COL_FORMAT_TRANSPARENT_COLOR = "alpha"
    COL_FORMAT_OPAQUE_COLOR = "no alpha"

    def __init__(self, *, auto_multiline_headers=True, column_control_info=None, allow_user_sorting=True,
            settings_attr=None):
        """ 'auto_multiline_headers' controls whether header titles can be split into multiple
            lines on word boundaries.

            'allow_user_sorting' controls whether mouse clicks on column headers will sort the
            columns.

            'column_control_info', if provided, is used to populate either a menu or widget with
            check box entries or check boxes (respectively) to control which columns are displayed.
            For a menu the value should be:
                (QMenu instance, chimerax.core.settings.Settings instance, defaults dictionary,
                  fallback default [, optional display callback])
            For a widget the value is:
                (QWidget instance, chimerax.core.settings.Settings instance, defaults dictionary,
                  fallback default, display callback, number of check box columns, show global buttons)
            The Settings instance will be used to remember the displayed column preferences (as
            an attribute given by 'settings_attr', which defaults to DEFAULT_SETTINGS_ATTR and which
            should be declared as 'EXPLICIT_SAVE').  The defaults dictionary controls whether the
            column is shown by default, which column titles as keys and booleans as values (True =
            displayed).  The fallback default (a boolean) is for columns missing from the defaults
            dictionary.  The display callback, if not None, is called when a column is configured
            in/out of the table.  It is called with a single argument: the ItemColumn instance (whose
            'display' attribute corresponds to the state _after_ the change).  Widget-style control
            areas have an additional field, which is the number of check box columns.  If None,
            the number will be determined automatically (approx. square root of number of check buttons,
            'show global buttons' determines whether the "Show All", etc.. buttons are added.  Typically
            should be set to True for tables with a fixed set of columns and False for variable sets.
        """
        self._table_model = None
        self._columns = []
        self._data = []
        self._allow_user_sorting = allow_user_sorting
        self._auto_multiline_headers = auto_multiline_headers
        self._column_control_info = column_control_info
        self._settings_attr = self.DEFAULT_SETTINGS_ATTR if settings_attr is None else settings_attr
        self._pending_columns = []
        if column_control_info:
            self._checkables = {}
            from Qt.QtWidgets import QVBoxLayout, QGridLayout, QHBoxLayout, QWidget, QLabel
            # QMenu is also a QWidget, so can't test isinstance(QWidget)...
            if not isinstance(column_control_info[0], QMenu):
                from Qt.QtCore import Qt
                main_layout = QVBoxLayout()
                column_control_info[0].setLayout(main_layout)
                self._col_checkbox_layout = QGridLayout()
                self._col_checkbox_layout.setContentsMargins(0,0,0,0)
                self._col_checkbox_layout.setSpacing(0)
                main_layout.addLayout(self._col_checkbox_layout)
                self._col_checkboxes = []
                if column_control_info[-1]:
                    from Qt.QtWidgets import QDialogButtonBox as qbbox
                    buttons_widget = QWidget()
                    main_layout.addWidget(buttons_widget, alignment=Qt.AlignLeft)
                    buttons_layout = QHBoxLayout()
                    buttons_layout.setContentsMargins(0,0,0,0)
                    buttons_widget.setLayout(buttons_layout)
                    buttons_layout.addWidget(QLabel("Show columns"))
                    bbox = qbbox()
                    buttons_layout.addWidget(bbox)
                    bbox.addButton("All", qbbox.ActionRole).clicked.connect(self._show_all_columns)
                    bbox.addButton("Default", qbbox.ActionRole).clicked.connect(self._show_default)
                    bbox.addButton("Standard", qbbox.ActionRole).clicked.connect(self._show_standard)
                    bbox.addButton("Set Default", qbbox.ActionRole).clicked.connect(self._set_default)
        self._highlighted = set()
        super().__init__()

    def add_column(self, title, data_fetch, *, format="%s", display=None, title_display=True,
            justification="center", balloon=None, font=None, refresh=True, color=None,
            header_justification=None):
        """ Add a column who's header text is 'title'.  It is allowable to add a column with the
            same title multiple times.  The duplicative additions will be ignored.

            'data_fetch' is how to fetch the value to display from a data item.  If it's a string,
            then it is assumed to be an attribute (or subattribute, e.g. "color.rgba") of the data
            item (see set_data()).  Otherwise it is a function which, when applied to a data item,
            returns the value that should be displayed in the corresponding cell (but see 'format').

            'format' describes how to show that data item's value.  If 'format' is COL_FORMAT_BOOLEAN,
            use a check box.  If it is COL_FORMAT_TRANSPARENT_COLOR or COL_FORMAT_OPAQUE_COLOR then
            use a color button with the appropriate transparency option.  Otherwise it should
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
            with right justification.  This means you should provide a 'format' with a fixed number
            of decimal places, possibly with trailing spaces to mitigate the right justification.

            'balloon' is explanatory text that will be shown in a tooltip if the user hover the mouse
            over the column header.

            'font' is the font employed to display column data items, and should be a QFont instance.
            If not provided, it defaults to a proportional font unless 'jusification' is "decimal",
            in which case a fixed-width font will be used.

            'refresh' is whether to refresh the table contents once the column is added.  It defaults
            to True, but if you are adding multiple columns to an existing large table, you might
            want to set it to False except for the last added column.

            'color' is the _foreground_ (text) color of the column header. Should be a chimerax.core.Color
            instance or a value that can be used as the Color constructor first argument.

            'header_justification' is the text justification of the header text.  Same values as
            'justification' except no "decimal". Default to the same justification as 'justification'
            (but "right" if 'justification' is "decimal").
        """
        titles = [c.title for c in self._columns]
        if title in titles:
            return self._columns[titles.index(title)]

        if display is None:
            if self._column_control_info:
                widget, settings, defaults, fallback = self._column_control_info[:4]
                display = getattr(settings, self._settings_attr).get(title, fallback)
            else:
                display = True
        if header_justification is None:
            header_justification = justification if justification != "decimal" else "right"

        c = _ItemColumn(title, data_fetch, format, title_display, justification, font, color,
            header_justification, balloon)

        if self._column_control_info:
            self._add_column_control_entry(c)
        if display != c.display:
            self.update_column(c, display=display)
        if not self._table_model:
            # not yet launch()ed
            self._columns.append(c)
            return c

        self._pending_columns.append(c)
        if refresh:
            num_existing = len(self._columns)
            self._table_model.beginInsertColumns(QModelIndex(),
                num_existing, num_existing + len(self._pending_columns)-1)
            self._columns.extend(self._pending_columns)
            self._table_model.endInsertColumns()
            self._pending_columns = []
            self.resizeColumnsToContents()
        return c

    @property
    def column_names(self):
        return [c.title for c in self._columns]

    @property
    def data(self):
        return self._data[:]

    @data.setter
    def data(self, data):
        """ Should just be a sequence of instances (rows) from which the table data can be retrieved by
            applying the 'data_fetch' function from each column.
        """
        if not self._table_model:
            self._data = data[:]
            return
        old_data_set = set(self._data)
        new_data_set = set(data)
        if old_data_set.isdisjoint(new_data_set):
            emit_signal = self.selected()
            self._table_model.beginResetModel()
            self._data = data[:]
            self._table_model.endResetModel()
            if emit_signal:
                self.selection_changed.emit([])
            return
        while True:
            for i, datum in enumerate(self._data):
                if datum not in new_data_set:
                    self._table_model.beginRemoveRows(QModelIndex(), i, i+1)
                    self._data = self._data[:i] + self._data[i+1]
                    self._table_model.endRemoveRows()
                    break
            else:
                break
        done = False
        while not done:
            for i, datum in enumerate(data):
                if i >= len(self._data):
                    self._table_model.beginInsertRows(QModelIndex(), i, len(data))
                    self._data.extend(data[i:])
                    self._table_model.endInsertRows()
                    done = True
                    break
                if self._data[i] != datum:
                    self._table_model.beginInsertRows(QModelIndex(), i, i+1)
                    self._data = self._data[:i] + [datum] + self._data[i:]
                    self._table_model.endInsertRows()
                    break
            else:
                done = True

    def destroy(self):
        self._data = []
        super().destroy()

    def highlight(self, highlight_data):
        new = set(highlight_data)
        if new == self._highlighted:
            return
        self._highlighted = new
        top_left = self._table_model.index(0, len(self._columns)-1)
        bottom_right = self._table_model.index(len(self._data)-1, len(self.columns)-1)
        self._table_model.dataChanged(top_left, bottom_right, [Qt.FontRole]).emit()

    def launch(self, *, select_mode=QAbstractItemView.ExtendedSelection, session_info=None):
        self._table_model = QCxTableModel(self)
        if self._allow_user_sorting:
            sort_model = NumSortingProxyModel()
            sort_model.setSourceModel(self._table_model)
            self.setModel(sort_model)
            self.setSortingEnabled(True)
        else:
            self.setModel(self._table_model)
        self.setSelectionBehavior(self.SelectRows)
        self.setSelectionMode(select_mode)
        if self._column_control_info and not isinstance(self._column_control_info[0], QMenu):
            self._arrange_col_checkboxes()
        if session_info:
            version, selected, column_display, highlighted, sort_info = session_info
            if self._allow_user_sorting and sort_info is not None:
                col_num, order = sort_info
                self.sortByColumn(col_num, order)
            sel_model = self.selectionModel()
            for i in selected:
                index = self._table_model.index(i,0)
                sel_model.select(index, sel_model.Rows | sel_model.SelectCurrent)
            self.highlight([self._data[i] for i in highlighted])
            for c in self._columns:
                self.update_column(c, display=column_display.get(c.title, True))
        self.selectionModel().selectionChanged.connect(self._relay_selection_change)
        self.resizeColumnsToContents()

    def scroll_to(self, datum):
        """ Scroll the table to ensure that the given data item is visible """
        self.scrollTo(self._table_model.index(self._data.index(datum), 0), self.PositionAtCenter)

    @property
    def selected(self):
        if self._allow_user_sorting:
            return [self._data[self.model().mapToSource(i).row()]
                for i in self.selectionModel().selectedRows()]
        return [self._data[i.row()] for i in self.selectionModel().selectedRows()]

    def session_info(self):
        version = 1
        selected = set([i.row() for i in self.selectedIndexes()])
        column_display = { c.title: c.display for c in self._columns }
        highlighted = [i for i, d in enumerate(self.data) if d in self._highlighted]
        if self._allow_user_sorting:
            sort_info = (self.model().sortColumn(), int(self.model().sortOrder()))
        else:
            sort_info = None
        return (version, selected, column_display, highlighted, sort_info)

    def update_column(self, column, **kw):
        display_change = 'display' in kw and column.display != kw['display']
        changes = column._update(**kw)
        if not self._table_model:
            return
        if display_change:
            if column.display:
                self.showColumn(self._columns.index(column))
            else:
                self.hideColumn(self._columns.index(column))
            if self._column_control_info:
                self._checkables[column.title].setChecked(kw['display'])
        if not changes:
            return
        top_left = self._table_model.index(0, self._columns.index(column))
        bottom_right = self._table_model.index(len(self._data)-1, self._columns.index(column))
        self._table_model.dataChanged.emit(top_left, bottom_right, changes)

    def _add_column_control_entry(self, col):
        action = QAction(col.title)
        if col.balloon:
            action.setToolTip(col.balloon)
        action.setCheckable(True)
        action.setChecked(col.display)
        qt_cb =lambda checked, c=col: self.update_column(c, display=checked)

        widget = self._column_control_info[0]
        if isinstance(widget, QMenu):
            action.triggered.connect(qt_cb)
            widget.addAction(action)
            self._checkables[col.title] = action
        else:
            check_box = QCheckBox(col.title)
            check_box.addAction(action)
            check_box.setChecked(col.display)
            check_box.stateChanged.connect(qt_cb)
            self._col_checkboxes.append(check_box)
            self._checkables[col.title] = check_box
            if self._table_model:
                # we've been launch()ed
                self._arrange_col_checkboxes()

    def _arrange_col_checkboxes(self):
        while self._col_checkbox_layout.count() > 0:
            self._col_checkbox_layout.takeAt(0)
        self._col_checkboxes.sort(key=lambda cb: cb.text())
        requested_cols = self._column_control_info[-2]
        num_buttons = len(self._col_checkboxes)
        from math import sqrt, ceil
        if requested_cols is None:
            num_cols = int(sqrt(num_buttons)+0.5)
        else:
            num_cols = requested_cols
        num_rows = int(ceil(num_buttons/num_cols))
        row = col = 0
        for checkbox in self._col_checkboxes:
            self._col_checkbox_layout.addWidget(checkbox, row, col, alignment=Qt.AlignLeft)
            row += 1
            if row >= num_rows:
                row = 0
                col += 1

    def _relay_selection_change(self, selected, deselected):
        if self._allow_user_sorting:
            sel_data, desel_data = [[self._data[self.model().mapToSource(i).row()] for i in x.indexes()]
                for x in (selected, deselected)]
        else:
            sel_data, desel_data = [[self._data[i.row()] for i in x.indexes()]
                for x in (selected, deselected)]
        self.selection_changed.emit(sel_data, desel_data)

    def _set_default(self):
        shown = {}
        for col in self._columns:
            shown[col.title] = col.display
        settings = self._column_control_info[1]
        setattr(settings, self._settings_attr, shown)
        settings.save(setting=self._settings_attr)

    def _show_all_columns(self):
        for col in self._columns:
            self.update_column(col, display=True)

    def _show_default(self):
        widget, settings, display_defaults, fallback = self._column_control_info[:4]
        for col in self._columns:
            display = display_defaults.get(col.title, fallback)
            self.update_column(col, display=display)

    def _show_standard(self):
        widget, settings, display_defaults, fallback = self._column_control_info[:4]
        for col in self._columns:
            display = display_defaults.get(col.title, fallback)
            self.update_column(col, display=display)

class _ItemColumn:
    def __init__(self, title, data_fetch, display_format, title_display, justification, font, color,
            header_justification, balloon):
        # set all args to corresponding 'self' attributes...
        import inspect
        args, varargs, keywords, locals = inspect.getargvalues(inspect.currentframe())
        for name in args:
            if name == 'self':
                continue
            setattr(self, name, locals[name])
        self.display = True
        #TODO: font is None and justification is "decimal" -- fixed-width font

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

    def _update(self, data=False, data_fetch=None, format=None, display=None, justification=None, font=None):
        changed = []
        if data:
            changed.append(Qt.DisplayRole)
        if data_fetch is not None and data_fetch != self.data_fetch:
            self.data_fetch = data_fetch
            changed.append(Qt.DisplayRole)
        if format is not None and format != self.display_format:
            self.display_format = format
            changed.append(Qt.DisplayRole)
        if display is not None and display != self.display:
            self.display = display
        if justification is not None and justification != self.justification:
            self.justification = justification
            changed.append(Qt.TextAlignmentRole)
        if font is not None and font != self.font:
            self.font = font
            changed.append(Qt.FontRole)
        return changed
