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

from Qt.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QPushButton, QMenu, \
    QSizePolicy, QWidget, QStackedWidget, QGridLayout, QLineEdit
from Qt.QtGui import QIntValidator
from Qt.QtCore import Qt

from chimerax.core.commands import plural_of
from chimerax.core.errors import UserError

class SaveOptionsWidget(QFrame):

    def __init__(self, session):
        super().__init__()
        self.session = session

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(5)

        models_layout = QVBoxLayout()
        layout.addLayout(models_layout, stretch=1)
        models_layout.setSpacing(0)
        models_label = QLabel("Save models")
        from chimerax.ui import shrink_font
        shrink_font(models_label)
        models_layout.addWidget(models_label, alignment=Qt.AlignLeft)
        from chimerax.atomic.widgets import StructureListWidget
        self.structure_list = StructureListWidget(session)
        self.structure_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        models_layout.addWidget(self.structure_list)

        self.setLayout(layout)

    def options_string(self):
        models = self.structure_list.value
        if not models:
            raise UserError("No models chosen for saving")
        from chimerax.atomic import Structure
        from chimerax.core.commands import concise_model_spec
        spec = concise_model_spec(self.session, models, relevant_types=Structure)
        if spec:
            cmd = "models " + spec
        else:
            cmd = ""
        return cmd

def fill_context_menu(menu, parent_tool_window, structure):
    from .manager import get_plotting_manager
    mgr = get_plotting_manager(structure.session)

    from Qt.QtGui import QAction
    plot_action = menu.addAction("Plot")
    plot_action.triggered.connect(lambda *args, tw=parent_tool_window, s=structure: _show_plot_dialog(tw, s))

class PlotDialog:
    exclude_interface_text = {
        "solution": "solvent and non-metal ions",
        "metals": "metal ions"
    }

    def __init__(self, plot_window, structure):
        self.tool_window = tw = plot_window
        tw.help = "help:user/commands/coordset.html#slider"
        def cleanup(pd=self):
            inst = pd.tool_window.tool_instance
            del _md_tool_windows[inst]["plot"]
            if not _md_tool_windows[inst]:
                del _md_tool_windows[inst]
            for handler in pd.handlers:
                handler.remove()
            pd.handlers.clear()
            for provider, cids in pd._mouse_handlers.items():
                canvas = pd._plot_stacks[provider].widget(1)
                for cid in cids:
                    canvas.mpl_disconnect(cid)
            pd._mouse_handlers.clear()
            delattr(pd.tool_window, 'cleanup')
        tw.cleanup = cleanup
        self.session = structure.session
        self.structure = structure
        self.handlers = [structure.triggers.add_handler('changes', self._changes_cb)]
        from .manager import get_plotting_manager
        self.mgr = get_plotting_manager(self.session)
        from Qt.QtWidgets import QHBoxLayout, QTabWidget
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)
        self.plot_tabs = QTabWidget()
        layout.addWidget(self.plot_tabs, stretch=1)

        self.tab_info = {}
        self._tables = {}
        self._plot_stacks = {}
        self._value_columns = {}
        self._frame_indicators = {}
        self._mouse_handlers = {}
        self._provider_widgets = {}

        for provider_name in self.mgr.provider_names:
            self.tab_info[provider_name] = self.make_tab(provider_name)

        tw.fill_context_menu = self.fill_context_menu
        tw.manage(None)

    @property
    def cur_provider(self):
        tab_widget = self.plot_tabs.currentWidget()
        for provider_name, info in self.tab_info.items():
            tab_name, page = info
            if tab_widget == page:
                break
        else:
            raise AssertionError("Current tab not found in tab data")
        return provider_name

    def fill_context_menu(self, menu, x, y):
        table = self._tables[self.cur_provider]
        enabled = bool(table.data)

        from Qt.QtGui import QAction
        act = QAction("Save Plot Image...", parent=menu)
        act.triggered.connect(self.save_plot)
        act.setEnabled(enabled)
        menu.addAction(act)

        act = QAction("Save CSV or TSV File...", parent=menu)
        act.triggered.connect(self.save_values)
        act.setEnabled(enabled)
        menu.addAction(act)

    def make_tab(self, provider_name):
        if self.mgr.num_atoms(provider_name) is None:
            return self._make_scalar_tab(provider_name)
        return self._make_atomic_tab(provider_name)

    def save_plot(self, *args):
        provider_name = self.cur_provider
        plot = self._plot_stacks[provider_name].widget(1)

        spd = SavePlotDialog(self.session, plot)
        if not spd.exec():
            return
        path = spd.path
        if path is None:
            return
        plot.figure.savefig(path, transparent=spd.transparent_background, dpi=spd.dpi)
        self.session.logger.info("%s plot saved to %s" % (self.mgr.ui_name(provider_name), path))

    def save_values(self, *args):
        table = self._tables[self.cur_provider]
        table.write_values(header_vals=[cn for cn in table.column_names[3:]] +
            ["Frame %d" % cs_id for cs_id in sorted(self.structure.coordset_ids)],
            row_func=lambda datum, *, table=table: self._table_row_output(table, datum))

    def show_tab(self, provider_name):
        tab_name, tab_widget = self.tab_info[provider_name]
        self.plot_tabs.setCurrentWidget(tab_widget)

    def _changes_cb(self, trig_name, data):
        s, changes = data
        if 'active_coordset changed' in changes.structure_reasons():
            for provider, table in self._tables.items():
                if not table.data:
                    continue
                table.update_column(self._value_columns[provider], data=True)
                self._frame_indicators[provider].set_xdata([s.active_coordset_id])
                self._plot_stacks[provider].widget(1).draw_idle()

    def _delete_table_entries(self, provider_name):
        table = self._tables[provider_name]
        entries = table.data
        ui_name = self.mgr.ui_name(provider_name)
        if not entries:
            raise UserError("No %s to delete" % plural_of(ui_name))
        if len(entries) == 1:
            death_row = entries
        else:
            death_row = table.selected
            if not death_row:
                raise UserError("No %s chosen to delete" % plural_of(ui_name))
        table.data = [x for x in entries if x not in death_row]
        self._update_plot(provider_name)

    def _make_scalar_tab(self, provider_name):
        #TODO
        raise NotImplementedError("Scalar plotting not implemented")

    def _make_atomic_tab(self, provider_name):
        ui_name = self.mgr.ui_name(provider_name)
        tab_name = plural_of(ui_name)
        if tab_name.lower() == tab_name:
            # no caps
            tab_name = tab_name.capitalize()
        from Qt.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
        page = QWidget()
        page_layout = QHBoxLayout()
        page_layout.setSpacing(0)
        page_layout.setContentsMargins(0,0,0,0)
        page.setLayout(page_layout)
        self._plot_stacks[provider_name] = stack = QStackedWidget()
        num_atoms = self.mgr.num_atoms(provider_name)
        atom_string = "any number of" if num_atoms == 0 else "%d" % num_atoms
        stack.addWidget(QLabel("Select " + atom_string + " atoms and click 'Plot' to begin plotting",
            alignment=Qt.AlignCenter))
        from matplotlib.backends.backend_qtagg import FigureCanvas
        from matplotlib.figure import Figure
        stack.addWidget(FigureCanvas(Figure()))
        page_layout.addWidget(stack, stretch=1)
        controls_area = QWidget()
        controls_layout = QVBoxLayout()
        controls_area.setLayout(controls_layout)
        self._tables[provider_name] = table = self._make_table(provider_name)
        controls_layout.addWidget(table, stretch=1)
        controls_layout.addWidget(self._make_buttons_area(provider_name), alignment=Qt.AlignCenter)
        page_layout.addWidget(controls_area)
        self.plot_tabs.addTab(page, tab_name)
        return tab_name, page

    def _make_buttons_area(self, provider_name):
        ui_name = self.mgr.ui_name(provider_name)

        buttons_area = QWidget()
        area_layout = QVBoxLayout()
        area_layout.setSpacing(0)
        area_layout.setContentsMargins(0,0,0,0)
        buttons_area.setLayout(area_layout)

        plot_control_area = QWidget()
        area_layout.addWidget(plot_control_area)
        pc_layout = QGridLayout()
        pc_layout.setSpacing(0)
        pc_layout.setContentsMargins(0,0,0,0)
        plot_control_area.setLayout(pc_layout)
        from itertools import count
        row = count()
        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(lambda *args, f=self._plot_atomic, pv=provider_name: f(pv))
        pc_layout.addWidget(plot_button, next(row), 0, alignment=Qt.AlignRight)
        num_atoms = self.mgr.num_atoms(provider_name)
        preposition = "for" if num_atoms == 0 else "from"
        atom_string = "" if num_atoms == 0 else "%d " % num_atoms
        pc_layout.addWidget(QLabel(" %s %s %sselected atoms" % (ui_name, preposition, atom_string)), 0, 1,
            alignment=Qt.AlignLeft)
        if num_atoms == 0:
            reminder = QLabel("(If no selection, all atoms)")
            from chimerax.ui import shrink_font
            shrink_font(reminder)
            pc_layout.addWidget(reminder, next(row), 0, 1, 2, alignment=Qt.AlignCenter)

        provider_widget_dict = self._provider_widgets.setdefault(provider_name, {})
        if self.mgr.need_ref_frame(provider_name):
            ref_container = QWidget()
            ref_layout = QHBoxLayout()
            ref_layout.setSpacing(0)
            ref_layout.setContentsMargins(2,2,2,2)
            ref_container.setLayout(ref_layout)
            ref_layout.addWidget(QLabel("Reference frame: "))
            ref_edit = QLineEdit()
            ref_edit.setText(str(min(self.structure.coordset_ids)))
            ref_edit.setFixedWidth(ref_edit.fontMetrics().boundingRect("9999").width())
            ref_edit.setAlignment(Qt.AlignCenter)
            ref_edit.setValidator(QIntValidator())
            ref_layout.addWidget(ref_edit)
            pc_layout.addWidget(ref_container, next(row), 0, 1, 2, alignment=Qt.AlignCenter)
            provider_widget_dict["ref-frame"] = ref_edit

        excludes = self.mgr.excludes(provider_name)
        if excludes:
            exclude_dict = provider_widget_dict["excludes"] = {}
            for exclude, default in excludes.items():
                layout_widget, value_widget = self._make_exclude_widget(exclude, default)
                exclude_dict[exclude] = value_widget
                pc_layout.addWidget(layout_widget, next(row), 0, 1, 2, alignment=Qt.AlignCenter)

        delete_control_area = QWidget()
        area_layout.addWidget(delete_control_area)
        dc_layout = QHBoxLayout()
        dc_layout.setSpacing(0)
        dc_layout.setContentsMargins(0,0,0,0)
        delete_control_area.setLayout(dc_layout)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda *args, f=self._delete_table_entries, pv=provider_name: f(pv))
        dc_layout.addWidget(delete_button, alignment=Qt.AlignRight)
        dc_layout.addWidget(QLabel(" chosen %s" % plural_of(ui_name)), alignment=Qt.AlignLeft)

        return buttons_area

    def _make_exclude_widget(self, kind, default):
        kind_text = self.exclude_interface_text.get(kind, kind)
        possible = self.mgr.exclude_info[kind]
        if possible == self.mgr.bools:
            # check button
            layout_widget = value_widget = QCheckBox("Ignore " + kind_text)
            value_widget.setChecked(eval(default.capitalize()))
        else:
            # popup menu
            layout_widget = QWidget()
            layout = QHBoxLayout()
            layout.setSpacing(0)
            layout.setContentsMargins(0,0,0,0)
            layout_widget.setLayout(layout)
            layout.addWidget(QLabel("Ignore " + kind_text +": "))
            value_widget = QPushButton(default)
            menu = QMenu(value_widget)
            for value in self.mgr.exclude_info[kind]:
                menu.addAction(value)
            menu.triggered.connect(lambda act, but=value_widget: but.setText(act.text()))
            value_widget.setMenu(menu)
            layout.addWidget(value_widget)

        return layout_widget, value_widget

    def _make_table(self, provider_name):
        from chimerax.ui.widgets import ItemTable
        table = ItemTable(allow_user_sorting=False)
        # These first three columns are not put in output files; if these columns are rearranged
        # or additional "skippable" columns are added, update the save_values method
        table.add_column("Color", "rgba8", format=table.COL_FORMAT_OPAQUE_COLOR, title_display=False)
        table.add_column("Shown", "shown", format=table.COL_FORMAT_BOOLEAN, icon="shown")
        val_col_name = self.mgr.ui_name(provider_name)
        if not val_col_name.isupper():
            val_col_name = val_col_name.capitalize()
        #justification="decimal"  guessing in most cases better to center; also avoids fixed-width font
        self._value_columns[provider_name] = table.add_column(val_col_name, "value",
            format=self.mgr.text_format(provider_name), header_justification="center")
        num_atoms = self.mgr.num_atoms(provider_name)
        if num_atoms == 0:
            table.add_column("# Atoms", "num_atoms")
        else:
            for i in range(num_atoms):
                table.add_column("Atom %d" % (i+1), lambda x, i=i: x.atoms[i],
                    format=lambda a: a.string(minimal=True))
        if self.mgr.need_ref_frame(provider_name):
            table.add_column("Ref Frame", "ref_frame")
        table.data = []
        table.launch()
        return table

    def _mouse_event(self, event):
        from matplotlib.backend_bases import MouseButton
        if event.name == "button_press_event":
            if event.button != MouseButton.LEFT:
                return
        elif event.name == "motion_notify_event":
            if MouseButton.LEFT not in event.buttons:
                return
        else:
            raise ValueError("Unexpected Matplotlib event: %s" % event.name)
        for provider, stack in self._plot_stacks.items():
            if event.canvas == stack.widget(1):
                if not event.inaxes:
                    break
                cs_id = round(event.xdata)
                if cs_id != self.structure.active_coordset_id:
                    # rather than directly check if the ID is valid (there could be many coord sets)
                    # just try to set it and catch the error
                    try:
                        self.structure.active_coordset_id = cs_id
                    except IndexError:
                        # non-existent
                        pass
                break

    def _plot_atomic(self, provider_name):
        from chimerax.ui import tool_user_error
        from chimerax.atomic import Atoms, selected_atoms # selected_atoms preserves selection order
        sel_atoms = Atoms([a for a in selected_atoms(self.session) if a.structure == self.structure])
        expected_sel = self.mgr.num_atoms(provider_name)
        ui_name = self.mgr.ui_name(provider_name)
        if expected_sel == 0:
            if not sel_atoms:
                sel_atoms = self.structure.atoms
            exclude_widgets = self._provider_widgets[provider_name].get("excludes", {})
            sel_atoms = self._process_exclude_widgets(sel_atoms, exclude_widgets)
            if not sel_atoms:
                from chimerax.core.commands import commas
                return tool_user_error("No atoms remain after removing %s" %
                    commas([self.exclude_interface_text.get(kind, kind)
                        for kind in self.mgr.exclude_info.keys()], conjunction="and"))
        elif len(sel_atoms) != expected_sel:
            return tool_user_error("Plotting %s requires exactly %d selected atoms in the structure;"
                " %d are currently selected" % (plural_of(ui_name), expected_sel, len(sel_atoms)))
        kw = {}
        if self.mgr.need_ref_frame(provider_name):
            ref_widget = self._provider_widgets[provider_name]["ref-frame"]
            if not ref_widget.hasAcceptableInput():
                return tool_user_error("Reference frame must be an integer")
            frame = int(ref_widget.text())
            if frame not in self.structure.coordset_ids:
                return tool_user_error("%s does not have a coordinate set %d" % (self.structure, frame))
            kw["ref_frame"] = frame
        table = self._tables[provider_name]
        from .manager import PlotValueError
        try:
            table.data += [TableEntry(self, provider_name, sel_atoms, **kw)]
        except PlotValueError as e:
            return tool_user_error("Cannot plot %s for selected atoms: %s" % (ui_name, str(e)))
        self._update_plot(provider_name)

    def _process_exclude_widgets(self, sel_atoms, exclude_widgets):
        for kind, value_widget in exclude_widgets.items():
            if self.mgr.exclude_info[kind] == self.mgr.bools:
                val = value_widget.isChecked()
            else:
                val_text = value_widget.text()
                if val_text in self.mgr.bools:
                    val = eval(val_text.capitalize())
                else:
                    val = val_text
            if val is False:
                continue
            from numpy import logical_not, logical_and
            if kind == "solution":
                sel_atoms = sel_atoms.filter(sel_atoms.structure_categories != "solvent")
                ions = sel_atoms.structure_categories == "ions"
                metals = sel_atoms.elements.is_metal
                non_metal_ions = logical_and(ions, logical_not(metals))
                sel_atom = sel_atoms.filter(logical_not(non_metal_ions))
            elif kind == "hydrogens":
                sel_atoms = sel_atoms.filter(sel_atoms.elements.numbers > 1)
            elif kind == "ligands":
                sel_atoms = sel_atoms.filter(sel_atoms.structure_categories != "ligand")
            elif kind == "metals":
                if val == "alkali":
                    metals = sel_atoms.elements.is_alkali_metal
                else:
                    metals = sel_atoms.elements.is_metal
                sel_atoms = sel_atoms.filter(logical_not(metals))
            else:
                raise ValueError("Unknown kind of atom for 'exclude': %s" % kind)
        return sel_atoms

    def _table_row_output(self, table, datum):
        return [col.display_value(datum) for col in table.columns[3:]] + ["%g" % datum.values[cs_id]
            for cs_id in sorted(list(datum.values.keys()))]

    def _update_plot(self, provider_name):
        stack = self._plot_stacks[provider_name]
        table = self._tables[provider_name]
        if table.data:
            stack.setCurrentIndex(1)
        else:
            stack.setCurrentIndex(0)
            return
        canvas = stack.currentWidget()
        figure = canvas.figure
        if figure.axes:
            axis = figure.axes[0]
            axis.clear()
        else:
            axis = figure.subplots()
            self._mouse_handlers[provider_name] = [
                canvas.mpl_connect('motion_notify_event', self._mouse_event),
                canvas.mpl_connect('button_press_event', self._mouse_event),
            ]
        from matplotlib.ticker import MaxNLocator
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        cs_ids = self.structure.coordset_ids
        cs_ids.sort()
        axis.set_xlim(cs_ids[0], cs_ids[-1])

        for table_entry in table.data:
            if table_entry.shown:
                axis.plot(cs_ids, [table_entry.values[cs_id] for cs_id in cs_ids],
                    color=table_entry.rgba[:3])
        y_min, y_max = self.mgr.min_val(provider_name), self.mgr.max_val(provider_name)
        if y_min is not None:
            axis.set_ylim(ymin=y_min)
        if y_max is not None:
            axis.set_ylim(ymax=y_max)
        axis.set_xlabel("Coord Set")
        ui_name = self.mgr.ui_name(provider_name)
        axis.set_ylabel(ui_name.title() if ui_name.islower() else ui_name)
        self._frame_indicators[provider_name] = axis.axvline(self.structure.active_coordset_id, color='k')
        canvas.draw_idle()

class TableEntry:
    def __init__(self, plot_dialog, provider_name, atoms, *, ref_frame=None):
        self.plot_dialog = plot_dialog
        mgr = plot_dialog.mgr
        self.provider_name = provider_name
        from chimerax.core.colors import distinguish_from
        self.rgba = distinguish_from([(1.0,1.0,1.0,1.0)]
            + [datum.rgba for datum in plot_dialog._tables[provider_name].data])
        self._shown = True
        kw = {}
        if ref_frame is not None:
            kw["ref_frame"] = ref_frame
        self._values = mgr.get_values(provider_name, structure=plot_dialog.structure, atoms=atoms, **kw)
        self.atoms = atoms
        self._ref_frame = ref_frame

    @property
    def num_atoms(self):
        return len(self.atoms)

    @property
    def ref_frame(self):
        if self._ref_frame is None:
            raise AssertionError("Table entry lacks reference frame info")
        return self._ref_frame

    @property
    def rgba8(self):
        return [round(c*255.0) for c in self.rgba]

    @rgba8.setter
    def rgba8(self, rgba8):
        self.rgba = [c/255.0 for c in rgba8]
        if self._shown:
            self.plot_dialog._update_plot(self.provider_name)

    @property
    def shown(self):
        return self._shown

    @shown.setter
    def shown(self, show):
        if show != self._shown:
            self._shown = show
            self.plot_dialog._update_plot(self.provider_name)

    @property
    def value(self):
        return self._values[self.plot_dialog.structure.active_coordset_id]

    @property
    def values(self):
        return self._values

from chimerax.core.settings import Settings
class SavePlotDialogSettings(Settings):
    AUTO_SAVE = {
        "dpi": None,
        "save_format": "PNG",
        "transparent_background": False,
    }

# Cribbed from chimerax.ui.open_save.SaveDialog, but since we need to save the formats ourselves and
# save some formats otherwise unknown to ChimeraX (e.g. EPS, SVG), we provide our own dialog
from Qt.QtWidgets import QFileDialog
class SavePlotDialog(QFileDialog):
    def __init__(self, session, parent = None, *args, **kw):
        self.format_info = [
            ("PNG", "Portable Network Graphics", "png"),
            ("JPEG/JPG", "Joint Photographic Experts Group", "jpg *.jpeg"),
            ("TIFF", "Tagged Image File Format", "tiff"),
            ("PDF", "Portable Document Format", "pdf"),
            ("SVG", "Scalable Vector Graphics", "svg"),
            ("EPS", "Encapsulated PostScript", "eps"),
            ("PS", "PostScript", "ps"),
        ]
        name_filters = ["%s [%s] (*.%s)" % fmt_info for fmt_info in self.format_info]
        self.filter_to_info = {flt: info for flt, info in zip(name_filters, self.format_info)}
        fmt_to_filter = { info[0]: flt for flt, info in self.filter_to_info.items() }
        super().__init__(parent, *args, **kw)
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setOption(QFileDialog.DontUseNativeDialog)
        self.setNameFilters(name_filters)
        self.settings = SavePlotDialogSettings(session, "MD save plot dialog")
        try:
            self.selectNameFilter(fmt_to_filter[self.settings.save_format])
        except KeyError:
            self.selectNameFilter(fmt_to_filter["PNG"])

        custom_area = QFrame(self)
        custom_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
        custom_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = self.layout()
        row = layout.rowCount()
        layout.addWidget(custom_area, row, 0, 1, -1)
        custom_layout = QHBoxLayout()
        custom_area.setLayout(custom_layout)
        custom_layout.addStretch(1)
        self._transparent_checkbox = QCheckBox("Transparent background")
        self._transparent_checkbox.setChecked(self.settings.transparent_background)
        custom_layout.addWidget(self._transparent_checkbox)
        custom_layout.addStretch(1)
        custom_layout.addWidget(QLabel("DPI:"))
        self._dpi_entry = QLineEdit()
        self._dpi_entry.setAlignment(Qt.AlignCenter)
        self._dpi_entry.setPlaceholderText("default")
        self._dpi_entry.setMaximumWidth(50)
        validator = QIntValidator()
        validator.setBottom(1)
        self._dpi_entry.setValidator(validator)
        if self.settings.dpi is not None:
            self._dpi_entry.setText(str(self.settings.dpi))
        custom_layout.addWidget(self._dpi_entry)
        custom_layout.addStretch(1)

    @property
    def dpi(self):
        if self._dpi_entry.hasAcceptableInput():
            return int(self._dpi_entry.text())
        return None

    @property
    def path(self):
        paths = self.selectedFiles()
        if not paths:
            return None
        path = paths[0]
        name_filter = self.selectedNameFilter()
        fmt_name, fmt_desc, suffix_info = self.filter_to_info[name_filter]
        self.settings.save_format = fmt_name
        self.settings.transparent_background = self.transparent_background
        self.settings.dpi = self.dpi
        suffix = '.' + (suffix_info[:suffix_info.index(' ')] if ' ' in suffix_info else suffix_info)
        if path.endswith(suffix):
            return path
        return path + suffix

    @property
    def transparent_background(self):
        return self._transparent_checkbox.isChecked()

_md_tool_windows = {}
def _show_plot_dialog(main_tool_window, structure):
    inst = main_tool_window.tool_instance
    inst_windows = _md_tool_windows.setdefault(inst, {})
    try:
        plot_dialog = inst_windows["plot"]
    except KeyError:
        plot_dialog = inst_windows["plot"] = PlotDialog(main_tool_window.create_child_window("MD Plots"),
            structure)

    plot_dialog.tool_window.shown = True

