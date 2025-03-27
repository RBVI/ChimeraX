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
    QSizePolicy, QWidget, QStackedWidget, QGridLayout
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
    plot_menu = menu.addMenu("Plot")

    for provider_name in mgr.provider_names:
        ui_name = mgr.ui_name(provider_name)
        menu_name = plural_of(ui_name)
        if ui_name.lower() == ui_name:
            # no caps
            menu_name = menu_name.capitalize()

        action = QAction(menu_name, plot_menu)
        action.triggered.connect(lambda *args, tw=parent_tool_window, s=structure, name=provider_name:
            _show_plot(name, tw, s))
        plot_menu.addAction(action)

class PlotDialog:
    def __init__(self, plot_window, structure):
        self.tool_window = tw = plot_window
        def cleanup(pd=self):
            inst = pd.tool_window.tool_instance
            del _md_tool_windows[inst]["plot"]
            if not _md_tool_windows[inst]:
                del _md_tool_windows[inst]
            for handler in pd.handlers:
                handler.remove()
            pd.handlers.clear()
            for provider, cids in pd._mouse_handlers.items():
                canvas = self._plot_stacks[provider].widget(1)
                for cid in cids:
                    canvas.mpl_disconnect(cid)
            pd._mouse_handlers.clear()
            delattr(pd.tool_window, 'cleanup')
        tw.cleanup = cleanup
        self.session = structure.session
        self.structure = structure
        self.handlers = [structure.triggers.add_handler('changes', self._changes_cb)]
        #TODO: respond to structure deletion
        from .manager import get_plotting_manager
        self.mgr = get_plotting_manager(self.session)
        from Qt.QtWidgets import QHBoxLayout, QTabWidget
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setTabsClosable(True)
        #TODO tabCloseRequested(index) signal
        layout.addWidget(self.plot_tabs, stretch=1)

        self.tab_info = {}
        self._tables = {}
        self._plot_stacks = {}
        self._value_columns = {}
        self._frame_indicators = {}
        self._mouse_handlers = {}

        tw.manage(None)

    def make_tab(self, provider_name):
        if self.mgr.num_atoms(provider_name) is None:
            return self._make_scalar_tab(provider_name)
        return self._make_atomic_tab(provider_name)

    def show_tab(self, provider_name):
        try:
            tab_name, tab_widget = self.tab_info[provider_name]
        except KeyError:
            tab_name, tab_widget = self.tab_info[provider_name] = self.make_tab(provider_name)

        self.plot_tabs.setCurrentWidget(tab_widget)

    def _changes_cb(self, trig_name, data):
        s, changes = data
        if 'active_coordset changed' in changes.structure_reasons():
            for provider, table in self._tables.items():
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
        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(lambda *args, f=self._plot_atomic, pv=provider_name: f(pv))
        pc_layout.addWidget(plot_button, 0, 0, alignment=Qt.AlignRight)
        num_atoms = self.mgr.num_atoms(provider_name)
        atom_string = "any number of" if num_atoms == 0 else "%d" % num_atoms
        pc_layout.addWidget(QLabel(" %s from %s selected atoms" % (ui_name, atom_string)), 0, 1,
            alignment=Qt.AlignLeft)
        if num_atoms == 0:
            reminder = QLabel("(If no selection, all atoms)")
            from chimerax.ui import shrink_font
            shrink_font(reminder)
            pc_layout.addWidget(reminder, 1, 0, 1, 2, alignment=Qt.AlignCenter)

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

    def _make_table(self, provider_name):
        from chimerax.ui.widgets import ItemTable
        table = ItemTable(allow_user_sorting=False)
        table.add_column("Color", "rgba8", format=table.COL_FORMAT_OPAQUE_COLOR, title_display=False)
        table.add_column("Shown", "shown", format=table.COL_FORMAT_BOOLEAN, icon="shown")
        self._value_columns[provider_name] = table.add_column(self.mgr.ui_name(provider_name).capitalize(),
            "value", format=self.mgr.text_format(provider_name),
            #justification="decimal"  guessing in most cases better to center; also avoids fixed-width font
            header_justification="center")
        for i in range(self.mgr.num_atoms(provider_name)):
            table.add_column("Atom %d" % (i+1), lambda x, i=i: x.atoms[i],
                format=lambda a: a.string(minimal=True))
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
                        # non-exstent
                        pass
                break

    def _plot_atomic(self, provider_name):
        from chimerax.ui import tool_user_error
        s_atoms = self.structure.atoms
        sel_atoms = s_atoms[s_atoms.selecteds == True]
        expected_sel = self.mgr.num_atoms(provider_name)
        ui_name = self.mgr.ui_name(provider_name)
        if expected_sel == 0:
            if not sel_atoms:
                sel_atoms = s_atoms
        elif len(sel_atoms) != expected_sel:
            return tool_user_error("Plotting %s requires exactly %d selected atoms in the structure;"
                " %d are currently selected" % (plural_of(ui_name), expected_sel, len(sel_atoms)))
        table = self._tables[provider_name]
        table.data += [TableEntry(self, provider_name, sel_atoms)]
        self._update_plot(provider_name)

    def _update_plot(self, provider_name):
        stack = self._plot_stacks[provider_name]
        stack.setCurrentIndex(1)
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

        table = self._tables[provider_name]
        for table_entry in table.data:
            axis.plot(cs_ids, [table_entry.values[cs_id] for cs_id in cs_ids], color=table_entry.rgba[:3])
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
    def __init__(self, plot_dialog, provider_name, atoms):
        self.plot_dialog = plot_dialog
        self.provider_name = provider_name
        self.atoms = atoms
        from chimerax.core.colors import distinguish_from
        self.rgba = distinguish_from([(1.0,1.0,1.0,1.0)]
            + [datum.rgba for datum in plot_dialog._tables[provider_name].data])
        self.shown = True
        self._values = plot_dialog.mgr.get_values(provider_name,
                            structure=plot_dialog.structure, atoms=atoms)

    @property
    def rgba8(self):
        return [round(c*255.0) for c in self.rgba]

    @rgba8.setter
    def rgba8(self, rgba8):
        self.rgba = [c/255.0 for c in rgba8]

    @property
    def value(self):
        return self._values[self.plot_dialog.structure.active_coordset_id]

    @property
    def values(self):
        return self._values

_md_tool_windows = {}
def _show_plot(provider_name, main_tool_window, structure):
    inst = main_tool_window.tool_instance
    inst_windows = _md_tool_windows.setdefault(inst, {})
    try:
        plot_dialog = inst_windows["plot"]
    except KeyError:
        plot_dialog = inst_windows["plot"] = PlotDialog(main_tool_window.create_child_window("MD Plots"),
            structure)

    plot_dialog.show_tab(provider_name)
    plot_dialog.tool_window.shown = True

