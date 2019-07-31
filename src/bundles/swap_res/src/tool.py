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

from chimerax.core.tools import ToolInstance


_prd = None
def prep_rotamers_dialog(session, rotamers_tool_name):
    global _prd
    if _prd is None:
        _prd = PrepRotamersDialog(session, rotamers_tool_name)
    return _prd

class PrepRotamersDialog(ToolInstance):

    #help = "help:user/tools/rotamers.html"

    def __init__(self, session, rotamers_tool_name):
        ToolInstance.__init__(self, session, "Choose Rotamer Parameters")
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel
        layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.addWidget(QLabel("Show rotamers for selected residues..."))

        installed_lib_names = set(session.rotamers.library_names(installed_only=True))
        all_lib_names = session.rotamers.library_names(installed_only=False)
        all_lib_names.sort()
        if not all_lib_names:
            raise AssertionError("No rotamers libraries available?!?")
        from chimerax.ui.options import SymbolicEnumOption, EnumOption, OptionsPanel
        class RotLibOption(SymbolicEnumOption):
            labels = [(session.rotamers.library(lib_name).display_name if lib_name in installed_lib_names
                else "%s [not installed]" % lib_name) for lib_name in all_lib_names]
            values = all_lib_names
        from .settings import get_settings
        settings = get_settings(session)
        if settings.library in all_lib_names:
            def_lib = settings.library
        else:
            def_lib = installed_lib_names[0] if installed_lib_names else all_lib_names[0]
        self.rot_lib_option = RotLibOption("Rotamer library", def_lib, self._lib_change_cb)

        rot_lib = session.rotamers.library(self.rot_lib_option.value)
        from chimerax.atomic import selected_atoms
        sel_residues = selected_atoms(session).residues.unique()
        sel_res_types = set([r.name for r in sel_residues])
        res_name_list = self.lib_res_list(rot_lib)
        if len(sel_res_types) == 1:
            def_res_type = rot_lib.map_res_name(sel_res_types.pop(), exemplar=sel_residues[0])
            if def_res_type is None:
                def_res_type = res_name_list[0]
        else:
            def_res_type = res_name_list[0]
        class ResTypeOption(EnumOption):
            values = res_name_list
        self.res_type_option = ResTypeOption("Rotamer type", def_res_type, None)

        opts = OptionsPanel(scrolled=False)
        opts.add_option(self.res_type_option)
        opts.add_option(self.rot_lib_option)
        layout.addWidget(opts)
        """
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.keyPressEvent = session.ui.forward_keystroke
        self.table.setHorizontalHeaderLabels(["Atom 1", "Atom 2", "Distance"])
        #self.table.itemClicked.connect(self._table_change_cb)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0,0,0,0)
        table_layout.setSpacing(0)
        table_layout.addWidget(self.table)
        table_layout.setStretchFactor(self.table, 1)
        layout.addLayout(table_layout)
        layout.setStretchFactor(table_layout, 1)
        button_layout = QHBoxLayout()
        create_button = QPushButton("Create")
        create_button.clicked.connect(self._create_distance)
        create_button.setToolTip("Create distance monitor between two (currently selected) atoms;\n"
            "Alternatively, control-click one atom in graphics view and control-shift-\n"
            "double-click another to bring up context menu with 'Distance' entry")
        button_layout.addWidget(create_button)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self._delete_distance)
        delete_button.setToolTip("Delete distances selected in table (or all if none selected)")
        button_layout.addWidget(delete_button)
        save_info_button = QPushButton("Save Info...")
        save_info_button.clicked.connect(self._save_info)
        save_info_button.setToolTip("Save distance information into a file")
        button_layout.addWidget(save_info_button)
        table_layout.addLayout(button_layout)

        from chimerax.ui.options import SettingsPanel, BooleanOption, ColorOption, IntOption, FloatOption
        panel = SettingsPanel()
        from chimerax.dist_monitor.settings import settings
        from chimerax.core.commands import run
        from chimerax.ui.widgets import hex_color_name
        for opt_name, attr_name, opt_class, opt_class_kw, cmd_arg in [
                ("Color", 'color', ColorOption, {}, 'color %s'),
                ("Number of dashes", 'dashes', IntOption, {'min': 0}, 'dashes %d'),
                ("Decimal places", 'decimal_places', IntOption, {'min': 0}, 'decimalPlaces %d'),
                ("Radius", 'radius', FloatOption, {'min': 'positive', 'decimal_places': 3}, 'radius %g'),
                ("Show \N{ANGSTROM SIGN} symbol", 'show_units', BooleanOption, {}, 'symbol %s')]:
            converter = hex_color_name if opt_class == ColorOption else None
            panel.add_option(opt_class(opt_name, None,
                lambda opt, run=run, converter=converter, ses=self.session, cmd_suffix=cmd_arg:
                run(ses, "distance style " + cmd_suffix
                % (opt.value if converter is None else converter(opt.value))),
                attr_name=attr_name, settings=settings, auto_set_attr=False))
        layout.addWidget(panel)

        from chimerax.dist_monitor.cmd import group_triggers
        self.handlers = [
            group_triggers.add_handler("update", self._fill_table),
            group_triggers.add_handler("delete", self._fill_table)
        ]
        self._fill_table()
        """
        tw.manage(placement=None)

    def delete(self):
        global _prd
        _prd = None
        super().delete()

    def lib_res_list(self, rot_lib):
        res_name_list = list(rot_lib.residue_names) + ["ALA", "GLY"]
        res_name_list.sort()
        return res_name_list

    def _lib_change_cb(self, opt):
        rot_lib = self.session.rotamers.library(self.rot_lib_option.value)
        cur_val = self.res_type_option.value
        self.res_type_option.values = self.lib_res_list(rot_lib)
        self.res_type_option.remake_menu()
