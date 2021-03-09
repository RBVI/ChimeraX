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

# -----------------------------------------------------------------------------
# Dialog for making copies of PDB molecule to fill out crystal unit cell.
#
from chimerax.core.tools import ToolInstance
class UnitCellGUI(ToolInstance):

    help = 'help:user/tools/unitcell.html'

    def __init__(self, session, tool_name):

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make menu to choose atomic structure
        self._structure_menu = sm = self._create_structure_menu(parent)
        layout.addWidget(sm.frame)

        # Unit cell info
        from Qt.QtWidgets import QLabel
        self._info_text = inf = QLabel(parent)
        layout.addWidget(inf)
        
        # Add buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Options panel
        options = self._create_options_pane(parent)
        layout.addWidget(options)

        layout.addStretch(1)    # Extra space at end

        # Show values for shown volume
        self._structure_chosen()
        
        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, UnitCellGUI, 'Unit Cell', create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_structure_menu(self, parent):
        from chimerax.atomic import AtomicStructure
        from chimerax.ui.widgets import ModelMenu
        # Do not list unit cell copies in menu.
        model_filter = lambda m: not hasattr(m, '_unit_cell_copy') and hasattr(m, 'metadata') and len(m.metadata) > 0
        m = ModelMenu(self.session, parent, label = 'Atomic structure',
                      model_types = [AtomicStructure], model_filter = model_filter,
                      model_chosen_cb = self._structure_chosen)
        return m
    
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Make Copies', self._make_copies),
                        ('Outline', self._show_or_hide_outline),
                        ('Delete Copies', self._delete_copies),
                        ('Options', self._show_or_hide_options),
                        ('Help', self._show_help)],
                       spacing = 10)
        return f
    
    # ---------------------------------------------------------------------------
    #
    def _create_options_pane(self, parent):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        from chimerax.ui.widgets import EntriesRow
        sf = EntriesRow(f, True, 'Use crystal symmetries from file')
        self._use_smtry_records = sf.values[0]

        ucr = EntriesRow(f, True, 'Use space group if symmetries are missing')
        self._use_cryst1_smtry = ucr.values[0]

        umr = EntriesRow(f, True, 'Use non-crystallographic symmetry')
        self._use_mtrix_records = umr.values[0]

        pam = EntriesRow(f, True, 'Pack structures in unit cell')
        self._pack_molecules = pam.values[0]

        co = EntriesRow(f, 'Cell origin ', '')
        self._grid_orig = go = co.values[0]
        go.value = '0 0 0'
        go.widget.returnPressed.connect(self._origin_changed)

        nc = EntriesRow(f, 'Number of cells', '', 'offset', '')
        self._ncells = nce = nc.values[0]
        nce.value = '1 1 1'
        nce.widget.returnPressed.connect(self._make_copies)
        self._ocells = oce = nc.values[1]
        oce.value = '0 0 0'
        oce.widget.returnPressed.connect(self._make_copies)

        return p

    # ---------------------------------------------------------------------------
    #
    def _make_copies(self):

        m = self._structure_menu.value
        if m is None:
            return

        use_sym_from_file = self._use_smtry_records.enabled
        use_spacegroup = self._use_cryst1_smtry.enabled
        use_ncs = self._use_mtrix_records.enabled
        pack = self._pack_molecules.enabled
        origin = self._grid_origin()
        num_cells = self._number_of_cells()
        offset = self._cell_offset()

        cmd = 'unitcell #%s' % m.id_string

        opts = [('cells', (1,1,1), num_cells, '%d,%d,%d'),
                ('offset', (0,0,0), offset, '%d,%d,%d'),
                ('origin', (0,0,0), origin, '%g,%g,%g'),
                ('symFromFile', True, use_sym_from_file, '%s'),
                ('spacegroup', True, use_spacegroup, '%s'),
                ('ncs', True, use_ncs, '%s'),
                ('pack', True, pack, '%s')]
        options = ['%s %s' % (opt, fmt % value)
                   for opt, default, value, fmt in opts if value != default]
        cmd = ' '.join([cmd] + options)

        from chimerax.core.commands import run
        run(self.session, cmd)
    
    # ---------------------------------------------------------------------------
    # Find origin of cell in unit cell grid containing specified point.
    #
    def _grid_origin(self):

        try:
            gorigin = tuple(float(s) for s in self._grid_orig.value.split())
        except ValueError:
            # TODO: should warn about unparsable origin values
            gorigin = (0,0,0)

        if len(gorigin) != 3:
            gorigin = (0,0,0)

        return gorigin

    # ---------------------------------------------------------------------------
    #
    def _number_of_cells(self):

        try:
            nc = tuple(int(s) for s in self._ncells.value.split())
        except ValueError:
            # TODO: should warn about unparsable origin values
            nc = (1,1,1)

        if len(nc) != 3:
            nc = (1,1,1)

        return nc

    # ---------------------------------------------------------------------------
    #
    def _cell_offset(self):

        try:
            oc = tuple(int(s) for s in self._ocells.value.split())
        except ValueError:
            # TODO: should warn about unparsable origin values
            oc = (1,1,1)

        if len(oc) != 3:
            oc = (1,1,1)

        return oc
    
    # ---------------------------------------------------------------------------
    #
    def _origin_changed(self):

        m = self._structure_menu.value
        if m is None:
            return

        from . import unitcell
        outline_model = unitcell.outline_model(m)
        if outline_model:
            unitcell.show_outline_model(m, self._grid_origin(), outline_model)

        if unitcell.showing_copies(m):
            self._make_copies()
  
    # ---------------------------------------------------------------------------
    #
    def _delete_copies(self):

        m = self._structure_menu.value
        if m is None:
            return

        cmd = 'unitcell delete #%s' % m.id_string
        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_outline(self):
        m = self._structure_menu.value
        if m is None:
            return

        cmd = 'unitcell outline #%s' % m.id_string
        from . import unitcell
        outline_model = unitcell.outline_model(m)
        if outline_model:
            cmd += ' close'
        else:
            origin = self._grid_origin()
            if origin != (0,0,0):
                cmd += ' origin %g,%g,%g' % origin

        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    # ---------------------------------------------------------------------------
    #
    def _structure_chosen(self):
        self._update_info()

    # ---------------------------------------------------------------------------
    #
    def _update_info(self):
        m = self._structure_menu.value
        if m is None:
            return
        from . import unitcell
        info = unitcell.unit_cell_info(m)
        self._info_text.setText(info)

    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')
