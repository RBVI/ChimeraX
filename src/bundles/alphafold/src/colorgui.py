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


# -----------------------------------------------------------------------------
# Panel for hiding surface dust
#
from chimerax.core.tools import ToolInstance
class AlphaFoldColoringGUI(ToolInstance):

    help = 'help:user/tools/alphafold.html#coloring'

    def __init__(self, session, tool_name):

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make menu to choose AlphaFold structure
        rm = self._create_residues_menu(parent)
        layout.addWidget(rm)

        # Color, hide, show, select buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        layout.addStretch(1)    # Extra space at end

        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, AlphaFoldColoringGUI, 'AlphaFold Coloring',
                                   create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_residues_menu(self, parent):

        from chimerax.ui.widgets import ModelMenuButton
        sm = ModelMenuButton(self.session, filter_func = _is_alphafold_model,
                             autoselect = ModelMenuButton.AUTOSELECT_FIRST_DISPLAYED)
        self._model_menu = sm
        mlist = [m for m in self.session.models.list() if _is_alphafold_model(m)]
        if mlist:
            sm.value = mlist[0]
        
        entries = ('all',
                   'confidence below',
                   'C-alpha distance greater than',
                   'missing structure',
                   'different sequence',
                   '-',
                   'confidence above',
                   'C-alpha distance less than',
                   'paired structure',
                   'same sequence',
                   )
        from chimerax.ui.widgets import EntriesRow
        er = EntriesRow(parent, 'Residues', sm, entries, 50.0, 3.0)
        self._filter_menu, self._confidence, self._ca_distance = fm,con,cd = er.values
        fm.widget.menu().triggered.connect(self._filter_changed)
        conw, cdw = con.widget, cd.widget
        conw.setMaximumWidth(25)
        conw.setVisible(False)
        cdw.setMaximumWidth(25)
        cdw.setVisible(False)
        return er.frame

    # ---------------------------------------------------------------------------
    #
    def _filter_changed(self, action):
        item = action.text()
        show_dist = show_conf = False
        if item.startswith('C-alpha distance'):
            show_dist = True
        elif item.startswith('confidence'):
            show_conf = True
        self._ca_distance.widget.setVisible(show_dist)
        self._confidence.widget.setVisible(show_conf)

    # ---------------------------------------------------------------------------
    #
    @property
    def _confidence_cutoff(self):
        return _string_to_float(self._confidence.value, 50.0)
    @property
    def _distance_cutoff(self):
        return _string_to_float(self._ca_distance.value, 3.0)

    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import ColorButton
        c = ColorButton
        from chimerax.ui.widgets import EntriesRow
        er = EntriesRow(parent,
                        'Color:',
                        ('Custom', self._custom_color),
                        c,c,c,c,c,c,	# Color buttons
                        ' ',
                        ('Hide', self._hide),
                        ('Show', self._show),
                        ('Select', self._select))

        # Use hidden ColorButton for custom coloring.
        self._custom_colorbutton = cc = ColorButton(parent)
        cc.color_changed.connect(self._set_color)
        cc.setVisible(False)

        color_buttons = er.values
        fixed_colors = ('lightgray', 'red', 'magenta', 'yellow', 'lime', 'cyan')
        for b, color in zip(color_buttons, fixed_colors):
            b.color = color
            b.clicked.disconnect(b.show_color_chooser)
            b.clicked.connect(lambda *args, color=color: self._set_color(color))

        return er.frame
        
    # ---------------------------------------------------------------------------
    #
    def _set_color(self, color):
        if not isinstance(color, str):
            from chimerax.core.colors import hex_color
            color = hex_color(color)
        self._run_command('color %%s %s' % color)
    def _custom_color(self):
        self._custom_colorbutton.show_color_chooser()
    def _hide(self):
        self._run_command('hide %s atoms,ribbons')
    def _show(self):
        self._run_command('show %s atoms,ribbons')
    def _select(self):
        self._run_command('select %s')

    # ---------------------------------------------------------------------------
    #
    def _run_command(self, command):
        spec = self._residue_specifier()
        if spec is None:
            return
        cmd = command % spec
        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _residue_specifier(self):
        m = self._model_menu.value
        if m is None:
            return None
        spec = '#' + m.id_string
        filter_mode = self._filter_menu.value
        if filter_mode == 'all':
            fspec = ''
        elif filter_mode.startswith('confidence'):
            op = '<' if filter_mode.endswith('below') else '>='
            fspec = '@@bfactor%s%.1f' % (op, self._confidence_cutoff)
        elif filter_mode.startswith('C-alpha distance'):
            op = '<' if filter_mode.endswith('less than') else '>='
            fspec = '::c_alpha_distance%s%.1f' % (op, self._distance_cutoff)
        elif filter_mode == 'missing structure':
            fspec = '::missing_structure=true'
        elif filter_mode == 'paired structure':
            fspec = '::missing_structure=false'
        elif filter_mode == 'different sequence':
            fspec = '::same_sequence=false'
        elif filter_mode == 'same sequence':
            fspec = '::same_sequence=true'
        spec += fspec
        return spec
    
    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')

# -----------------------------------------------------------------------------
#
def _is_alphafold_model(m):
    return getattr(m, 'alphafold', False) or m.name.endswith('AlphaFold')

# ---------------------------------------------------------------------------
#
def _string_to_float(string, default):
    try:
        v = float(string)
    except ValueError:
        v = default
    return v
    
# -----------------------------------------------------------------------------
#
def alphafold_coloring_panel(session, create = False):
    return AlphaFoldColoringGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_alphafold_coloring_panel(session):
    return alphafold_coloring_panel(session, create = True)
