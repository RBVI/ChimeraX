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
class AlphaFoldGUI(ToolInstance):

    help = 'help:user/tools/alphafold.html'

    def __init__(self, session, tool_name):

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        heading = ('<html>'
                   'AlphaFold database and structure prediction'
                   '<ul style="margin-top: 5;">'
                   '<li><b>Fetch</b> - Open the database structure with the most similar sequence'
                   '<li><b>Search</b> - Find similar sequences in the AlphaFold database using BLAST'
                   '<li><b>Predict</b> - Compute a new structure using AlphaFold on Google servers'
                   '</ul></html>')
        from Qt.QtWidgets import QLabel
        hl = QLabel(heading)
        layout.addWidget(hl)
        
        # Make menu to choose sequence
        sm = self._create_sequence_menu(parent)
        self._sequence_frame = sm
        layout.addWidget(sm)

        # Sequence entry field
        from Qt.QtWidgets import QTextEdit
        self._sequence_entry = se = QTextEdit(parent)
        layout.addWidget(se)
                
        # Search, Fetch, and Predict buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        layout.addStretch(1)    # Extra space at end

        self._update_entry_display()
        
        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, AlphaFoldGUI, 'AlphaFold', create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_sequence_menu(self, parent):
        from Qt.QtWidgets import QFrame, QLabel, QPushButton, QMenu
        f = QFrame(parent)
        from chimerax.ui.widgets import horizontal_layout
        layout = horizontal_layout(f, margins = (2,0,0,0), spacing = 5)
        ml = QLabel('Sequence')
        layout.addWidget(ml)
        self._seq_button = mb = QPushButton(f)
        layout.addWidget(mb)
        mb.pressed.connect(self._update_sequence_menu)
        me = self._menu_entries()
        if me:
            mb.setText(me[0])
        self._seq_menu = m = QMenu(mb)
        mb.setMenu(m)
        m.triggered.connect(self._menu_selection_cb)

        # UniProt entry field
        from Qt.QtWidgets import QLineEdit
        self._uniprot_entry = ue = QLineEdit(f)
        ue.setMaximumWidth(200)
        layout.addWidget(ue)

        layout.addStretch(1)	# Extra space at end
        return f

    # ---------------------------------------------------------------------------
    #
    def _menu_selection_cb(self, action):
        text = action.text()
        self._seq_button.setText(text)
        self._update_entry_display()

    # ---------------------------------------------------------------------------
    #
    def _update_entry_display(self):
        text = self._seq_button.text()
        if text == 'Paste':
            show_seq, show_uniprot = True, False
        elif text == 'UniProt identifier':
            show_seq, show_uniprot = False, True
        else:
            show_seq, show_uniprot = False, False
        self._sequence_entry.setVisible(show_seq)
        self._uniprot_entry.setVisible(show_uniprot)

    # ---------------------------------------------------------------------------
    #
    def _update_sequence_menu(self):
        m = self._seq_menu
        m.clear()
        for value in self._menu_entries():
            m.addAction(value)

    # ---------------------------------------------------------------------------
    #
    def _menu_entries(self):
        from chimerax.atomic import all_atomic_structures, Residue
        slist = all_atomic_structures(self.session)
        values = []
        for s in slist:
            if len(s.chains) > 1:
                for c in s.chains:
                    if c.polymer_type == Residue.PT_AMINO:
                        values.append('#%s/%s' % (s.id_string, c.chain_id))
            if len([c for c in s.chains if c.polymer_type == Residue.PT_AMINO]) > 0:
                values.append('#%s' % (s.id_string))
        values.extend(['Paste', 'UniProt identifier'])
        return values

    # ---------------------------------------------------------------------------
    #
    def _sequence_specifier(self, action):
        e = self._seq_button.text()
        if e == 'Paste':
            seq = _remove_whitespace(self._sequence_entry.toPlainText())
        elif e == 'UniProt identifier':
            seq = _remove_whitespace(self._uniprot_entry.text())
        else:
            # Chain specifier
            seq = e
        if len(seq) == 0:
            seq = None
        return seq
        
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Fetch', self._fetch),
                        ('Search', self._search),
                        ('Predict', self._predict),
                        ('Coloring', self._coloring),
                        ('Help', self._show_help)],
                       spacing = 10)
        return f

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)
        
    # ---------------------------------------------------------------------------
    #
    def _search(self):
        self._run_command('search')
    def _fetch(self):
        self._run_command('match')
    def _predict(self):
        self._run_command('predict')
    def _coloring(self):
        from . import colorgui
        colorgui.show_alphafold_coloring_panel(self.session)
        
    # ---------------------------------------------------------------------------
    #
    def _run_command(self, action):
        seq = self._sequence_specifier(action)
        if seq is None:
            self.warn('No sequence chosen for AlphaFold %s' % action)
            return

        if action in ('search', 'predict'):
            nseq = self._sequence_count(seq)
            if nseq > 1:
                self.warn('AlphaFold %s requires a single sequence, got %d sequences'
                          % (action, nseq))
                return
            
        cmd = 'alphafold %s %s' % (action, seq)

        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def _sequence_count(self, seq):
        if seq.startswith('#') and '/' not in seq:
            from chimerax.atomic import UniqueChainsArg
            try:
                chains, used, rest = UniqueChainsArg.parse(seq, self.session)
            except Exception:
                return 1
            return len(chains)
        return 1
            
    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')

# -----------------------------------------------------------------------------
#
def _remove_whitespace(string):
    from string import whitespace
    return string.translate(str.maketrans('', '', whitespace))

# -----------------------------------------------------------------------------
#
def alphafold_panel(session, create = False):
    return AlphaFoldGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_alphafold_panel(session):
    return alphafold_panel(session, create = True)
