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
# Dialog for showing contacts between asymmetric units of a crystal.
#
from chimerax.core.tools import ToolInstance
class CrystalContactsGUI(ToolInstance):

    help = 'help:user/tools/crystalcontacts.html'

    def __init__(self, session, tool_name):

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))


        # Put menu and distance on same line.
        from chimerax.ui.widgets import row_frame
        mdf, mdlayout = row_frame(parent, spacing = 10)
        layout.addWidget(mdf)
        
        # Make menu to choose atomic structure
        self._structure_menu = sm = self._create_structure_menu(mdf)
        mdlayout.addWidget(sm.frame)

        # Contact distance entry
        from chimerax.ui.widgets import EntriesRow
        cd = EntriesRow(mdf, 'contact distance', 3.0, '\N{ANGSTROM SIGN}')
        self._contact_distance = cd.values[0]
        
        mdlayout.addStretch(1)
        
        # Add buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Options panel
        options = self._create_options_pane(parent)
        layout.addWidget(options)

        layout.addStretch(1)    # Extra space at end

        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, CrystalContactsGUI, 'Crystal Contacts', create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_structure_menu(self, parent):
        from chimerax.atomic import AtomicStructure
        from chimerax.ui.widgets import ModelMenu
        # Do not list crystal contact copies in menu.
        model_filter = lambda m: not hasattr(m, '_crystal_contacts_copy') and hasattr(m, 'metadata') and len(m.metadata) > 0
        m = ModelMenu(self.session, parent, label = 'Atomic structure',
                      model_types = [AtomicStructure], model_filter = model_filter)
        return m
    
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Show Contacts', self._show_contacts),
                        ('Delete Contacts', self._delete_contacts),
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
        cp = EntriesRow(f, True, 'Create copies of contacting structures')
        self._copies = cp.values[0]

        rc = EntriesRow(f, True, 'Rainbow color copies')
        self._rainbow = rc.values[0]

        sc = EntriesRow(f, False, 'Show schematic of contacting structures')
        self._schematic = sc.values[0]

        return p

    # ---------------------------------------------------------------------------
    #
    def _show_contacts(self):

        m = self._structure_menu.value
        if m is None:
            return

        cmd = 'crystalcontacts #%s' % m.id_string

        opts = [('distance', 3.0, self._contact_distance.value, '%.4g'),
                ('copies', True, self._copies.value, '%s'),
                ('rainbow', True, self._rainbow.value, '%s'),
                ('schematic', False, self._schematic.value, '%s')]
        options = ['%s %s' % (opt, fmt % value)
                   for opt, default, value, fmt in opts if value != default]
        cmd = ' '.join([cmd] + options)

        from chimerax.core.commands import run
        run(self.session, cmd)
  
    # ---------------------------------------------------------------------------
    #
    def _delete_contacts(self):

        m = self._structure_menu.value
        if m is None:
            return

        cmd = 'crystalcontacts delete #%s' % m.id_string
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
