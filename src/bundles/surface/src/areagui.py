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
# Panel for reporting surface area and volume
#
from chimerax.core.tools import ToolInstance
class VolumeAreaGUI(ToolInstance):

    help = 'help:user/tools/measurevolume.html'

    def __init__(self, session, tool_name):

        self._change_handler = None	# Used for auto update when surface changes
        self._last_surface_hash = None
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        tw.shown_changed = lambda shown, self=self: self._track_surface_changes()
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make menus to choose surface menu
        self._surface_menu = sm = self._create_surface_menu(parent)
        layout.addWidget(sm.frame)

        # Volume and area report, and auto update checkbutton
        from chimerax.ui.widgets import EntriesRow
        re = EntriesRow(parent, 'Volume = ', '',
                        'Area = ', '',
                        '     ',
                        True, 'Update automatically')
        self._volume_entry, self._area_entry, self._auto_update = re.values
        self._auto_update.changed.connect(self._track_surface_changes)

        # Update and Help buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        layout.addStretch(1)    # Extra space at end

        # Show values for shown volume
        self._surface_chosen()
        self._track_surface_changes()
        
        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, VolumeAreaGUI, 'Measure Volume and Area',
                                   create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_surface_menu(self, parent):
        from chimerax.core.models import Surface
        from chimerax.map import Volume, VolumeSurface
        def _not_volume_surface(m):
            return not isinstance(m, VolumeSurface)
        from chimerax.ui.widgets import ModelMenu
        m = ModelMenu(self.session, parent, label = 'Surface',
                      model_types = [Volume, Surface],
                      model_filter = _not_volume_surface,
                      model_chosen_cb = self._surface_chosen)
        return m

    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Update', self._update),
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
    def _surface_chosen(self):
        self._update(log = False)

    # ---------------------------------------------------------------------------
    #
    def _update(self, *, log = True):

        surface = self._surface_menu.value
        if surface is None:
            if log:
                self.warn('No surface chosen')
            return

        from . import surface_volume_and_area
        volume, area, hole_count = surface_volume_and_area(surface)
        if volume is None:
            vstr = 'N/A (non-oriented surface)'
        else:
            vstr = _engineering_notation(volume, 4)
            if hole_count > 0:
                vstr += ' (%d holes)' % hole_count

        astr = _engineering_notation(area, 4)

        self._volume_entry.value = vstr
        self._area_entry.value = astr

        if self._auto_update.enabled:
            self._surface_changed(surface)
        
        if log:
            msg = ('Surface %s (#%s): volume = %s, area = %s\n'
                   % (surface.name, surface.id_string, vstr, astr))
            self.session.logger.status(msg)

    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')

    # ---------------------------------------------------------------------------
    #
    def _track_surface_changes(self):
        enable = (self._auto_update.enabled and self.tool_window.shown)
        if enable:
            if self._change_handler is None:
                h = self.session.triggers.add_handler('new frame',
                                                      self._check_for_surface_change)
                self._change_handler = h
        elif self._change_handler:
            self.session.triggers.remove_handler(self._change_handler)
            self._change_handler = None
            
    # ---------------------------------------------------------------------------
    #
    def _check_for_surface_change(self, *unused):
        if self._surface_changed(self._surface_menu.value):
            self._update(log = False)
            
    # ---------------------------------------------------------------------------
    #
    def _surface_changed(self, surface):
        if surface is None:
            return False
        # Submodels are included in surface so check if any changed.
        from chimerax.core.models import Surface
        surfaces = [s for s in surface.all_models()
                    if isinstance(s, Surface)]
        hash = [(len(s.vertices), id(s.vertices), len(s.triangles), id(s.triangles))
                for s in surfaces if s.vertices is not None and s.triangles is not None]
        changed = (hash != self._last_surface_hash)
        self._last_surface_hash = hash
        return changed

# -----------------------------------------------------------------------------
#
def _engineering_notation(value, precision):

    from decimal import Decimal
    format = '%%#.%dg' % precision
    e = Decimal(format % value).to_eng_string()
    e = e.replace('E', 'e')
    e = e.replace('e+', 'e')
    return e

# -----------------------------------------------------------------------------
#
def volume_area_panel(session, create = False):
    return VolumeAreaGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_volume_area_panel(session):
    return volume_area_panel(session, create = True)
