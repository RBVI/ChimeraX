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
# Panel for showing surface zone
#
from chimerax.core.tools import ToolInstance
class SurfaceZoneGUI(ToolInstance):

    help = 'help:user/tools/surfacezone.html'

    def __init__(self, session, tool_name):

        self._distance_range = (1, 1000)
        self._block_zoning = 0

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make menus to choose surface menu
        self._map_menu, self._atoms_menu, mf = self._create_menus(parent)
        layout.addWidget(mf)

        # Zone distance slider
        self._slider = sl = self._create_slider(parent)
        layout.addWidget(sl.frame)

        # Create buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        layout.addStretch(1)    # Extra space at end

        # Set slider range for shown volume
        self._map_chosen()

        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, SurfaceZoneGUI, 'Surface Zone', create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_menus(self, parent):

        from Qt.QtWidgets import QFrame, QHBoxLayout
        f = QFrame(parent)
        layout = QHBoxLayout(f)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(10)
        f.setLayout(layout)
        
        from chimerax.map import Volume
        from chimerax.ui.widgets import ModelMenu
        m = ModelMenu(self.session, f, label = 'Show surface', model_types = [Volume],
                      model_chosen_cb = self._map_chosen)
        layout.addWidget(m.frame)
        
        from chimerax.atomic import Structure
        a = ModelMenu(self.session, f, label = 'near', model_types = [Structure],
                      special_items = ['selected atoms'])
        layout.addWidget(a.frame)

        layout.addStretch(1)
        
        return m, a, f
    
    # ---------------------------------------------------------------------------
    #
    def _create_slider(self, parent):
        from chimerax.ui.widgets import LogSlider
        s = LogSlider(parent, label = 'Radius', range = self._distance_range,
                      value_change_cb = self._distance_changed,
                      release_cb = self._slider_released_cb)
        return s
  
    # ---------------------------------------------------------------------------
    #
    def _update_slider_distance_range(self):

        v = self._map_menu.value
        if v is None:
            return

        min_dist = min(v.data.step)
        self._distance_range = r = (min_dist, 1000*min_dist)
        self._slider.set_range(r[0], r[1])

    # ---------------------------------------------------------------------------
    #
    def _show_default_distance(self):
        '''Set initial slider value.  Updates zoning if currently zoning.'''
        self._slider.value = 6*self._distance_range[0]

    # ---------------------------------------------------------------------------
    #
    def _show_current_distance(self):
        '''Do not update zoning.'''
        z = self._zoning
        if z:
            # Currently zoning so show current distance limit
            self._block_zoning += 1
            self._slider.set_value(z.distance)
            self._block_zoning -= 1

    # ---------------------------------------------------------------------------
    #
    @property
    def _distance_limit(self):
        return self._slider.value
    
    # ---------------------------------------------------------------------------
    #
    def _slider_released_cb(self):
        if self._zoning:
            self._zone(log_command_only = True)
    
    # ---------------------------------------------------------------------------
    #
    def _distance_changed(self, distance, slider_down):
        if self._block_zoning == 0 and self._zoning:
            self._zone(log_command = not slider_down)

    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Zone', self._zone),
                        ('No Zone', self._unzone),
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
    def _map_chosen(self):
        self._update_slider_distance_range()
        if self._zoning:
            self._show_current_distance()
        else:
            self._show_default_distance()

    # ---------------------------------------------------------------------------
    #
    @property
    def _atoms_specifier(self):
        a = self._atoms_menu.value
        if a is None:
            spec = None
        elif a == 'selected atoms':
            spec = 'sel'
        else:
            spec = '#%s' % a.id_string
        return spec
    
    # ---------------------------------------------------------------------------
    #
    @property
    def _zoning(self):
        v = self._map_menu.value
        if v is None:
            return None
        from .zone import zoning
        for s in v.surfaces:
            z = zoning(s)
            if z:
                return z
        return None
        
    # ---------------------------------------------------------------------------
    #
    def _zone(self, *, log_command = True, log_command_only = False):

        v = self._map_menu.value
        if v is None:
            if log_command:
                self.warn('No surface chosen for zoning')
            return

        aspec = self._atoms_specifier
        if aspec is None:
            if log_command:
                self.warn('No atoms chosen for zoning')
            return
        elif aspec == 'sel':
            from chimerax.atomic import selected_atoms
            if len(selected_atoms(self.session)) == 0:
                if log_command:
                    self.warn('No atoms selected for zoning')
                return

        cmd = ('surface zone #%s near %s distance %.5g'
               % (v.id_string, aspec, self._distance_limit))

        if log_command_only:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, cmd)
        else:
            from chimerax.core.commands import run
            run(self.session, cmd, log = log_command)

    # ---------------------------------------------------------------------------
    #
    def _unzone(self):

        v = self._map_menu.value
        if v is None:
            return

        cmd = 'surface unzone #%s' % v.id_string
        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def warn(self, message):
        log = self.session.logger
        log.warning(message)
        log.status(message, color='red')

# -----------------------------------------------------------------------------
#
def surface_zone_panel(session, create = False):
    return SurfaceZoneGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_surface_zone_panel(session):
    return surface_zone_panel(session, create = True)
