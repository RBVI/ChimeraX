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
class HideDustGUI(ToolInstance):

    help = 'help:user/tools/hidedust.html'

    def __init__(self, session, tool_name):

        self._size_range = (1, 1000)
        self._block_dusting = 0

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make menus to choose surface menu
        self._map_menu = vm = self._create_volume_menu(parent)
        layout.addWidget(vm.frame)

        # Dust size slider
        self._slider = sl = self._create_slider(parent)
        layout.addWidget(sl.frame)

        # Hide Dust, Show Dust and Options buttons
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Options panel
        options = self._create_options_gui(parent)
        layout.addWidget(options)

        layout.addStretch(1)    # Extra space at end

        # Set slider range for shown volume
        self._map_chosen()

        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, HideDustGUI, 'Hide Dust', create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_volume_menu(self, parent):
        from chimerax.map import Volume
        from chimerax.ui.widgets import ModelMenu
        m = ModelMenu(self.session, parent, label = 'Dust surface',
                      model_types = [Volume], model_chosen_cb = self._map_chosen)
        return m
    
    # ---------------------------------------------------------------------------
    #
    def _create_slider(self, parent):
        from chimerax.ui.widgets import LogSlider
        s = LogSlider(parent, label = 'Size limit', range = self._size_range,
                      value_change_cb = self._size_changed,
                      release_cb = self._slider_released_cb)
        return s
  
    # ---------------------------------------------------------------------------
    #
    def _update_slider_size_range(self):

        v = self._map_menu.value
        if v is None:
            return

        min_size = min(v.data.step)
        size_metric = self._size_metric.value
        if size_metric == 'size':
            r = (min_size, 1000*min_size)
        elif size_metric == 'area':
            r = (min_size*min_size, 1e6*min_size)
        elif size_metric == 'volume':
            r = (min_size**3, 1e9*min_size)
        else:
            # Rank metrics
            r = (1, 1000)

        self._size_range = r
        self._slider.set_range(r[0], r[1])

        precision = 0 if size_metric.endswith('rank') else 2
        self._slider.set_precision(precision)

    # ---------------------------------------------------------------------------
    #
    def _show_default_size(self):
        '''Set initial slider value.  Updates dusting if currently dusting.'''
        self._slider.value = 6*self._size_range[0]

    # ---------------------------------------------------------------------------
    #
    def _show_current_size_and_metric(self):
        '''Do not update dusting.'''
        d = self._dusting
        if d:
            # Currently hiding dust so show current size limit
            self._block_dusting += 1
            if d.metric != self._size_metric.value:
                self._size_metric.value = d.metric	# Does not fire callack
                self._size_metric_changed_cb()
            self._slider.set_value(d.limit)
            self._block_dusting -= 1

    # ---------------------------------------------------------------------------
    #
    @property
    def _size_limit(self):
        return self._slider.value
    
    # ---------------------------------------------------------------------------
    #
    def _slider_released_cb(self):
        if self._dusting:
            self._dust(log_command_only = True)
    
    # ---------------------------------------------------------------------------
    #
    def _size_changed(self, size, slider_down):
        if self._block_dusting == 0 and self._dusting:
            self._dust(log_command = not slider_down)

    # ---------------------------------------------------------------------------
    #
    def _size_metric_changed_cb(self):
        self._update_slider_size_range()
        d = self._dusting
        if d and d.metric == self._size_metric:
            self._show_current_size_and_metric()
        else:
            self._show_default_size()
      
    # ---------------------------------------------------------------------------
    #
    def _create_options_gui(self, parent):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        from chimerax.ui.widgets import EntriesRow
        se = EntriesRow(f, 'Hide small blobs based on',
                        ('size', 'area', 'volume', 'size rank', 'area rank', 'volume rank'))
        self._size_metric = sm = se.values[0]
        sm.widget.menu().triggered.connect(self._size_metric_changed_cb)

        return p

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Hide Dust', self._dust),
                        ('Show Dust', self._undust),
                        ('Options', self._show_or_hide_options),
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
        self._update_slider_size_range()
        if self._dusting:
            self._show_current_size_and_metric()
        else:
            self._show_default_size()

    # ---------------------------------------------------------------------------
    #
    @property
    def _dusting(self):
        v = self._map_menu.value
        if v is None:
            return None
        from .dust import dusting
        for s in v.surfaces:
            d = dusting(s)
            if d:
                return d
        return None
        
    # ---------------------------------------------------------------------------
    #
    def _dust(self, *, log_command = True, log_command_only = False):

        v = self._map_menu.value
        if v is None:
            if log_command:
                self.warn('No surface chosen for hiding dust')
            return

        cmd = 'surface dust #%s size %.5g' % (v.id_string, self._size_limit)

        size_metric = self._size_metric.value
        if size_metric != 'size':
            from chimerax.core.commands import quote_if_necessary
            cmd += ' metric %s' % quote_if_necessary(size_metric)

        if log_command_only:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, cmd)
        else:
            from chimerax.core.commands import run
            run(self.session, cmd, log = log_command)

    # ---------------------------------------------------------------------------
    #
    def _undust(self):

        v = self._map_menu.value
        if v is None:
            return

        cmd = 'surface undust #%s' % v.id_string
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
def hide_dust_panel(session, create = False):
    return HideDustGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_hide_dust_panel(session):
    return hide_dust_panel(session, create = True)
