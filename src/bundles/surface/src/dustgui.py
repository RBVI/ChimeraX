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

    self._block_dusting = 0

    ToolInstance.__init__(self, session, tool_name)

    from chimerax.ui import MainToolWindow
    tw = MainToolWindow(self)
    self.tool_window = tw
    parent = tw.ui_area

    from Qt.QtWidgets import QVBoxLayout, QSlider
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(5,0,0,0)
    layout.setSpacing(0)
    parent.setLayout(layout)
        
    # Make menus to choose surface menu
    vm = self._create_volume_menu(parent)
    layout.addWidget(vm)

    # Dust size slider
    sl = self._create_slider(parent)
    layout.addWidget(sl)

    # Hide Dust and Show Dust buttons
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

    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
        
    f = QFrame(parent)
    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(10)
        
    fl = QLabel('Dust surface', f)
    layout.addWidget(fl)

    from chimerax.map import Volume
    from chimerax.ui.widgets import ModelMenuButton
    sm = ModelMenuButton(self.session, class_filter = (Volume,), parent = f)
    self._map_menu = sm
    vlist = self.session.models.list(type = Volume)
    if vlist:
      sm.value = vlist[0]
    sm.value_changed.connect(self._map_chosen)
    layout.addWidget(sm)

    layout.addStretch(1)    # Extra space at end
    
    return f
    
  # ---------------------------------------------------------------------------
  #
  def _create_slider(self, parent):

    self._size_range = (1, 1000)
    
    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel, QDoubleSpinBox, QSlider
    f = QFrame(parent)
    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(4)

    vl = QLabel('Size limit')
    layout.addWidget(vl)

    self._size_entry = se = QDoubleSpinBox(f)
    se.setRange(self._size_range[0], self._size_range[1])
    se.setStepType(se.AdaptiveDecimalStepType)
    se.valueChanged.connect(self._size_entry_changed_cb)
    layout.addWidget(se)

    from Qt.QtCore import Qt
    self._slider = sl = QSlider(Qt.Horizontal, f)
    self._slider_range = (0,10000)
    sl.setRange(self._slider_range[0], self._slider_range[1])
    sl.valueChanged.connect(self._slider_moved_cb)
    sl.sliderReleased.connect(self._slider_released_cb)
    layout.addWidget(sl)

    return f

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
    se = self._size_entry
    se.setRange(r[0], r[1])

    precision = 0 if size_metric.endswith('rank') else 2
    se.setDecimals(precision)

  # ---------------------------------------------------------------------------
  #
  def _show_default_size(self):
    '''Set initial slider value.  Updates dusting if currently dusting.'''
    self._size_entry.setValue(6*self._size_range[0])

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
        self._size_entry.setValue(d.limit)
        self._block_dusting -= 1

  # ---------------------------------------------------------------------------
  # Logarithmic slider range.
  #
  def _size_to_slider_value(self, size):
      smin, smax = self._size_range
      if size <= smin:
          f = 0
      elif size >= smax:
          f = 1
      else:
          from math import log10
          f = log10(size/smin) / log10(smax/smin)

      rmin, rmax = self._slider_range
      v = rmin + f * (rmax-rmin)
      v = int(v + 0.5)
      return v

  # ---------------------------------------------------------------------------
  # Logarithmic slider range.
  #
  def _slider_value_to_size(self, value):
      rmin, rmax = self._slider_range
      f = (value - rmin) / (rmax - rmin)
      smin, smax = self._size_range
      from math import log10, pow
      size = smin * pow(10, f * log10(smax/smin))
      return size

  # ---------------------------------------------------------------------------
  #
  @property
  def _size_limit(self):
      return self._size_entry.value()
  
  # ---------------------------------------------------------------------------
  #
  def _size_entry_changed_cb(self, event):
      size = self._size_entry.value()
      if self._slider_value_to_size(self._slider.value()) != size:
          self._slider.setValue(self._size_to_slider_value(size))
      self._size_changed(size, log_command = not self._slider.isSliderDown())
    
  # ---------------------------------------------------------------------------
  #
  def _slider_moved_cb(self):
      value = self._slider.value()
      if self._size_to_slider_value(self._size_entry.value()) != value:
          size = self._slider_value_to_size(value)
          self._size_entry.setValue(size)
    
  # ---------------------------------------------------------------------------
  #
  def _slider_released_cb(self):
      if self._dusting:
          self._dust(log_command_only = True)
    
  # ---------------------------------------------------------------------------
  #
  def _size_changed(self, size, log_command):
      if self._block_dusting == 0 and self._dusting:
          self._dust(log_command = log_command)

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
    p = CollapsiblePanel(parent, title = None)
    self._options_panel = p
    f = p.content_area

    from Qt.QtWidgets import QVBoxLayout
    layout = QVBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    import sys
    if sys.platform == 'darwin':
      layout.setSpacing(0)  # Avoid very large spacing Qt 5.15.2, macOS 10.15.7

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
    
    from Qt.QtWidgets import QFrame, QHBoxLayout, QPushButton
    f = QFrame(parent)
    layout = QHBoxLayout(f)
    layout.setContentsMargins(0,0,0,0)
    layout.setSpacing(10)
    
    for name, callback in (('Hide dust', self._dust),
                           ('Show dust', self._undust),
                           ('Options', self._show_or_hide_options),
                           ('Help', self._show_help)):
      b = QPushButton(name, f)
      b.clicked.connect(callback)
      layout.addWidget(b)
        
    layout.addStretch(1)    # Extra space at end

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
            self.session.logger.status('No surface chosen for hiding dust')
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
    
# -----------------------------------------------------------------------------
#
def hide_dust_panel(session, create = False):

  return HideDustGUI.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_hide_dust_panel(session):

  return hide_dust_panel(session, create = True)
