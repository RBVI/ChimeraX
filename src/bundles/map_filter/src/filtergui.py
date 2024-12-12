# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
# Panel for running various filters (Gaussian, median, binning, ...)
#
from chimerax.core.tools import ToolInstance
class MapFilterPanel(ToolInstance):

  help = 'help:user/tools/mapfilter.html'

  def __init__(self, session, tool_name):

    self._last_filter_map = None
    self._last_command = None
    self._displayed_filter_widget = None
    self._parameter_widgets = pwidgets = {}
    
    ToolInstance.__init__(self, session, tool_name)

    from chimerax.ui import MainToolWindow
    tw = MainToolWindow(self)
    self.tool_window = tw
    parent = tw.ui_area

    from chimerax.ui.widgets import vertical_layout, EntriesRow

    layout = vertical_layout(parent, margins = (5,0,5,0))

    ft = EntriesRow(parent, 'Filter type',
                    ('Gaussian', 'Sharpen', 'Median 3x3x3', 'Bin', 'Scale',
                     'Fourier Transform', 'Laplacian', 'Flatten'))
    self._filter_type = fm = ft.values[0]
    fm.widget.menu().triggered.connect(self._filter_type_changed_cb)

    # Volume menu
    mf = self._create_map_menu(parent)
    ft.frame.layout().insertWidget(2, mf)

    # Width range
    self._width_range = (1,100)
    v = self._map_menu.value
    if v:
      s = min(v.data.step)
      self._width_range = (0.5*s, 100*s)

    # Gaussian width slider
    self._width_slider = sl = self._create_width_slider(parent)
    layout.addWidget(sl.frame)
    pwidgets['Gaussian'] = sl.frame
    self._displayed_filter_widget = sl.frame
    
    # Bfactor slider
    self._bfactor_range = (1,500)
    self._bfactor_slider = bsl = self._create_bfactor_slider(parent)
    layout.addWidget(bsl.frame)
    bsl.frame.setVisible(False)
    pwidgets['Sharpen'] = bsl.frame

    # Median filter iterations
    from chimerax.ui.widgets import EntriesRow
    mfi = EntriesRow(parent, 'Iterations', 1)
    self._median_iterations = mfi.values[0]
    mfi.frame.setVisible(False)
    pwidgets['Median 3x3x3'] = mfi.frame

    # Bin size
    from chimerax.ui.widgets import EntriesRow
    bsi = EntriesRow(parent, 'Bin size', '')	# Use string to allow 3 comma-separated values
    self._bin_size = bsz = bsi.values[0]
    bsz.value = '2'
    bsi.frame.setVisible(False)
    pwidgets['Bin'] = bsi.frame

    # Shift and scale
    ssc = EntriesRow(parent, 'Shift', 0.0, 'Scale', 1.0)
    self._shift, self._scale = ssc.values
    ssc.frame.setVisible(False)
    pwidgets['Scale'] = ssc.frame
    
    # Flatten method
    fme = EntriesRow(parent, 'Method', ('multiply linear', 'divide linear'))
    self._flatten_method = fme.values[0]
    fme.frame.setVisible(False)
    pwidgets['Flatten'] = fme.frame
    
    # Filter, Options, Help buttons
    bf = self._create_action_buttons(parent)
    layout.addWidget(bf)

    # Options panel
    options = self._create_options_gui(parent)
    layout.addWidget(options)

    tw.manage(placement="side")

  # ---------------------------------------------------------------------------
  #
  @classmethod
  def get_singleton(self, session, create=True):
    from chimerax.core import tools
    return tools.get_singleton(session, MapFilterPanel, 'Map Filter', create=create)

  # ---------------------------------------------------------------------------
  #
  def _create_map_menu(self, parent):

    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
        
    mf = QFrame(parent)
    mlayout = QHBoxLayout(mf)
    mlayout.setContentsMargins(0,0,0,0)
    mlayout.setSpacing(10)
        
    fl = QLabel('Map', mf)
    mlayout.addWidget(fl)

    from chimerax.map import Volume
    from chimerax.ui.widgets import ModelMenuButton
    self._map_menu = mm = ModelMenuButton(self.session, class_filter = Volume)
    vlist = self.session.models.list(type = Volume)
    vdisp = [v for v in vlist if v.display]
    if vdisp:
      mm.value = vdisp[0]
    elif vlist:
      mm.value = vlist[0]
    mm.value_changed.connect(self._map_menu_changed)
    mlayout.addWidget(mm)
    mlayout.addStretch(1)    # Extra space at end

    return mf

  # ---------------------------------------------------------------------------
  #
  def _map_menu_changed(self):
    self._last_filter_map = None
    
  # ---------------------------------------------------------------------------
  #
  def _create_action_buttons(self, parent):
    from chimerax.ui.widgets import button_row
    f = button_row(parent,
                   [('Filter', self._filter),
                    ('Options', self._show_or_hide_options),
                    ('Help', self._show_help)],
                   spacing = 10)
    return f

  # ---------------------------------------------------------------------------
  #
  def _filter_type_changed_cb(self):
    '''Show the GUI controls for the parameters of the current filter method.'''

    ftype = self._filter_type.value
    pwidgets = self._parameter_widgets.get(ftype)

    dfw = self._displayed_filter_widget
    if pwidgets is not dfw:
      if dfw:
        dfw.setVisible(False)
      if pwidgets:
        pwidgets.setVisible(True)
      self._displayed_filter_widget = pwidgets
      self._last_filter_map = None

    # Gray out value type for filters that don't support it.
    supports_value_type = ftype in ('Gaussian', 'Sharpen', 'Scale')
    self._value_type_frame.setEnabled(supports_value_type)

  # ---------------------------------------------------------------------------
  #
  def _filter(self):
    ftype = self._filter_type.value
    if ftype == 'Gaussian':
      return self._filter_gaussian()
    elif ftype == 'Sharpen':
      return self._filter_sharpen()
    elif ftype == 'Median 3x3x3':
      return self._filter_median()
    elif ftype == 'Bin':
      return self._filter_bin()
    elif ftype == 'Scale':
      return self._filter_scale()
    elif ftype == 'Fourier Transform':
      return self._filter_fourier()
    elif ftype == 'Laplacian':
      return self._filter_laplacian()
    elif ftype == 'Flatten':
      return self._filter_flatten()
    
  # ---------------------------------------------------------------------------
  #
  def _filter_gaussian(self, sharpen = False):
    v = self._map_menu.value
    if sharpen:
      args = ['volume sharpen', '#' + v.id_string,
              'bfactor', '%.5g' % self._bfactor_slider.value]
    else:
      args = ['volume gaussian', '#' + v.id_string,
              'sdev', '%.5g' % self._width_slider.value]
    args.extend(self._value_type_args())
    args.extend(self._subregion_and_step_args(v))
    vg = self._run_filter(args, v)
    return vg

  # ---------------------------------------------------------------------------
  #
  def _subregion_and_step_args(self, v):
    args = []
    ijk_min, ijk_max, ijk_step = [tuple(ijk) for ijk in v.region]
    if self._use_displayed_subregion.enabled and not v.is_full_region(any_step = True):
      args.extend(('subregion', '%d,%d,%d' % ijk_min + ',%d,%d,%d' % ijk_max))
    if self._use_displayed_step.enabled and ijk_step != (1,1,1):
      iso_step = (ijk_step[0] == ijk_step[1] and ijk_step[1] == ijk_step[2])
      step = ('%d' % ijk_step[0]) if iso_step else ('%d,%d,%d' % ijk_step)
      args.extend(('step', step))
    return args

  # ---------------------------------------------------------------------------
  #
  def _value_type_args(self):
    args = []
    vtype = self._value_type.value
    if vtype != 'same':
      args.extend(('valueType', vtype))
    return args

  # ---------------------------------------------------------------------------
  #
  def _replace_last_filter_map(self, args):
    vlast = self._last_filter_map
    if vlast and vlast.deleted:
      vlast = None
    last_color = None
    if vlast:
      args.extend(('model', '#' + vlast.id_string))
      last_color = vlast.overall_color
      self.session.models.close([vlast])
      self._last_filter_map = None
    return last_color
  
  # ---------------------------------------------------------------------------
  #
  def _update_levels(self, vsource, vfiltered):
    if self._adjust_threshold.enabled:
      if vsource.surfaces:
        surf = vsource.surfaces[0]
        from chimerax import surface
        evol = surface.surface_volume_and_area(surf)[0]
        level = vfiltered.surface_level_for_enclosed_volume(evol)
        vfiltered.set_parameters(surface_levels = [level])
    
  # ---------------------------------------------------------------------------
  #
  def _preserve_color(self, vfiltered, last_color):
    if last_color is not None:
      vfiltered.update_drawings()
      vfiltered.overall_color = last_color
    
  # ---------------------------------------------------------------------------
  #
  def _run_filter(self, args, v):
    last_color = self._replace_last_filter_map(args)

    cmd = ' '.join(args)
    log = not self._immediate_update.enabled

    from chimerax.core.commands import run
    vg = run(self.session, cmd, log = log)

    self._last_command = cmd
    self._last_filter_map = vg

    self._update_levels(v, vg)
    self._preserve_color(vg, last_color)

    return vg
    
  # ---------------------------------------------------------------------------
  #
  def _filter_sharpen(self):
    self._filter_gaussian(sharpen = True)
    
  # ---------------------------------------------------------------------------
  #
  def _filter_median(self):
    v = self._map_menu.value
    args = ['volume median', '#' + v.id_string]
    niter = self._median_iterations.value
    if niter != 1:
      args.extend(['iterations', '%d' % niter])
    args.extend(self._subregion_and_step_args(v))
    vg = self._run_filter(args, v)
    return vg
    
  # ---------------------------------------------------------------------------
  #
  def _filter_bin(self):
    v = self._map_menu.value
    args = ['volume bin', '#' + v.id_string]
    bsize = self._bin_size.value
    args.extend(['binSize', bsize])
    args.extend(self._subregion_and_step_args(v))
    vg = self._run_filter(args, v)
    return vg
    
  # ---------------------------------------------------------------------------
  #
  def _filter_scale(self):
    v = self._map_menu.value
    args = ['volume scale', '#' + v.id_string]
    shift, scale = self._shift.value, self._scale.value
    if shift != 0:
      args.extend(['shift', '%.5g' % shift])
    if scale != 0:
      args.extend(['factor', '%.5g' % scale])
    args.extend(self._value_type_args())
    args.extend(self._subregion_and_step_args(v))
    vg = self._run_filter(args, v)
    return vg
    
  # ---------------------------------------------------------------------------
  #
  def _filter_fourier(self):
    v = self._map_menu.value
    args = ['volume fourier', '#' + v.id_string]
    args.extend(self._subregion_and_step_args(v))
    vg = self._run_filter(args, v)
    return vg
  
  # ---------------------------------------------------------------------------
  #
  def _filter_laplacian(self):
    v = self._map_menu.value
    args = ['volume laplacian', '#' + v.id_string]
    args.extend(self._subregion_and_step_args(v))
    vg = self._run_filter(args, v)
    return vg
  
  # ---------------------------------------------------------------------------
  #
  def _filter_flatten(self):
    v = self._map_menu.value
    args = ['volume flatten', '#' + v.id_string]
    method = self._flatten_method.value
    meth = 'multiply' if method == 'multipy linear' else 'divide'
    args.extend(['method', meth])
    args.extend(self._subregion_and_step_args(v))
    vg = self._run_filter(args, v)
    return vg
    
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
  def _create_options_gui(self, parent):

    from chimerax.ui.widgets import CollapsiblePanel
    self._options_panel = p = CollapsiblePanel(parent, title = None)
    f = p.content_area

    from chimerax.ui.widgets import EntriesRow

    dro = EntriesRow(f, True, 'Use displayed subregion')
    self._use_displayed_subregion = dro.values[0]

    dso = EntriesRow(f, False, 'Use displayed step size')
    self._use_displayed_step = dso.values[0]

    ath = EntriesRow(f, True, 'Adjust threshold for constant volume')
    self._adjust_threshold = ath.values[0]

    iu = EntriesRow(f, False, 'Immediate update')
    self._immediate_update = iu.values[0]

    vt = EntriesRow(f, 'Value type', ('same', 'int8', 'uint8', 'int16', 'uint16',
                                      'int32', 'uint32', 'float32', 'float64'))
    self._value_type_frame = vt.frame
    self._value_type = vt.values[0]

    return p
    
  # ---------------------------------------------------------------------------
  #
  def _create_width_slider(self, parent):
    from chimerax.ui.widgets import LogSlider
    s = LogSlider(parent, label = 'Width', range = self._width_range,
                  value_change_cb = self._update_if_needed,
                  release_cb = self._log_last_command)
    return s

  # ---------------------------------------------------------------------------
  #
  def _update_if_needed(self, width, slider_down):
    if self._immediate_update.enabled:
      self._filter()
    
  # ---------------------------------------------------------------------------
  #
  def _log_last_command(self):
    if self._last_command:
      if self._immediate_update.enabled:
        from chimerax.core.commands import log_equivalent_command
        log_equivalent_command(self.session, self._last_command)
      self._last_command = None
    
  # ---------------------------------------------------------------------------
  #
  def _create_bfactor_slider(self, parent):
    from chimerax.ui.widgets import LogSlider
    s = LogSlider(parent, label = 'B-factor', range = self._bfactor_range,
                  value_change_cb = self._update_if_needed,
                  release_cb = self._log_last_command)
    return s

# -----------------------------------------------------------------------------
#
def map_filter_panel(session, create = False):
    return MapFilterPanel.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_map_filter_panel(session):
    return map_filter_panel(session, create = True)
