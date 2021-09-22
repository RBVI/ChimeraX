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
# Dialog for fitting molecules or maps in maps.
#
from chimerax.core.tools import ToolInstance
class FitMapDialog(ToolInstance):

  help = 'help:user/tools/fitmap.html'

  def __init__(self, session, tool_name):

    self._requested_halt = False
    self._model_move_handler = None
    self._last_relative_position = None

    self.max_steps = 2000
    self.ijk_step_size_min = 0.01
    self.ijk_step_size_max = 0.5
    self._last_status_time = 0
    self._status_interval = 0.5    # seconds

    ToolInstance.__init__(self, session, tool_name)

    from chimerax.ui import MainToolWindow
    tw = MainToolWindow(self)
    self.tool_window = tw
    parent = tw.ui_area

    from chimerax.ui.widgets import vertical_layout
    layout = vertical_layout(parent, margins = (5,0,0,0))
        
    # Make menus to choose molecule and map for fitting
    mf = self._create_mol_map_menu(parent)
    layout.addWidget(mf)

    # Report correlation
    from chimerax.ui.widgets import EntriesRow
    cr = EntriesRow(parent, 'Correlation', 0.0, 'Average map value', 0.0,
                    ('Update', lambda *args, self=self: self._update_metric_values(log = True)))
    self._corr_label, self._ave_label = cl, al = cr.values
    cl.value = al.value = None	# Make fields blank

    # Fit, Undo, Redo buttons
    bf = self._create_action_buttons(parent)
    layout.addWidget(bf)

    # Options panel
    options = self._create_options_gui(parent)
    layout.addWidget(options)
  
    # Status line
    from Qt.QtWidgets import QLabel
    self._status_label = sl = QLabel(parent)
    layout.addWidget(sl)
        
    layout.addStretch(1)    # Extra space at end

    tw.manage(placement="side")

  # ---------------------------------------------------------------------------
  #
  def delete(self):
    
    mmh = self._model_move_handler
    if mmh:
      self.session.triggers.remove_handler(mmh)
      self._model_move_handler = None

    ToolInstance.delete(self)
    
  # ---------------------------------------------------------------------------
  #
  def _create_mol_map_menu(self, parent):

    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
        
    mf = QFrame(parent)
    mlayout = QHBoxLayout(mf)
    mlayout.setContentsMargins(0,0,0,0)
    mlayout.setSpacing(10)
        
    fl = QLabel('Fit', mf)
    mlayout.addWidget(fl)

    from chimerax.map import Volume
    from chimerax.atomic import Structure
    from chimerax.ui.widgets import ModelMenuButton
    self._object_menu = om = ModelMenuButton(self.session,
                                             class_filter = (Structure, Volume),
                                             special_items = ['selected atoms'])
    mlist = self.session.models.list(type = Structure)
    vlist = self.session.models.list(type = Volume)
    if mlist:
      om.value = mlist[0]
    elif vlist:
      om.value = vlist[0]
    om.value_changed.connect(self._object_chosen)
    mlayout.addWidget(om)

    iml = QLabel('in map', mf)
    mlayout.addWidget(iml)

    self._map_menu = mm = ModelMenuButton(self.session, class_filter = Volume)
    mlayout.addWidget(mm)
    if vlist:
      mm.value = vlist[0] if mlist else vlist[-1]
    mlayout.addStretch(1)    # Extra space at end
    
    return mf
        
  # ---------------------------------------------------------------------------
  #
  def _create_correlation_gui(self, parent):

    from Qt.QtWidgets import QFrame, QHBoxLayout, QLabel
        
    mf = QFrame(parent)
    mlayout = QHBoxLayout(mf)
    mlayout.setContentsMargins(0,0,0,0)
    mlayout.setSpacing(10)
        
    fl = QLabel('Fit', mf)
    mlayout.addWidget(fl)

    from chimerax.map import Volume
    from chimerax.atomic import Structure
    from chimerax.ui.widgets import ModelMenuButton
    self._object_menu = om = ModelMenuButton(self.session, class_filter = (Structure, Volume))
    om.value_changed.connect(self._object_chosen)
    mlayout.addWidget(om)

    iml = QLabel('in map', mf)
    mlayout.addWidget(iml)

    self._map_menu = mm = ModelMenuButton(self.session, class_filter = Volume)
    mlayout.addWidget(mm)
    
    mlayout.addStretch(1)    # Extra space at end
    
    return mf
    
  # ---------------------------------------------------------------------------
  #
  def _create_options_gui(self, parent):

    from chimerax.ui.widgets import CollapsiblePanel
    self._options_panel = p = CollapsiblePanel(parent, title = None)
    f = p.content_area

    from chimerax.ui.widgets import EntriesRow, radio_buttons
    
    rue = EntriesRow(f, False, 'Real-time correlation / average update')
    self._realtime_update = ru = rue.values[0]
    ru.changed.connect(self._realtime_changed)
    
    se = EntriesRow(f, False, 'Use map simulated from atoms, resolution', 5.0)
    self._simulate_map, self._simulate_resolution = (sm, sr) = se.values
    sm.changed.connect(self._simulate_map_changed)
    sr.return_pressed.connect(self._simulate_resolution_changed)
    
    atr = EntriesRow(f, True, 'Use only data above contour level from first map')
    self._above_threshold = atr.values[0]
    self._above_threshold_frame = atr.frame

    oer = EntriesRow(f, 'Optimize', True, 'overlap', False, 'correlation')
    self._opt_overlap, self._opt_correlation = oer.values
    radio_buttons(self._opt_overlap, self._opt_correlation)
    self._optimize_frame = oer.frame

    came = EntriesRow(f, False, 'Correlation calculated about mean data value')
    self._corr_about_mean = cam = came.values[0]
    cam.changed.connect(self._update_metric_values)
    self._corr_about_mean_frame = came.frame

    ar = EntriesRow(f, 'Allow', True, 'rotation', True, 'shift')
    self._allow_rotation, self._allow_shift = ar.values

    mwme = EntriesRow(f, True, 'Move whole molecules')
    self._move_whole_molecules = mwme.values[0]
    self._move_whole_molecules_frame = mwme.frame

    self._gray_out_options()
    
    return p

  # ---------------------------------------------------------------------------
  #
  def _show_or_hide_options(self):
    self._options_panel.toggle_panel_display()

  # ---------------------------------------------------------------------------
  #
  def _create_action_buttons(self, parent):
    from chimerax.ui.widgets import button_row
    f, buttons = button_row(parent,
                            [('Fit', self._fit),
                             ('Undo', self._undo),
                             ('Redo', self._redo),
                             ('Options', self._show_or_hide_options)],
                            spacing = 10,
                            button_list = True)
    self._undo_button, self._redo_button = buttons[1], buttons[2]

    self._activate_undo_redo()

    return f

  # ---------------------------------------------------------------------------
  #
  @classmethod
  def get_singleton(self, session, create=True):
    from chimerax.core import tools
    return tools.get_singleton(session, FitMapDialog, 'Fit in Map', create=create)

  # ---------------------------------------------------------------------------
  #
  def status(self, message, log = False):
    self._status_label.setText(message)
    if log:
      self.session.logger.info(message)

  # ---------------------------------------------------------------------------
  # Map chosen to fit into base map.
  #
  def _fit_map(self):

    m = self._object_menu.value
    from chimerax.map import Volume
    return m if isinstance(m, Volume) else None

  # ---------------------------------------------------------------------------
  # Atoms chosen in dialog for fitting.
  #
  def _fit_atoms(self):

    m = self._object_menu.value
    if m == 'selected atoms':
      from chimerax.atomic import selected_atoms
      return selected_atoms(self.session)

    from chimerax.atomic import Structure
    if isinstance(m, Structure):
      return m.atoms
    
    return None

  # ---------------------------------------------------------------------------
  #
  def _object_chosen(self):

    self._gray_out_options()

  # ---------------------------------------------------------------------------
  #
  def _gray_out_options(self):

    fitting_map = not self._fitting_atoms()
    self._above_threshold_frame.setEnabled(fitting_map)
    self._optimize_frame.setEnabled(fitting_map)
    self._corr_about_mean_frame.setEnabled(fitting_map)
    self._move_whole_molecules_frame.setEnabled(self._fit_map() is None)

  # ---------------------------------------------------------------------------
  #
  def _fitting_atoms(self):

    return not (self._simulate_map.value or self._fit_map())

  # ---------------------------------------------------------------------------
  #
  def _realtime_changed(self):

    h = self._model_move_handler
    if self._realtime_update.value:
      if h is None:
        from chimerax.core.models import MODEL_POSITION_CHANGED
        h = self.session.triggers.add_handler(MODEL_POSITION_CHANGED, self._model_moved)
        self._model_move_handler = h
    elif h:
      self.session.triggers.remove_handler(h)
      self._model_move_handler = None

  # ---------------------------------------------------------------------------
  #
  def _model_moved(self, trigger_name, model):

    xf = self._relative_position()
    if xf:
      lxf = self._last_relative_position
      if lxf is None or not xf.same(lxf):
          self._last_relative_position = xf
          self._update_metric_values()

  # ---------------------------------------------------------------------------
  #
  def _relative_position(self):

    fatoms = self._fit_atoms()
    fmap = self._fit_map()
    bmap = self._map_menu.value
    if (fatoms is None and fmap is None) or bmap is None:
      return None
    if fatoms is not None:
      if len(fatoms) == 0:
        return None
      xfo = fatoms[0].structure.scene_position      # Atom list case
    elif fmap is not None:
      xfo = fmap.scene_position

    xf = xfo.inverse() * bmap.scene_position
    return xf

  # ---------------------------------------------------------------------------
  #
  def _simulate_map_changed(self):

    self._gray_out_options()

  # ---------------------------------------------------------------------------
  #
  def _simulate_resolution_changed(self):

    self._simulated_map()        # Create map.

  # ---------------------------------------------------------------------------
  #
  def _simulated_map(self):

    if not self._simulate_map.value:
      return None

    res = self._simulate_resolution.value
    if res is None:
      self.status('No resolution specified for simulated map.')
      return None

    atoms = self._fit_atoms()
    if atoms is None or len(atoms) == 0:
      return None       # Not fitting atoms

    from .fitmap import simulated_map
    v = simulated_map(atoms, res, self.session)
    
    return v

  # ---------------------------------------------------------------------------
  #
  def _metric(self):

    if self._fitting_atoms():
      m = 'sum product'
    else:
      if self._opt_overlap.value:
        m = 'sum product'
      elif self._opt_correlation.value:
        m = 'correlation about mean' if self._corr_about_mean.value else 'correlation'
    return m
      
  # ---------------------------------------------------------------------------
  #
  def _fit(self):

    fatoms = self._fit_atoms()
    fmap = self._fit_map()
    bmap = self._map_menu.value
    if fatoms is None and fmap is None:
      self.status('Choose model or map to fit.')
      return
    if fatoms is not None and len(fatoms) == 0:
      self.status('No atoms selected for fitting.')
      return
    if bmap is None:
      self.status('Choose map to fit into.')
      return
    if fmap == bmap:
      self.status('Map to fit must be different from map being fit into.')
      return

    opt_r = self._allow_rotation.value
    opt_t = self._allow_shift.value
    if not opt_r and not opt_t:
      self.status('Enable rotation or translation.')
      return

    smap = self._simulated_map()
    if fatoms and smap is None:
      self._fit_atoms_in_map(fatoms, bmap, opt_r, opt_t)
    elif fmap:
      self._fit_map_in_map(fmap, bmap, opt_r, opt_t)
    elif smap:
      self._fit_map_in_map(smap, bmap, opt_r, opt_t, atoms = fatoms)
    if smap:
      self._show_average_map_value(fatoms, bmap, log=True)

    self._activate_undo_redo()

  # ---------------------------------------------------------------------------
  #
  def _fit_map_in_map(self, mmap, fmap, opt_r, opt_t, atoms = None):

    use_threshold = self._above_threshold.value
    metric = self._metric()
    from .fitmap import map_points_and_weights, motion_to_maximum
    try:
      points, point_weights = map_points_and_weights(mmap, use_threshold)
    except (MemoryError, ValueError):
      self.session.logger.error('Out of memory fitting %s in %s due to large size of first map.'
                                % (mmap.name, fmap.name))
      return

    if len(points) == 0:
      if use_threshold:
        self.status('No grid points above map threshold.')
      else:
        self.status('Map has no non-zero values.')
      return

    self._allow_halt(True)
    move_tf, stats = motion_to_maximum(points, point_weights, fmap,
                                       self.max_steps,
                                       self.ijk_step_size_min,
                                       self.ijk_step_size_max,
                                       opt_t, opt_r, metric,
                                       request_stop_cb = self._report_status)
    self._allow_halt(False)

    mwm = self._move_whole_molecules.value
    from . import move
    move.move_models_and_atoms(move_tf, [mmap], atoms, mwm, fmap)

    self._report_map_fit(mmap, fmap, stats)
    self._report_transformation_matrix(mmap, fmap)

  # ---------------------------------------------------------------------------
  #
  def _fit_atoms_in_map(self, atoms, fmap, opt_r, opt_t):

    self._allow_halt(True)

    mwm = self._move_whole_molecules.value
    from .fitmap import move_atoms_to_maximum
    stats = move_atoms_to_maximum(atoms, fmap,
                                  self.max_steps,
                                  self.ijk_step_size_min,
                                  self.ijk_step_size_max,
                                  optimize_translation = opt_t,
                                  optimize_rotation = opt_r,
                                  move_whole_molecules = mwm,
                                  request_stop_cb = self._report_status)
    self._allow_halt(False)
    
    if stats:
      self._report_atom_fit(atoms, fmap, stats)
      if mwm:
        for m in stats['molecules']:
          self._report_transformation_matrix(m, fmap)
    
  # ---------------------------------------------------------------------------
  # Report optimization statistics to reply log and status line.
  #
  def _report_map_fit(self, moved_map, fixed_map, stats):

    cor = stats['correlation']
    corm = stats['correlation about mean']
    self._report_correlation(cor, corm)
    self._report_average(None)

    from .fitmap import map_fit_message
    message = map_fit_message(moved_map, fixed_map, stats)
    self.session.logger.info(message)

  # ---------------------------------------------------------------------------
  #
  def _report_correlation(self, cor = None, corm = None):

    if cor is None or corm is None:
      s = None
    elif self._corr_about_mean.value:
      s = corm
    else:
      s = cor

    self._corr_label.value = s
    
  # ---------------------------------------------------------------------------
  # Report optimization statistics to reply log and status line.
  #
  def _report_atom_fit(self, atoms, volume, stats):

    from .fitmap import atom_fit_message
    message = atom_fit_message(atoms.unique_structures, volume, stats)
    self.session.logger.info(message)

    ave = stats['average map value']
    self._report_average(ave)
    self._report_correlation(None)

    aoc = stats['atoms outside contour']
    if not aoc is None:
      natom = stats['points']
      status_message = '%d of %d atoms outside contour' % (aoc, natom)
      self.status(status_message)

  # ---------------------------------------------------------------------------
  #
  def _report_average(self, ave):

    s = None if ave is None else ave
    self._ave_label.value = s

  # ---------------------------------------------------------------------------
  # Print transformation matrix that places model in map in reply log.
  #
  def _report_transformation_matrix(self, model, map):

    from .fitmap import transformation_matrix_message
    message = transformation_matrix_message(model, map)
    self.session.logger.info(message)
    
  # ---------------------------------------------------------------------------
  #
  def _report_status(self, status):

    from time import time
    t = time()
    if t - self._last_status_time < self._status_interval:
      return
    self._last_status_time = t
    self.status(status)
#    from chimera import update
#    update.processWidgetEvents(self.halt_button)
    return self._requested_halt

  # ---------------------------------------------------------------------------
  #
  def _allow_halt(self, allow):

    if allow:
      self._requested_halt = False
      state = 'normal'
    else:
      state = 'disabled'
#    self.halt_button['state'] = state
#    from chimera import update
#    update.processWidgetEvents(self.halt_button)
    
  # ---------------------------------------------------------------------------
  #
  def _update_metric_values(self, log = False):

    fatoms = self._fit_atoms()
    fmap = self._fit_map()
    bmap = self._map_menu.value
    if (fatoms is None and fmap is None) or bmap is None:
      self.status('Choose model and map.')
      return

    if fatoms:
      self._show_average_map_value(fatoms, bmap, log=log)
      v = self._simulated_map()
      if v:
        self._show_correlation(v, bmap, log=log)
      else:
        self._report_correlation(None)
    elif fmap:
      self._show_correlation(fmap, bmap, log=log)
      self._report_average(None)
    
  # ---------------------------------------------------------------------------
  # Report correlation between maps.
  #
  def _show_correlation(self, mmap, fmap, log=False):
      
    about_mean = self._corr_about_mean.value
    from .fitmap import map_overlap_and_correlation as oc
    olap, cor, corm = oc(mmap, fmap, self._above_threshold.value)
    self._report_correlation(cor, corm)

    if log:
      msg = 'Correlation = %.4g, Correlation about mean = %.4g, Overlap = %.4g\n' % (cor, corm, olap)
      self.session.logger.info(msg)
    
  # ---------------------------------------------------------------------------
  # Report average map value at selected atom positions.
  #
  def _show_average_map_value(self, atoms, fmap, log=False):

    if atoms is None or len(atoms) == 0:
        self.status('No atoms selected.')
        return
      
    from .fitmap import average_map_value_at_atom_positions, atoms_outside_contour
    amv, npts = average_map_value_at_atom_positions(atoms, fmap)
    aoc, clevel = atoms_outside_contour(atoms, fmap)

    self._report_average(amv)

    msg = 'Average map value = %.4g for %d atoms' % (amv, npts)
    if aoc is not None:
        msg += ', %d outside contour' % (aoc,)
    self.status(msg, log=log)
    
  # ---------------------------------------------------------------------------
  #
  def _undo(self):

    from .move import position_history
    position_history.undo()
    self._activate_undo_redo()
    self._update_metric_values()

  # ---------------------------------------------------------------------------
  #
  def _redo(self):

    from .move import position_history
    position_history.redo()
    self._activate_undo_redo()
    self._update_metric_values()

  # ---------------------------------------------------------------------------
  #
  def _activate_undo_redo(self):

    from .move import position_history as h
    self._undo_button.setEnabled(h.can_undo())
    self._redo_button.setEnabled(h.can_redo())

# -----------------------------------------------------------------------------
#
def fit_map_dialog(session, create = False):

  return FitMapDialog.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_fit_map_dialog(session):

  return fit_map_dialog(session, create = True)
