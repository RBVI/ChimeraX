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
#
from ...models import Model
class Map_Series(Model):

  def __init__(self, name, maps, session):

    Model.__init__(self, name, session)
    
    self.add(maps)
    self.show_first_map_only(maps)
    self.set_maps(maps)
    
    self.surface_level_ranks = []  # Cached for normalization calculation
    self.solid_level_ranks = []  # Cached for normalization calculation

    self._timer = None		# Timer for updating volume viewer dialog

  # ---------------------------------------------------------------------------
  #
  def set_maps(self, maps):
    self.maps = maps
    self.shown_times = t = set(i for i,m in enumerate(maps) if m.display)
    self.last_shown_time = tuple(t)[0] if len(t) > 0 else 0
    for m in maps:
      m.series = self

  # ---------------------------------------------------------------------------
  #
  def show_first_map_only(self, maps):
    v0 = maps[0]
    v0.initialize_thresholds()
    v0.show()	# Show first map of series
    for v in maps[1:]:
      v.display = False

  # ---------------------------------------------------------------------------
  #
  def number_of_times(self):

    return len(self.maps)

  # ---------------------------------------------------------------------------
  #
  def grid_size(self):

    return self.maps[0].data.size if self.maps else (0,0,0)

  # ---------------------------------------------------------------------------
  #
  def _get_single_color(self):
    return self.maps[0].single_color if self.maps else None
  def _set_single_color(self, color):
    for m in self.maps:
      m.single_color = color
  single_color = property(_get_single_color, _set_single_color)

  # ---------------------------------------------------------------------------
  #
  def volume_closed(self, v):

    t = self.maps.index(v)
    self.maps[t] = None

  # ---------------------------------------------------------------------------
  #
  def is_volume_closed(self, t):

    return self.maps[t] is None

  # ---------------------------------------------------------------------------
  #
  def show_time(self, time, only = True):

    if only:
      for t in tuple(self.shown_times):
        if t != time:
          self.unshow_time(t)

    v = self.maps[time]
    if v is None:
      return

    self.shown_times.add(time)
    v.show()

    self.last_shown_time = time

    self.update_volume_dialog()

  # ---------------------------------------------------------------------------
  #
  def unshow_time(self, time, cache_rendering = True):

    v = self.maps[time]
    if v is None:
      return

    self.shown_times.discard(time)

    v.show(show = False)

    if not cache_rendering:
      v.remove_surfaces()
      v.close_solid()

  # ---------------------------------------------------------------------------
  #
  def time_shown(self, time):

    v = self.maps[time]
    shown = (v and v.shown())
    return shown

  # ---------------------------------------------------------------------------
  #
  def surface_model(self, time):

    return self.maps[time]
      
  # ---------------------------------------------------------------------------
  #
  def update_volume_dialog(self, delay_seconds = 1):

    ui = self.session.ui
    if ui.is_gui:
      delay_msec = int(1000 * delay_seconds)
      t = self._timer
      if t is None:
        self._timer = t = ui.timer(delay_msec, self.show_time_in_volume_dialog)
      else:
        t.start(delay_msec)

  # ---------------------------------------------------------------------------
  #
  def show_time_in_volume_dialog(self):

    self._timer = None

    i = self.last_shown_time
    v = self.maps[i] if i < len(self.maps) else None
    if v is None:
      return

    from chimerax.volume_viewer.volumedialog import set_active_volume
    set_active_volume(self.session, v)

  # ---------------------------------------------------------------------------
  #
  def copy_display_parameters(self, t1, t2, normalize_thresholds = False):

    v1 = self.maps[t1]
    v2 = self.maps[t2]
    if v1 is None or v2 == None:
      return

    v2.data.set_step(v1.data.step)
    v2.data.set_origin(v1.data.origin)
    v2.copy_settings_from(v1, copy_xform = False)
    if normalize_thresholds:
      self.copy_threshold_rank_levels(v1, v2)

  # ---------------------------------------------------------------------------
  #
  def copy_threshold_rank_levels(self, v1, v2):

    levels, ranks = equivalent_rank_values(v1, v1.surface_levels,
                                           v2, v2.surface_levels,
                                           self.surface_level_ranks)
    v2.surface_levels = levels
    self.surface_level_ranks = ranks

    lev1 = [l for l,b in v1.solid_levels]
    lev2 = [l for l,b in v2.solid_levels]
    levels, ranks = equivalent_rank_values(v1, lev1, v2, lev2,
                                           self.solid_level_ranks)
    v2.solid_levels = list(zip(levels, [b for lev,b in v1.solid_levels]))
    self.solid_level_ranks = ranks

  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    from ...state import CORE_STATE_VERSION
    data = {'model state': Model.take_snapshot(self, session, flags),
            # Can't reference maps directly because it creates cyclic dependency.
            'map ids': [m.id for m in self.maps],
            'version': CORE_STATE_VERSION}
    return data

  @staticmethod
  def restore_snapshot(session, data):
    maps = []
    s = Map_Series('series', maps, session)
    Model.set_state_from_snapshot(s, session, data['model state'])

    # Parent models are always restored before child models.
    # Restore child map list after child maps are restored.
    def restore_maps(trigger_name, session, series = s, map_ids = data['map ids']):
      idm = {m.id : m for m in s.child_models()}
      maps = [idm[id] for id in map_ids if id in idm]
      series.set_maps(maps)
      from ...triggerset import DEREGISTER
      return DEREGISTER
    session.triggers.add_handler('end restore session', restore_maps)
    
    return s

  def reset_state(self):
    pass

# -----------------------------------------------------------------------------
# Avoid creep due to rank -> value and value -> rank not being strict inverses
# by using passed in ranks if they match given values.
#
def equivalent_rank_values(v1, values1, v2, values2, ranks):

  ms1 = v1.matrix_value_statistics()
  ms2 = v2.matrix_value_statistics()
  rlev = [ms1.rank_data_value(r) for r in ranks]
  if rlev != values1:
    ranks = [ms1.data_value_rank(lev) for lev in values1]
  if [ms2.data_value_rank(lev) for lev in values2] != ranks:
    values2 = [ms2.rank_data_value(r) for r in ranks]
  return values2, ranks
