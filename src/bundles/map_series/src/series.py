# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# If MAPSERIES_STATE_VERSION changes, then bump the bundle's
# (maximum) session version number.
MAPSERIES_STATE_VERSION = 1

# -----------------------------------------------------------------------------
#
from chimerax.core.models import Model
class MapSeries(Model):

  def __init__(self, name, maps, session):

    Model.__init__(self, name, session)

    self.add(maps)
    if maps:
      self.show_first_map_only(maps)
    self.set_maps(maps)
    
    self.surface_level_ranks = []  # Cached for normalization calculation
    self.image_level_ranks = []    # Cached for normalization calculation

    self._timer = None		# Timer for updating volume viewer dialog

    h = session.triggers.add_handler('remove models', self._model_closed)
    self._close_handler = h

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
  def added_to_session(self, session):
    maps = self.maps
    if maps and maps[0].other_channels():
      return  # MultiChannelSeries will report info.
    msg = ('Opened map series %s as #%s, %d images'
           % (self.name, self.id_string, len(maps)))
    if maps:
      msg += ', ' + maps[0].info_string()
    session.logger.info(msg)

  # ---------------------------------------------------------------------------
  #
  def show_first_map_only(self, maps):
    v0 = maps[0]
    v0.display = True	# Show first map of series
    for v in maps[1:]:
      v.display = False
  
  # ---------------------------------------------------------------------------
  #
  def first_map(self):
    return self.maps[0] if self.maps else None
  
  # ---------------------------------------------------------------------------
  #
  def number_of_times(self):

    return len(self.maps)

  # ---------------------------------------------------------------------------
  #
  def grid_size(self):

    v = self.first_map()
    return v.data.size if v else (0,0,0)

  # ---------------------------------------------------------------------------
  #
  def _get_model_color(self):
    v = self.first_map()
    return v.model_color if v else None
  def _set_model_color(self, color):
    for m in self.maps:
      m.model_color = color
  model_color = property(_get_model_color, _set_model_color)

  # ---------------------------------------------------------------------------
  #
  def _model_closed(self, trigger_name, models):
    if self.deleted:
      self._close_handler.remove()
      self._close_handler = None
      return

    for m in models:
      if hasattr(m, 'series') and m.series is self:
        self.volume_closed(m)

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
    v.display = True

    self.last_shown_time = time

    self.update_volume_dialog()

  # ---------------------------------------------------------------------------
  #
  def unshow_time(self, time, cache_rendering = True):

    v = self.maps[time]
    if v is None:
      return

    self.shown_times.discard(time)

    v.display = False

    if not cache_rendering:
      v.remove_surfaces()
      v.close_image()

  # ---------------------------------------------------------------------------
  #
  def time_shown(self, time):

    v = self.maps[time]
    shown = (v and v.shown())
    return shown

  # ---------------------------------------------------------------------------
  #
  def volume_model(self, time):

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

    if self.deleted:
      return
    
    self._timer = None

    i = self.last_shown_time
    v = self.maps[i] if i < len(self.maps) else None
    if v is None:
      return

    from chimerax.map.volume_viewer import set_active_volume
    set_active_volume(self.session, v)

  # ---------------------------------------------------------------------------
  #
  def copy_display_parameters(self, t1, t2, normalize_thresholds = False):

    v1 = self.maps[t1]
    v2 = self.maps[t2]
    if v1 is None or v2 is None:
      return

    v2.data.set_step(v1.data.step)
    v2.data.set_origin(v1.data.origin)
    v2.copy_settings_from(v1, copy_xform = False, copy_region = False)
    if v1.is_full_region():
      # Handle case where some times have smaller map size than others.
      ijk_min, ijk_max, ijk_step = v2.full_region()
    else:
      ijk_min, ijk_max, ijk_step = v1.region
    v2.new_region(ijk_min, ijk_max, ijk_step)
    
    if normalize_thresholds:
      self.copy_threshold_rank_levels(v1, v2)

  # ---------------------------------------------------------------------------
  #
  def copy_threshold_rank_levels(self, v1, v2):

    levels, ranks = equivalent_rank_values(v1, [s.level for s in v1.surfaces],
                                           v2, [s.level for s in v2.surfaces],
                                           self.surface_level_ranks)
    for s, lev in zip(v2.surfaces, levels):
      s.level = lev
    self.surface_level_ranks = ranks

    lev1 = [l for l,b in v1.image_levels]
    lev2 = [l for l,b in v2.image_levels]
    levels, ranks = equivalent_rank_values(v1, lev1, v2, lev2,
                                           self.image_level_ranks)
    v2.image_levels = list(zip(levels, [b for lev,b in v1.image_levels]))
    self.image_level_ranks = ranks

  # State save/restore in ChimeraX
  def take_snapshot(self, session, flags):
    data = {'model state': Model.take_snapshot(self, session, flags),
            # Can't reference maps directly because it creates cyclic dependency.
            'map ids': [m.id for m in self.maps if m is not None],
            'version': MAPSERIES_STATE_VERSION}
    return data

  @staticmethod
  def restore_snapshot(session, data):
    maps = []
    s = MapSeries('series', maps, session)
    Model.set_state_from_snapshot(s, session, data['model state'])

    # Parent models are always restored before child models.
    # Restore child map list after child maps are restored.
    def restore_maps(trigger_name, model_table, series = s, map_ids = data['map ids']):
      idm = {m.id : m for m in s.child_models()}
      maps = [idm[id] for id in map_ids if id in idm]
      series.set_maps(maps)
      from chimerax.core.triggerset import DEREGISTER
      return DEREGISTER
    from chimerax.core.models import RESTORED_MODEL_TABLE
    session.triggers.add_handler(RESTORED_MODEL_TABLE, restore_maps)
    
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
