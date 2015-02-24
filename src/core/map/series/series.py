# vi: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
#
from ...models import Model
class Map_Series(Model):

  def __init__(self, name, maps):

    Model.__init__(self, name)
    self.maps = maps

    for m in maps:
      self.add_model(m)

    self.shown_times = t = set(i for i,m in enumerate(maps) if m.display)
    self.last_shown_time = tuple(t)[0] if len(t) > 0 else 0

    self.surface_level_ranks = []  # Cached for normalization calculation
    self.solid_level_ranks = []  # Cached for normalization calculation

  # ---------------------------------------------------------------------------
  #
  def add_model(self, m):
    if hasattr(Model, 'add_model'):
      Model.add_model(self, m)          # Hydra
    else:
      # Chimera 2
      self.add_drawing(m)
      if m.id is None:
        m.id = len(self.child_drawings())

  # ---------------------------------------------------------------------------
  #
  def number_of_times(self):

    return len(self.maps)

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

  # ---------------------------------------------------------------------------
  #
  def unshow_time(self, time, cache_rendering = True):

    v = self.maps[time]
    if v is None:
      return

    self.shown_times.discard(time)

    if v.representation == 'solid' and cache_rendering:
      vs = v.solid_model()
      if vs:
        vs.display = False
    else:
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

  # ---------------------------------------------------------------------------
  #
  def first_intercept(self, mxyz1, mxyz2, exclude = None):
    fmin = pmin = None
    for m in self.maps:
      if m.display:
        f,p = m.first_intercept(mxyz1, mxyz2, exclude)
        if not f is None and (fmin is None or f < fmin):
          fmin,pmin = f,p
    return fmin,pmin

  # State save/restore in Chimera 2
  def take_snapshot(self, session, flags):
    pass
  def restore_snapshot(self, phase, session, version, data):
    pass
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
