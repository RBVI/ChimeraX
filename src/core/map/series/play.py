# vi: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
# Show a series of maps.
#
class Play_Series:

  def __init__(self, series = [], session = None, range = None, start_time = None, time_step_cb = None,
               play_direction = 'forward', loop = False, max_frame_rate = None,
               markers = None,
               preceding_marker_frames = 0, following_marker_frames = 0,
               color_range = None,
               normalize_thresholds = False,
               rendering_cache_size = 1):

    self.series = series
    self.session = session
    self.current_time = None
    self.time_range = r = (0, len(series[0].maps)-1, 1) if range is None else range
    self.start_time = (r[0] if play_direction == 'forward' else r[1]) if start_time is None else start_time
    self.time_step_cb = time_step_cb

    self.play_handler = None
    self.set_play_direction(play_direction) # 'forward', 'backward', 'oscillate'
    self.loop = loop
    self.max_frame_rate = max_frame_rate
    self.last_rendering_walltime = None

    self.markers = markers      # Marker molecule
    self.preceding_marker_frames = preceding_marker_frames
    self.following_marker_frames = following_marker_frames

    self.color_range = color_range

    self.normalize_thresholds = normalize_thresholds

    self.rendering_cache_size = rendering_cache_size
    self.rendered_times = []       # For limiting cached renderings
    self.rendered_times_table = {}

  # ---------------------------------------------------------------------------
  #
  def view(self):
    s = self.session
    if hasattr(s, 'main_view'):
      v = s.main_view        # Chimera 2
    else:
      v = s.view	     # Hydra
    return v

  # ---------------------------------------------------------------------------
  #
  def set_play_direction(self, direction):

    self.play_direction = direction
    self.step = {'forward':1, 'backward':-1, 'oscillate':1}[direction]
  
  # ---------------------------------------------------------------------------
  #
  def play(self):

    if self.play_handler is None:
      self.play_handler = h = self.next_time_cb
      self.view().add_new_frame_callback(h)
  
  # ---------------------------------------------------------------------------
  #
  def stop(self):

    h = self.play_handler
    if h:
      self.view().remove_new_frame_callback(h)
      self.play_handler = None

  # ---------------------------------------------------------------------------
  #
  def next_time_cb(self):

    if self.delay_next_frame():
      return
      
    t = self.current_time
    if t is None:
      if not self.start_time is None:
        self.change_time(self.start_time)
      return

    tslist = self.series
    if len(tslist) == 0:
      return

    ts, te = self.time_range[:2]
    nt = te-ts+1
    if self.play_direction == 'oscillate':
      if self.step > 0:
        if t == te:
          self.step = -1
      elif t == ts:
        self.step = 1

    tn = t + self.step
    if self.loop:
      tn = ts + (tn-ts)%nt
    elif (tn-ts) % nt != (tn-ts):
      self.stop()       # Reached the end or the beginning
      return

    self.change_time(tn)

  # ---------------------------------------------------------------------------
  #
  def delay_next_frame(self):

    if self.max_frame_rate is None:
      return False

    t0 = self.last_rendering_walltime
    import time
    t = time.time()
    if t0 != None:
      r = self.max_frame_rate
      if r != None and (t-t0)*r < 1:
        return True
      
    self.last_rendering_walltime = t
    return False

  # ---------------------------------------------------------------------------
  #
  def change_time(self, t):

    self.current_time = t
    if t is None:
      return

    tslist = self.series
    tslist = [ts for ts in tslist
              if t < ts.number_of_times() and not ts.is_volume_closed(t)]

    for ts in tslist:
      t0 = self.update_rendering_settings(ts, t)
      self.show_time(ts, t)
      if ts.last_shown_time != t:
        self.unshow_time(ts, ts.last_shown_time)
      if t0 != t:
        self.unshow_time(ts, t0)
      ts.last_shown_time = t

    if tslist:
      self.update_marker_display()
#      self.update_color_zone()

    if self.time_step_cb:
      self.time_step_cb(t)
    
  # ---------------------------------------------------------------------------
  # Update based on active volume viewer data set if it is part of series,
  # otherwise use previously shown time.
  #
  def update_rendering_settings(self, ts, t):

    t0 = ts.last_shown_time

    if t0 != t:
      ts.copy_display_parameters(t0, t, self.normalize_thresholds)
      
    return t0

  # ---------------------------------------------------------------------------
  #
  def show_time(self, ts, t):

    ts.show_time(t)
    self.cache_rendering(ts, t)

  # ---------------------------------------------------------------------------
  #
  def unshow_time(self, ts, t):

    cache_rendering = (self.rendering_cache_size > 1)
    ts.unshow_time(t, cache_rendering)
    if not cache_rendering:
      self.uncache_rendering(ts, t)

  # ---------------------------------------------------------------------------
  #
  def cache_rendering(self, ts, t):

    rtt = self.rendered_times_table
    if not (ts,t) in rtt:
      rtt[(ts,t)] = 1
      self.rendered_times.append((ts,t))
    self.trim_rendering_cache()

  # ---------------------------------------------------------------------------
  #
  def trim_rendering_cache(self):

    climit = self.rendering_cache_size
    rt = self.rendered_times
    rtt = self.rendered_times_table
    k = 0
    while len(rtt) > climit and k < len(rt):
      ts, t = rt[k]
      if ts.time_shown(t):
        k += 1
      else:
        ts.unshow_time(t, cache_rendering = False)
        del rtt[(ts,t)]
        del rt[k]
    
  # ---------------------------------------------------------------------------
  #
  def uncache_rendering(self, ts, t):

    rtt = self.rendered_times_table
    if (ts,t) in rtt:
      del rtt[(ts,t)]
      self.rendered_times.remove((ts,t))

  # ---------------------------------------------------------------------------
  #
  def marker_set(self):

    return None

    import VolumePath
    d = VolumePath.volume_path_dialog(create = False)
    if d is None:
      return None
    return d.active_marker_set

  # ---------------------------------------------------------------------------
  #
  def update_marker_display(self):

    m = self.markers
    if m is None:
      return

    fmin, fmax = self.marker_frame_range()
    if fmin is None or fmax is None:
      return

    m.display = True
    a = m.atom_subset(residue_range = (fmin, fmax))
    a.show_atoms(only_these = True)
        
  # ---------------------------------------------------------------------------
  #
  def current_markers_and_links(self):

    mset = self.marker_set()
    if mset == None:
      return [], []

    t = self.current_time
    if t is None:
      return [], []
    tstr = str(t)

    mlist = [m for m in mset.markers() if m.note_text == tstr]
    llist = [l for l in mset.links()
             if l.marker1.note_text == tstr and l.marker2.note_text == tstr]

    return mlist, llist
        
  # ---------------------------------------------------------------------------
  #
  def marker_frame_range(self):

    t = self.current_time
    if t is None:
      return None, None
    fmin = t - self.preceding_marker_frames
    fmax = t + self.following_marker_frames
    return fmin, fmax
    
  # ---------------------------------------------------------------------------
  #
  def update_color_zone(self):

    t = self.current_time
    if t is None:
      return

    tslist = self.series
    tslist = [ts for ts in tslist if not ts.surface_model(t) is None]

    for ts in tslist:
      r = self.color_range
      if not r is None:
        mlist, llist = self.current_markers_and_links()
        if mlist or llist:
          atoms = [m.atom for m in mlist]
          bonds = [l.bond for l in llist]
          model = ts.surface_model(t)
          xform_to_surface = model.openState.xform.inverse()
          from ColorZone import points_and_colors, color_zone
          points, point_colors = points_and_colors(atoms, bonds,
                                                   xform_to_surface)
          if hasattr(model, 'series_zone_coloring'):
            zp, zpc, zr = model.series_zone_coloring
            from numpy import all
            if all(zp == points) and all(zpc == point_colors) and zr == r:
              return        # No change in coloring.
          model.series_zone_coloring = (points, point_colors, r)
          color_zone(model, points, point_colors, r, auto_update = True)
      else:
        for t in range(ts.number_of_times()):
          model = ts.surface_model(t)
          if model and hasattr(model, 'series_zone_coloring'):
            from ColorZone import uncolor_zone
            uncolor_zone(model)
            delattr(model, 'series_zone_coloring')

from ...ui import MouseMode
class Play_Series_Mouse_Mode(MouseMode):

  def __init__(self, play_series):
    MouseMode.__init__(self, play_series.session)
    self.play_series = play_series
    self.last_mouse_x = None

  def mouse_up(self, event):
    self.last_mouse_x = None

  def mouse_down(self, event):
    x,y = event.position()
    self.last_mouse_x = x
  
  def mouse_drag(self, event):

    x,y = event.position()
    if self.last_mouse_x is None:
      self.last_mouse_x = x
      return

    dx = x - self.last_mouse_x
    tstep = int(round(dx/50))
    if tstep == 0:
      return
    self.last_mouse_x = x

    p = self.play_series
    ser = p.series
    if len(ser) == 0:
      return 
    s0 = ser[0]
    t = s0.last_shown_time
    tn = t + tstep
    tmax = s0.number_of_times() - 1
    if tn > tmax:
      tn = tmax
    elif tn < 0:
      tn = 0
    if tn != t:
      p.change_time(tn)
    p.session.logger.status('%s time %d' % (s0.name, tn+1))

# -----------------------------------------------------------------------------
#
def label_value_in_range(text, imin, imax):

  try:
    i = int(text)
  except:
    return False
  return i >= imin and i <= imax
  
# -----------------------------------------------------------------------------
# TODO: If no series opened it should still work after one does get opened.
#
def map_series_mouse_mode(session):
  from . import Map_Series
  series = [m for m in session.models.list() if isinstance(m, Map_Series)]
  p = Play_Series(series, session, rendering_cache_size = 10) if series else None
  if p is None:
    m = None
  else:
    m = Play_Series_Mouse_Mode(p)
  return m
