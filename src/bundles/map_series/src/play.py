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

# -----------------------------------------------------------------------------
# Show a series of maps.
#
class Play_Series:

  def __init__(self, series = [], session = None, range = None, start_time = None, time_step_cb = None,
               play_direction = 'forward', loop = False, max_frame_rate = None, pause_frames = 0,
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
    self.pause_frames = pause_frames
    self._pause_count = 0
    self.last_rendering_walltime = None

    self.markers = markers      # Marker molecule
    self.preceding_marker_frames = preceding_marker_frames
    self.following_marker_frames = following_marker_frames

    self.color_range = color_range

    self.normalize_thresholds = normalize_thresholds

    self.rendering_cache_size = max(rendering_cache_size, len(series))
    self.rendered_times = []       # For limiting cached renderings
    self.rendered_times_table = {}

    self._model_close_handler = session.triggers.add_handler('remove models', self._models_closed)

  # ---------------------------------------------------------------------------
  #
  def __del__(self):
    h = self._model_close_handler
    if h:
      self.session.triggers.remove_handler(h)

  # ---------------------------------------------------------------------------
  #
  def _models_closed(self, tname, models):
    mset = set(models)
    ser = self.series
    sopen = [s for s in ser if s not in mset]
    if len(sopen) < len(ser):
      self.series = sopen
      if len(sopen) == 0:
        self.stop()

  # ---------------------------------------------------------------------------
  #
  def view(self):

    return self.session.main_view

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
      self.handler = self.session.triggers.add_handler('new frame', h)
  
  # ---------------------------------------------------------------------------
  #
  def stop(self):

    h = self.play_handler
    if h:
      self.session.triggers.remove_handler(self.handler)
      self.play_handler = None

  # ---------------------------------------------------------------------------
  #
  def next_time_cb(self, *_):

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
    if nt == 0:
      return	# Series has no maps
    if self.play_direction == 'oscillate':
      if self.step > 0:
        if t == te:
          self.step = -1
      elif t == ts:
        self.step = 1
        if not self.loop:
          self.stop()
          return

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

    pf = self.pause_frames
    if pf > 0:
      pc = self._pause_count
      self._pause_count = pc+1
      if pc % pf != 0:
        return True

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
      self.update_color_zone()

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
  def update_marker_display(self):

    markers = self.markers
    if markers is None:
      return

    fmin, fmax = self.marker_frame_range()
    if fmin is None or fmax is None:
      return

    for m in markers:
      m.display = (hasattr(m, 'frame') and m.frame >= fmin and m.frame <= fmax)

      for l in markers.intra_bonds:
        a1,a2 = l.atoms
        f1, f2 = getattr(a1, 'frame', None), getattr(a2, 'frame', None)
        l.display = (f1 is not None and f1 >= fmin and f1 <= fmax and
                     f2 is not None and f2 >= fmin and f2 <= fmax)
        
  # ---------------------------------------------------------------------------
  #
  def current_markers_and_links(self):

    markers = self.markers
    if markers is None:
      return [], []

    t = self.current_time
    if t is None:
      return [], []

    mlist = [m for m in markers if hasattr(m, 'frame') and m.frame == t]
    llist = [l for l in markers.intra_bonds
             if getattr(l.atoms[0], 'frame', None) == t and getattr(l.atoms[1], 'frame', None) == t]

    from chimerax.atomic import Atoms, Bonds
    return Atoms(mlist), Bonds(llist)
        
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
    r = self.color_range
    if r is None:
      return

    t = self.current_time
    if t is None:
      return

    tslist = self.series
    tslist = [ts for ts in tslist if not ts.volume_model(t) is None]

    from chimerax.surface.colorzone import points_and_colors, color_zone
    for ts in tslist:
        mlist, llist = self.current_markers_and_links()
        if mlist or llist:
          v = ts.volume_model(t)
          points, point_colors = points_and_colors(mlist, llist)
          for surf in v.surfaces:
            color_zone(surf, points, point_colors, r, auto_update = True)

from chimerax.mouse_modes import MouseMode
class PlaySeriesMouseMode(MouseMode):

  name = 'play map series'
  icon_file = 'vseries.png'

  def __init__(self, session):
    MouseMode.__init__(self, session)
  
    self._series = None
    self._player = None
    self.last_mouse_x = None

  def play_series(self):
    from . import MapSeries
    series = tuple(self.session.models.list(type = MapSeries))
    if series != self._series:
      self._player = Play_Series(series, self.session, rendering_cache_size = 10) if series else None
    return self._player

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

    self._take_step(tstep)
    
  def wheel(self, event):
    d = event.wheel_value()
    self._take_step(-int(d))

  def _take_step(self, tstep):
    p = self.play_series()
    if p is None:
      return

    ser = p.series
    if len(ser) == 0:
      return 
    s0 = ser[0]
    t = s0.last_shown_time
    tn = t + tstep
    nt = s0.number_of_times()
    tmax = nt - 1
    if tn > tmax or tn < 0:
      tn = tn % nt
    if tn != t:
      p.change_time(tn)
    p.session.logger.status('%s time %d' % (s0.name, tn+1))

  def vr_motion(self, event):
    # Virtual reality hand controller motion.
    tstep = int(round(20*event.room_vertical_motion))
    if tstep == 0:
      return 'accumulate drag'
    self._take_step(tstep)

  def vr_thumbstick(self, event):
    # Virtual reality hand controller thumbstick tilt.
    step = event.thumbstick_step()
    if step != 0:
      self._take_step(step)
        

# -----------------------------------------------------------------------------
#
def label_value_in_range(text, imin, imax):

  try:
    i = int(text)
  except Exception:
    return False
  return i >= imin and i <= imax

# -----------------------------------------------------------------------------
#
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(PlaySeriesMouseMode(session))
