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
class Interpolated_Map:

  def __init__(self, volumes, scale_factors = None, adjust_thresholds = False,
               add_mode = False, interpolate_colors = True,
               subregion = 'all', step = 1, model_id = None):

    self.volumes = volumes
    v0 = volumes[0]
    self.session = session = v0.session

    r = None
    if model_id:
      vlist = session.models.list(model_id = model_id)
      if len(vlist) == 1:
        v = vlist[0]
        if (v.matrix_size(step = (1,1,1), subregion = 'all')
            == v0.matrix_size(step = step, subregion = subregion)):
          r = v
    if r is None:
      r = v0.writable_copy(require_copy = True, copy_colors = False,
                           unshow_original = False,
                           subregion = subregion, step = step,
                           model_id = model_id, name = 'morph')
    self.result = r
    self.subregion = subregion
    self.step = step

    if scale_factors is None:
      scale_factors = [1] * len(volumes)
    self.scale_factors = scale_factors
    self.add_mode = add_mode

    self.interpolate_colors = interpolate_colors

    self.f = self.fmin = self.fmax = self.fstep = 0
    self.steps = None
    self.f_changed_cb = None
    self.play_handler = None
    self.step_direction = 1     # 1 or -1, current direction when looping
    self.recording = False

    self.adjust_thresholds = adjust_thresholds
    self.surface_level_ranks = []       # For avoiding creep during threshold
    self.image_level_ranks = []         #   normalization.

  # ---------------------------------------------------------------------------
  #
  def interpolate(self, f):

    v = self.result
    if self.adjust_thresholds:
      self.record_threshold_ranks(v)

    f1, f2, sf1, sf2, v1, v2 = self.coefficients(f)
    if v.data is None or v1.data is None or v2.data is None:
      return False

    linear_combination(sf1, v1, sf2, v2, v, self.subregion, self.step)
    self.f = f

    if self.adjust_thresholds:
      self.set_threshold_ranks(v)

    if self.interpolate_colors:
      interpolate_colors(f1, v1, f2, v2, v)

    fccb = self.f_changed_cb
    if fccb:
      fccb(self.f)

    return True

  # ---------------------------------------------------------------------------
  #
  def coefficients(self, f):

    vlist = self.volumes
    n = len(vlist)
    i0 = min(n-2,max(0,int(f*(n-1))))
    f0 = float(i0)/(n-1)
    if self.add_mode:
      f1 = 1.0
      f2 = f
    else:
      f2 = (f-f0)*(n-1)
      f1 = 1-f2
    sf = self.scale_factors
    sf1 = f1 * sf[i0]
    sf2 = f2 * sf[i0+1]

    return f1, f2, sf1, sf2, vlist[i0], vlist[i0+1]
  
  # ---------------------------------------------------------------------------
  #
  def play(self, f, fmin, fmax, fstep, f_changed_cb, fdir = None, steps = None):

    self.f = f
    self.fmin = fmin
    self.fmax = fmax
    self.fstep = fstep
    self.f_changed_cb = f_changed_cb
    self.steps = steps
    if not fdir is None:
      if fdir >= 0: self.step_direction = 1
      else:         self.step_direction = -1
    if self.play_handler is None:
      h = self.session.triggers.add_handler('new frame', self.next_frame_cb)
      self.play_handler = h

  # ---------------------------------------------------------------------------
  #
  def stop_playing(self):

    if self.play_handler:
      self.session.triggers.remove_handler(self.play_handler)
      self.play_handler = None
    if self.recording:
      self.finish_recording()

  # ---------------------------------------------------------------------------
  #
  def playing(self):

    return self.play_handler != None

  # ---------------------------------------------------------------------------
  #
  def next_frame_cb(self, *_):

    if not self.steps is None:
      if self.steps <= 0:
        self.stop_playing()
        return
      else:
        self.steps -= 1

    fmin, fmax = self.fmin, self.fmax
    next_f = self.f + self.fstep * self.step_direction
    if next_f >= fmax:
      next_f = fmax
      self.step_direction = -1
    elif next_f <= fmin:
      next_f = fmin
      self.step_direction = 1

    if not self.interpolate(next_f):
      self.stop_playing()       # Volume closed.

  # ---------------------------------------------------------------------------
  #
  def record(self, fmin, fmax, fstep, f_changed_cb, roundtrip, record_args, save_movie_cb):

    if self.recording:
      return

    self.f = self.fmin = fmin
    self.fmax = fmax
    self.fstep = fstep
    self.f_changed_cb = f_changed_cb
    self.save_movie_cb = save_movie_cb
    
    self.recording = True

    from math import ceil
    steps = int(ceil((fmax - fmin) / fstep))
    if roundtrip:
      steps *= 2

    from chimerax.core.commands import run
    run(self.session, 'movie record ' + record_args)

    self.play(fmin, fmin, fmax, fstep, f_changed_cb, 1, steps)

  # ---------------------------------------------------------------------------
  #
  def finish_recording(self):

    self.recording = False

    if self.play_handler:
      self.stop_playing()

    from chimerax.core.commands import run
    run(self.session, 'movie stop')

    self.save_movie_cb()

  # ---------------------------------------------------------------------------
  #
  def record_threshold_ranks(self, v):

    ms = v.matrix_value_statistics()

    rlev = [ms.rank_data_value(r) for r in self.surface_level_ranks]
    slev = [s.level for s in v.surfaces]
    if slev != rlev:
      self.surface_level_ranks = [ms.data_value_rank(lev) for lev in slev]

    rlev = [ms.rank_data_value(r) for r in self.image_level_ranks]
    slev = [lev for lev,b in v.image_levels]
    if slev != rlev:
      self.image_level_ranks = [ms.data_value_rank(lev) for lev in slev]

  # ---------------------------------------------------------------------------
  #
  def set_threshold_ranks(self, v):

    ms = v.matrix_value_statistics()

    sflev = [ms.rank_data_value(r) for r in self.surface_level_ranks]
    imlev = list(zip([ms.rank_data_value(r) for r in self.image_level_ranks],
                     [b for lev,b in v.image_levels]))
    v.set_parameters(surface_levels = sflev, image_levels = imlev)

# -----------------------------------------------------------------------------
#
def linear_combination(f1, v1, f2, v2, v, subregion, step):
  
  m = v.full_matrix()
  m1 = v1.matrix(step = step, subregion = subregion)
  m2 = v2.matrix(step = step, subregion = subregion)
  if (m.flags.contiguous and m1.flags.contiguous and m2.flags.contiguous and
      m1.dtype == m.dtype and m2.dtype == m.dtype):
    # Optimize calculation of linear combination of matrices.
    # C++ routine is 7x faster (.1 vs .7 sec) than numpy on 256^3 matrix.
    from chimerax.map import linear_combination
    linear_combination(f1, m1, f2, m2, m)
  else:
    m[:,:,:] = f1*m1[:,:,:] + f2*m2[:,:,:]

  v.data.values_changed()

# -----------------------------------------------------------------------------
#
def interpolate_colors(f1, v1, f2, v2, v):

  nc = len(v.surfaces)
  if len(v1.surfaces) == nc and len(v2.surfaces) == nc:
    from chimerax.geometry import linear_combination
    for s, s1, s2 in zip(v.surfaces, v1.surfaces, v2.surfaces):
      s.rgba = linear_combination(f1, s1.rgba, f2, s2.rgba)

  nc = len(v.image_colors)
  if len(v1.image_colors) == nc and len(v2.image_colors) == nc:
    from chimerax.geometry import linear_combination
    scolors = [linear_combination(f1, v1.image_colors[c], f2, v2.image_colors[c])
               for c in range(nc)]
    v.set_parameters(image_colors = scolors)

# -----------------------------------------------------------------------------
#
def morph_maps(volumes, play_steps, play_start, play_step, play_direction,
               play_range, add_mode, adjust_thresholds, scale_factors,
               hide_maps, interpolate_colors, subregion, step, model_id):

  if hide_maps:
    for v in volumes:
      v.display = False

  im = Interpolated_Map(volumes, scale_factors, adjust_thresholds, add_mode,
                        interpolate_colors, subregion, step, model_id)
  if play_steps > 0:
    fmin, fmax = play_range
    im.play(play_start, fmin, fmax, play_step, None, play_direction, play_steps)
  else:
    im.interpolate(play_start)

  return im
