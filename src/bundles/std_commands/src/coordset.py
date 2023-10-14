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
# coordset command to play frames of a trajectory.
#
# Syntax: coordset <structure-id>
#                  <start>[,<end>][,<step>]     # frame range
#                  [holdSteady <atomSpec>]
#
# Unspecified start or end defaults to current frame, last frame.
# Unspecified step is 1 or -1 depending on if end > start.
# Can use -1 for last frame.  Frame numbers start at 1.
#
def coordset(session, structures, index_range, hold_steady = None,
             pause_frames = 1, loop = 1, bounce = False, compute_ss = False):
  '''Change which coordinate set is shown for a structure.
  Can play through a range of coordinate sets.

  Parameters
  ----------
  structures : list of Structure
    List of structures to show as assemblies.
  index_range : 3-tuple with integer or None elements, or just None
    Starting, ending and step coordinate set ids.  If starting id is None start with
    the currently shown coordinate set.  If ending id is None treat it as the last
    coordset.  If step id is None use step 1 if start < end else step -1.  Otherwise
    coordinate set is changed from start to end incrementing by step with one step
    taken per graphics frame.  Negative start / end ids are relative to the (one past)
    the last coordinate set, so -1 refers to the last coordinate set.
    If index_range is just None, then treated as 1,None,None.
  hold_steady : Atoms
    Collection of atoms to hold steady while changing coordinate set.
    The atomic structure is repositioned to minimize change in RMSD of these atoms.
  pause_frames : integer
    Stay at each coordset for this number of graphics frames.  This is to slow
    down playback.  Default 1.
  loop : integer
    How many times to repeat playing through the coordinates in the specified range.
  bounce : bool
    Whether to reverse direction instead of jumping to beginning when looping.  Default false.
  compute_ss : bool
    Whether to recompute secondary structure using dssp for every new frame.  Default false.
  '''

  if len(structures) == 0:
    from chimerax.core.errors import UserError
    raise UserError('No structures specified')

  if index_range is None:
    index_range = (1,None,None)
  immediate = (index_range[1] == index_range[0] and index_range[2] is None)
  for m in structures:
    s,e,step = absolute_index_range(index_range, m)
    hold = hold_steady.intersect(m.atoms) if hold_steady else None
    csp = CoordinateSetPlayer(m, s, e, step, hold, pause_frames, loop, bounce, compute_ss)
    csp.start()
    if immediate:
      # For just one frame execute immediately so scripts don't need wait command.
      csp.next_frame()

# -----------------------------------------------------------------------------
#
def coordset_stop(session, structures = None):
  '''Stop playing coordinate sets.'''

  if hasattr(session, '_coord_set_players'):
    for csp in tuple(session._coord_set_players):
      if structures is None or csp.structure in structures:
        csp.stop()

  if hasattr(session, '_coord_set_sliders'):
    for css in tuple(session._coord_set_sliders):
      if structures is None or css.structure in structures:
        css.stop()

# -----------------------------------------------------------------------------
#
def coordset_slider(session, structures, hold_steady = None,
                    pause_frames = 1, loop = 1, compute_ss = False,
                    movie_framerate = 25.0):
  '''Show a slider that controls which coordinate set is shown.

  Parameters
  ----------
  structures : List of Structure
    Make a slider for each structure specified.
  hold_steady : Atoms
    Collection of atoms to hold steady while changing coordinate set.
    The atomic structure is repositioned to minimize change in RMSD of these atoms.
  pause_frames : integer
    Stay at each coordset for this number of graphics frames when Play button used.
    This is to slow down playback.  Default 1.
  compute_ss : bool
    Whether to recompute secondary structure using dssp for every new frame.  Default false.
  movie_framerate : float
    The playback speed for a recorded movie. Default 25 frames/sec.
  '''

  if len(structures) == 0:
    from chimerax.core.errors import UserError
    raise UserError('No structures specified')

  for m in structures:
    hold = hold_steady.intersect(m.atoms) if hold_steady else None
    from .coordset_gui import CoordinateSetSlider
    CoordinateSetSlider(session, m, steady_atoms = hold,
                        pause_frames = pause_frames, compute_ss = compute_ss,
                        movie_framerate = movie_framerate)

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, ListOf
    from chimerax.core.commands import IntArg, BoolArg, FloatArg, Or, EmptyArg
    from chimerax.atomic import AtomsArg, StructuresArg
    desc = CmdDesc(
        required = [('structures', StructuresArg),
                    ('index_range', Or(IndexRangeArg,EmptyArg))],
        keyword = [('hold_steady', AtomsArg),
                   ('pause_frames', IntArg),
                   ('loop', IntArg),
                   ('bounce', BoolArg),
                   ('compute_ss', BoolArg)],
        synopsis = 'show coordinate sets')
    register('coordset', desc, coordset, logger=logger)

    desc = CmdDesc(
        optional = [('structures', StructuresArg)],
        synopsis = 'stop playback of coordinate sets')
    register('coordset stop', desc, coordset_stop, logger=logger)

    desc = CmdDesc(
        required = [('structures', StructuresArg)],
        keyword = [('hold_steady', AtomsArg),
                   ('pause_frames', IntArg),
                   ('compute_ss', BoolArg),
                   ('movie_framerate', FloatArg)],
        synopsis = 'show slider for coordinate sets')
    register('coordset slider', desc, coordset_slider, logger=logger)

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation
class IndexRangeArg(Annotation):
    """Start, end, step index range"""
    name = "index range"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import next_token, AnnotationError
        if not text:
            raise AnnotationError('Missing index range argument')
        token, text, rest = next_token(text)
        fields = token.split(',')
        if len(fields) > 3:
          raise AnnotationError("Index range has at most 3 comma-separated value")
        try:
            ses = [(None if f in ('', '.') else int(f)) for f in fields]
        except ValueError:
            raise AnnotationError("Index range values must be integers")
        if len(ses) == 1:
          ses.extend((ses[0],None))
        elif len(ses) == 2:
          ses.append(None)
        return ses, text, rest

# -----------------------------------------------------------------------------
#
def absolute_index_range(index_range, mol):

  # Find available coordsets
  ids = mol.coordset_ids
  imin, imax = min(ids), max(ids)

  s,e,st = index_range
  if s is None:
    si = mol.active_coordset_id
  elif s < 0:
    si = s + imax + 1
  else:
    si = s
  si = max(si, imin)
  si = min(si, imax)

  if e is None:
    ei = imax
  elif e < 0:
    ei = e + imax + 1
  else:
    ei = e
  ei = max(ei, imin)
  ei = min(ei, imax)

  if st is None:
    sti = 1 if ei >= si else -1
  else:
    sti = st

  return (si,ei,sti)

# -----------------------------------------------------------------------------
#
class CoordinateSetPlayer:

  def __init__(self, structure, istart, iend, istep,
               steady_atoms = None, pause_frames = 1, loop = 1, bounce = False,
               compute_ss = False):

    self.structure = structure
    # structure deletes its 'session' attr when the structure is deleted,
    # and we need it after deletion, so remember it separately
    self.session = session = structure.session
    self.istart = istart
    self.iend = iend
    self.istep = istep
    self.inext = None
    self.steady_atoms = steady_atoms
    self.pause_frames = pause_frames
    self.loop = loop
    self.bounce = bounce
    self._reverse = False   # Whether playing in opposite direction after bounce
    self.compute_ss = compute_ss
    self._pause_count = 0
    self._steady_coords = None
    self._steady_transforms = {}
    self._handler = None

  def start(self):

    self.inext = self.istart
    session = self.structure.session
    t = session.triggers
    self._handler = t.add_handler('new frame', self.frame_cb)
    if not hasattr(session, '_coord_set_players'):
      session._coord_set_players = set()
    session._coord_set_players.add(self)

  def stop(self):

    if self._handler is None:
      return
    self.session._coord_set_players.remove(self)
    t = self.session.triggers
    t.remove_handler(self._handler)
    self._handler = None
    self.inext = None

  def frame_cb(self, tname, tdata):

    self.next_frame()

  def next_frame(self):

    m = self.structure
    if m.deleted:
      self.stop()
      return
    pc = self._pause_count
    self._pause_count = (pc + 1) % self.pause_frames
    if pc > 0:
      return
    i = self.inext
    self.change_coordset(i)
    s,e,st = self.istart, self.iend, self.istep
    i += (-st if self._reverse else st)
    if (s <= e and s <= i and i <= e) or (s > e and e <= i and i <= s):
      self.inext = i
    else:
      # Reached the end of the range.  Loop or stop.
      if self.bounce:
        self._reverse = not self._reverse
      r = self._reverse
      self.inext = e if r else s
      if not r:
        self.loop -= 1
      if self.loop <= 0:
        self.stop()

  def change_coordset(self, cs):
    m = self.structure
    last_cs = m.active_coordset_id
    try:
      m.active_coordset_id = cs
      compute_ss = self.compute_ss
    except Exception:
      # No such coordset.
      compute_ss = False
    if compute_ss:
      from . import dssp
      dssp.compute_ss(m.session, m)
    else:
      if self.steady_atoms:
        self.hold_steady(last_cs)

  def hold_steady(self, last_cs):

    m = self.structure
    tf = self.steady_transform(last_cs).inverse() * self.steady_transform(m.active_coordset_id)
    m.position = m.position * tf

  def steady_transform(self, cset):

    tfc = self._steady_transforms
    if cset in tfc:
      return tfc[cset]
    atoms = self.steady_atoms
    coords = coordset_coords(atoms, cset, self.structure)
    if self._steady_coords is None:
      self._steady_coords = coords
    from chimerax.geometry import align_points
    tf = align_points(coords, self._steady_coords)[0]
    tfc[cset] = tf
    return tf

# -----------------------------------------------------------------------------
#
def coordset_coords(atoms, cset, structure):
  cs = structure.active_coordset_id
  if cset == cs:
    xyz = atoms.coords
  else:
    structure.active_coordset_id = cset
    xyz = atoms.coords
    structure.active_coordset_id = cs
  return xyz
