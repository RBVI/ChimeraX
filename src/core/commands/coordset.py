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
             pause = 1, loop = 1, compute_ss = False):
  '''
  Change which coordinate set is shown for a structure.  Can play through
  a range of coordinate sets.  

  Parameters
  ----------
  structures : list of Structure
    List of structures to show as assemblies.
  index_range : 3-tuple with integer or None elements
    Starting, ending and step coordinate set ids.  If starting id is None start with
    the currently shown coordinate set.  If ending id is None treat it as the last
    coordset.  If step id is None use step 1 if start < end else step -1.  Otherwise
    coordinate set is changed from start to end incrementing by step with one step
    taken per graphics frame.  Negative start / end ids are relative to the (one past)
    the last coordinate set, so -1 refers to the last coordinate set.
  hold_steady : Atoms
    Collection of atoms to hold steady while changing coordinate set.
    The atomic structure is repositioned to minimize change in RMSD of these atoms.
  pause : integer
     Stay at each coordset for this number of graphics frames.  This is to slow
     down playback.  Default 1.
  loop : integer
    How many times to repeat playing through the coordinates in the specified range.
  compute_ss : bool
    Whether to recompute secondary structure using dssp for every new frame.  Default false.
  '''

  if len(structures) == 0:
    from ..errors import UserError
    raise UserError('No structures specified')

  for m in structures:
    s,e,step = absolute_index_range(index_range, m)
    hold = hold_steady.intersect(m.atoms) if hold_steady else None
    CoordinateSetPlayer(m, s, e, step, hold, pause, loop, compute_ss).start()

# -----------------------------------------------------------------------------
#
def coordset_slider(session, structures, hold_steady = None,
                    pause = 1, loop = 1, compute_ss = False):
  '''
  Show a slider that controls which coordinate set is shown.

  Parameters
  ----------
  structures : List of Structure
    Make a slider for each structure specified.
  hold_steady : Atoms
    Collection of atoms to hold steady while changing coordinate set.
    The atomic structure is repositioned to minimize change in RMSD of these atoms.
  pause : integer
     Stay at each coordset for this number of graphics frames when Play button used.
     This is to slow down playback.  Default 1.
  compute_ss : bool
    Whether to recompute secondary structure using dssp for every new frame.  Default false.
  '''

  if len(structures) == 0:
    from ..errors import UserError
    raise UserError('No structures specified')

  for m in structures:
    hold = hold_steady.intersect(m.atoms) if hold_steady else None
    CoordinateSetSlider(session, m, steady_atoms = hold,
                        pause_frames = pause, compute_ss = compute_ss)

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import CmdDesc, register, StructuresArg, ListOf, IntArg, AtomsArg, BoolArg
    desc = CmdDesc(
        required = [('structures', StructuresArg),
                    ('index_range', IndexRangeArg)],
        keyword = [('hold_steady', AtomsArg),
                   ('pause', IntArg),
                   ('loop', IntArg),
                   ('compute_ss', BoolArg)],
        synopsis = 'show coordinate sets')
    register('coordset', desc, coordset, logger=session.logger)

    desc = CmdDesc(
        required = [('structures', StructuresArg)],
        keyword = [('hold_steady', AtomsArg),
                   ('pause', IntArg),
                   ('compute_ss', BoolArg)],
        synopsis = 'show slider for coordinate sets')
    register('coordset slider', desc, coordset_slider, logger=session.logger)

# -----------------------------------------------------------------------------
#
from . import Annotation
class IndexRangeArg(Annotation):
    """Start, end, step index range"""
    name = "index range"

    @staticmethod
    def parse(text, session):
        from . import next_token, AnnotationError
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
               steady_atoms = None, pause = 1, loop = 1, compute_ss = False):

    self.structure = structure
    self.istart = istart
    self.iend = iend
    self.istep = istep
    self.inext = None
    self.steady_atoms = steady_atoms
    self.pause = pause
    self.loop = loop
    self.compute_ss = compute_ss
    self._pause_count = 0
    self._steady_coords = None
    self._steady_transforms = {}
    self._handler = None

  def start(self):

    self.inext = self.istart
    t = self.structure.session.triggers
    self._handler = t.add_handler('new frame', self.frame_cb)

  def stop(self):

    if self._handler is None:
      return
    t = self.structure.session.triggers
    t.remove_handler(self._handler)
    self._handler = None
    self.inext = None

  def frame_cb(self, tname, tdata):

    m = self.structure
    if m.deleted:
      self.stop()
      return
    pc = self._pause_count
    self._pause_count = (pc + 1) % self.pause
    if pc > 0:
      return
    i = self.inext
    self.change_coordset(i)
    self.inext += self.istep
    if ((self.istep > 0 and self.inext > self.iend) or
        (self.istep < 0 and self.inext < self.iend)):
      if self.loop <= 1:
        self.stop()
      else:
        self.inext = self.istart
        self.loop -= 1

  def change_coordset(self, cs):
    m = self.structure
    last_cs = m.active_coordset_id
    try:
      m.active_coordset_id = cs
      compute_ss = self.compute_ss
    except:
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
    from ..geometry import align_points
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

# -----------------------------------------------------------------------------
#
from chimerax.core.ui.widgets.slider import Slider
class CoordinateSetSlider(Slider):

    SESSION_SKIP = True

    def __init__(self, session, structure, pause_frames = 1, movie_framerate = 25,
                 steady_atoms = None, compute_ss = False):

        self.structure = structure

        title = 'Coordinate sets %s (%d)' % (structure.name, structure.num_coord_sets)
        csids = structure.coordset_ids
        id_start, id_end = min(csids), max(csids)
        self.coordset_ids = set(csids)
        Slider.__init__(self, session, 'Model Series', 'Model', title, value_range = (id_start, id_end),
                        pause_frames = pause_frames, pause_when_recording = True,
                        movie_framerate = movie_framerate)

        self._player = CoordinateSetPlayer(structure, id_start, id_end, istep = 1, pause = pause_frames, loop = 1,
                                           compute_ss = compute_ss, steady_atoms = steady_atoms)
        self.update_value(structure.active_coordset_id)

        from chimerax.core.models import REMOVE_MODELS
        self._model_close_handler = session.triggers.add_handler(REMOVE_MODELS, self.models_closed_cb)

    def change_value(self, i, playing = False):
      self._player.change_coordset(i)

    def valid_value(self, i):
        return i in self.coordset_ids
            
    def models_closed_cb(self, name, models):
      if self.structure in models:
        self.delete()

    # Override ToolInstance method
    def delete(self):
        t = self.session.triggers
        t.remove_handler(self._model_close_handler)
        self._model_close_handler = None
        super().delete()
        self.structure = None
