# -----------------------------------------------------------------------------
# coordset command to play frames of a trajectory.
#
# Syntax: coordset <molecule-id>
#                  <start>[,<end>][,<step>]     # frame range
#                  [holdSteady <atomSpec>]
#
# Unspecified start or end defaults to current frame, last frame.
# Unspecified step is 1 or -1 depending on if end > start.
# Can use -1 for last frame.  Frame numbers start at 1.
#
def coordset(session, structures, index_range, hold_steady = None, loop = 1):
  '''
  Change which coordinate set is shown for a structure.  Can play through
  a range of coordinate sets.  

  Parameters
  ----------
  structures : list of AtomicStructure
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
  loop : integer
    How many times to repeat playing through the coordinates in the specified range.
  '''

  if len(structures) == 0:
    from ..errors import UserError
    raise UserError('No structures specified')

  for m in structures:
    s,e,step = absolute_index_range(index_range, m)
    hold = hold_steady.intersect(m.atoms) if hold_steady else None
    Coordinate_Set_Player(m, s, e, step, hold, loop).start()

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import CmdDesc, register, AtomicStructuresArg, ListOf, IntArg, AtomsArg
    desc = CmdDesc(
        required = [('structures', AtomicStructuresArg),
                    ('index_range', IndexRangeArg)],
        keyword = [('hold_steady', AtomsArg),
                   ('loop', IntArg)],
        synopsis = 'show coordinate sets')
    register('coordset', desc, coordset, logger=session.logger)

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
class Coordinate_Set_Player:

  def __init__(self, molecule, istart, iend, istep, steady_atoms = None, loop = 1):

    self.molecule = molecule
    self.istart = istart
    self.iend = iend
    self.istep = istep
    self.inext = None
    self.steady_atoms = steady_atoms
    self.loop = loop
    self._steady_coords = None
    self._steady_transforms = {}
    self._handler = None

  def start(self):

    self.inext = self.istart
    t = self.molecule.session.triggers
    self._handler = t.add_handler('new frame', self.frame_cb)

  def stop(self):

    if self._handler is None:
      return
    t = self.molecule.session.triggers
    t.remove_handler(self._handler)
    self._handler = None
    self.inext = None

  def frame_cb(self, tname, tdata):

    m = self.molecule
    if m.deleted:
      self.stop()
      return
    i = self.inext
    last_cs = m.active_coordset_id
    try:
      m.active_coordset_id = i
    except:
      # No such coordset.
      pass
    else:
      if self.steady_atoms:
        self.hold_steady(last_cs)
    self.inext += self.istep
    if ((self.istep > 0 and self.inext > self.iend) or
        (self.istep < 0 and self.inext < self.iend)):
      if self.loop <= 1:
        self.stop()
      else:
        self.inext = self.istart
        self.loop -= 1

  def hold_steady(self, last_cs):

    m = self.molecule
    tf = self.steady_transform(last_cs).inverse() * self.steady_transform(m.active_coordset_id)
    m.position = m.position * tf
    
  def steady_transform(self, cset):

    tfc = self._steady_transforms
    if cset in tfc:
      return tfc[cset]
    atoms = self.steady_atoms
    coords = coordset_coords(atoms, cset, self.molecule)
    if self._steady_coords is None:
      self._steady_coords = coords
    from ..geometry import align_points
    tf = align_points(coords, self._steady_coords)[0]
    tfc[cset] = tf
    return tf

def coordset_coords(atoms, cset, structure):
  cs = structure.active_coordset_id
  if cset == cs:
    xyz = atoms.coords
  else:
    structure.active_coordset_id = cset
    xyz = atoms.coords
    structure.active_coordset_id = cs
  return xyz
  
