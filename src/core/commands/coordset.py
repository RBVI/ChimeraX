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
def coordset(session, molecules, index_range, hold_steady = None, loop = 1):
  '''
  Change which coordinate set is shown for a structure.  Can play through
  a range of coordinate sets.  

  Parameters
  ----------
  molecules : list of AtomicStructure
    List of molecules to show as assemblies.
  index_range : list of 1 to 3 integers
    Starting, ending and step coordinate set ids.  If only one value is given that
    coordinate set becomes the active coordinate set.  If two values are given then
    the coordinate set is change from the start id to the end id incrementing or
    decrementing the id by 1 each frame.  If the third value is given the id is
    incremented by this value each frame.
  hold_steady : Atoms
    Collection of atoms to hold steady while changing coordinate set.
    The atomic structure is repositioned to minimize change in RMSD of these atoms.
  loop : integer
    How many times to repeat playing through the coordinates in the specified range.
  '''

  if len(molecules) == 0:
    from ..errors import UserError
    raise UserError('No molecules specified')

  for m in molecules:
    s,e,step = parse_index_range(index_range, m)
    hold = hold_steady.intersect(m.atoms) if hold_steady else None
    Coordinate_Set_Player(m, s, e, step, hold, loop).start()

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import CmdDesc, register, AtomicStructuresArg, ListOf, IntArg, AtomsArg
    desc = CmdDesc(
        required = [('molecules', AtomicStructuresArg),
                    ('index_range', ListOf(IntArg,1,3))],
        keyword = [('hold_steady', AtomsArg),
                   ('loop', IntArg)],
        synopsis = 'show coordinate sets')
    register('coordset', desc, coordset)

# -----------------------------------------------------------------------------
#
def parse_index_range(index_range, mol):

  irange = list(index_range)
  if len(irange) == 1:
    irange += [irange[0]]

  # Convert negative indices as relative to imax
  for a in (0,1):
    if irange[a] < 0:
      irange[a] += imax + 1

  # Clamp ranges to available coordsets.
  ids = mol.coordset_ids
  imin, imax = min(ids), max(ids)
  for a in (0,1):  
      irange[a] = min(max(irange[a], imin), imax)    # clamp to [imin,imax]

  if len(irange) == 2:
    step = 1 if irange[1] >= irange[0] else -1
    irange += [step]

  return irange

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
  
