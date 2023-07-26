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
# Remember model and atom positions relative to a given model.
#
class Position_History:

    def __init__(self):

        self.cur_position = -1
        self.positions = []
        self.max_positions = 100

    def record_position(self, models, atoms, base_model):

        plist = self.positions
        c = self.cur_position
        if c < len(plist)-1:
            del plist[c+1:]      # Erase redo states.
        ps = position_state(models, atoms, base_model)
        self.positions.append(ps)
        if len(plist) > self.max_positions:
            del plist[0]
        else:
            self.cur_position += 1

    def undo(self):

        while self.can_undo():
            ps = self.positions[self.cur_position]
            self.cur_position -= 1
            if restore_position(ps):
                return True
        return False

    def can_undo(self):
        return self.cur_position >= 0

    def redo(self):

        while self.can_redo():
            self.cur_position += 1
            ps = self.positions[self.cur_position]
            if restore_position(ps):
                return True
        return False

    def can_redo(self):
        return self.cur_position+1 < len(self.positions)

# -----------------------------------------------------------------------------
#
def position_state(models, atoms, base_model):

  btfinv = base_model.position.inverse()
  mset = set(models)
  mset.update(tuple(atoms.unique_structures))
  model_transforms = []
  for m in mset:
      model_transforms.append((m, btfinv * m.position))
  atom_positions = (atoms, atoms.coords.copy())
  return (base_model, model_transforms, atom_positions)

# -----------------------------------------------------------------------------
#
def restore_position(pstate, angle_tolerance = 1e-5, shift_tolerance = 1e-5):

    base_model, model_transforms, atom_positions = pstate
    if base_model.was_deleted:
        return False
    changed = False
    for m, mtf in model_transforms:
        if m.was_deleted:
            continue
        tf = base_model.position * mtf
        if not tf.same(m.position, angle_tolerance, shift_tolerance):
            changed = True
        m.position = tf

    atoms, xyz = atom_positions
    from numpy import array_equal
    if not array_equal(atoms.coords, xyz):
        if len(atoms) == len(xyz):
            changed = True
            atoms.coords = xyz
        else:
            # Atoms have been deleted, cannot undo positions.
            pass

    return changed

# -----------------------------------------------------------------------------
#
def move_models_and_atoms(tf, models, atoms, move_whole_molecules, base_model):

    if move_whole_molecules and atoms is not None and len(atoms) > 0:
        models = list(models) + list(atoms.unique_structures)
        from chimerax.atomic import Atoms
        atoms = Atoms()
    if atoms is None:
        from chimerax.atomic import Atoms
        atoms = Atoms()
    global position_history
    position_history.record_position(models, atoms, base_model)
    for m in models:
        # TODO: Handle case where parent of volume has non-identity position.  tf is in scene coords.
        m.scene_position = tf * m.scene_position
    if len(atoms) > 0:
        atoms.scene_coords = tf * atoms.scene_coords
    position_history.record_position(models, atoms, base_model)

# -----------------------------------------------------------------------------
#
position_history = Position_History()
