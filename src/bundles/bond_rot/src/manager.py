# vim: set expandtab shiftwidth=4 softtabstop=4:

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

class BondRotationError(Exception):
    pass

from chimerax.core.state import StateManager
class BondRotationManager(StateManager):
    """Manager for bond rotations"""
    CREATED, MODIFIED, REVERSED, DELETED = trigger_names = ("created", "modified",
        "reversed", "deleted")

    def __init__(self, session, bundle_info):
        self.bond_rotations = {} # bond -> BondRotation
        self.bond_rotaters = {} # ident -> BondRotater
        self.session = session
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        for trig_name in self.trigger_names:
            self.triggers.add_trigger(trig_name)

    def delete_rotation(self, rotater):
        del self.bond_rotaters[rotater.ident]
        rotation = rotater.rotation
        rotation.rotaters.remove(rotater)
        if not rotation.rotaters:
            del self.bond_rotations[rotation.bond]
            if not self.bond_rotations:
                from chimerax.atomic import get_triggers
                get_triggers().remove_handler(self._handler_ID)
        if not rotater.one_shot:
            self.triggers.activate_trigger(self.DELETED, rotater)

    def delete_all_rotations(self):
        for rotater in self.bond_rotaters.values():
            self.triggers.activate_trigger(self.DELETED, rotater)
        if self.bond_rotations:
            from chimerax.atomic import get_triggers
            get_triggers().remove_handler(self._handler_ID)
            for rotation in self.bond_rotations.values():
                # break reference loops
                rotation.rotaters =  []
            self.bond_rotations.clear()
            self.bond_rotaters.clear()

    clear = delete_all_rotations

    def new_rotation(self, bond, ident=None, move_smaller_side=True, one_shot=True):
        """Create bond rotation for 'bond'

        Parameters
        ----------
        bond : Bond
            Bond to rotate
        ident: an integer or None
            Number used to refer to bond rotation in commands.  If None, automatically assign
            one.
        move_smaller_side: bool
            If True, move the "smaller" side (side with fewer atoms attached) when the
            bond rotation moves.  Otherwise move the bigger side.
        one_shot: bool
            True if the rotation is going to be used to change the torsion and then immediately
            deleted -- so don't bother firing triggers.

        Returns the created BondRotater
        """
        if ident is None:
            ident = 1
            while ident in self.bond_rotaters:
                ident += 1
        elif ident in self.bond_rotaters:
            raise BondRotationError("Bond rotation identifier %s already in use" % ident)

        try:
            moving_side = bond.smaller_side
        except ValueError:
            raise BondRotationError("Bond %s is part of a ring/cycle and cannot be rotated" % bond)
        if not move_smaller_side:
            moving_side = bond.other_atom(moving_side)

        if not self.bond_rotations:
            from chimerax.atomic import get_triggers
            self._handler_ID = get_triggers().add_handler('changes', self._changes_cb)

        if bond in self.bond_rotations:
            rotation = self.bond_rotations[bond]
        else:
            from .bond_rot import BondRotation
            rotation = BondRotation(self.session, bond)
        self.bond_rotations[bond] = rotation

        rotater = rotation.new_rotater(ident, moving_side, one_shot)
        self.bond_rotaters[ident] = rotater

        if not one_shot:
            self.triggers.activate_trigger(self.CREATED, rotater)
            self.session.logger.status("Bond rotation identifier is %s" % ident, log=True)
        return rotater

    def rotation_for_ident(self, ident):
        try:
            return self.bond_rotaters[ident]
        except KeyError:
            raise BondRotationError("No such bond rotation ident: %s" % ident)

    def _changes_cb(self, trig_name, changes):
        if changes.num_deleted_bonds() > 0:
            for br in list(self.bond_rotations.values()):
                if br.bond.deleted:
                    self._delete_rotation(br)
        new_bonds = changes.created_bonds(include_new_structures=False)
        if new_bonds:
            nb_structures = new_bonds.unique_structures
            for br in list(self.bond_rotations.values()):
                if br.bond.structure in nb_structures:
                    if br.bond.rings(cross_residue=True):
                        # new bond closed a cycle involving bond-rotation bond
                        self._delete_rotation(br)

        changed_structures = set()
        if 'active_coordset changed' in changes.structure_reasons():
            changed_structures.update(changes.modified_structures())
        if 'coordset changed' in changes.coordset_reasons():
            changed_structures.update(changes.modified_coordsets().unique_structures)
        if 'coord changed' in changes.atom_reasons():
            changed_structures.update(changes.modified_atoms().unique_structures)
        if changed_structures:
            for br in self.bond_rotaters.values():
                if not br.one_shot and br.rotation.bond.structure in changed_structures:
                    self.triggers.activate_trigger(self.MODIFIED, br)

    def _delete_rotation(self, br):
        # delete a BondRotation rather than a BondRotater
        for rotater in br.rotaters[:]:
            self.delete_rotation(rotater)

    # session methods
    def reset_state(self, session):
        self.clear()

    @staticmethod
    def restore_snapshot(session, data):
        mgr = session.bond_rotations
        mgr._ses_restore(data)
        return mgr

    def take_snapshot(self, session, flags):
        # viewer_info is "session independent"
        return {
            'version': 1,

            'rotations': self.bond_rotations,
            'rotaters': self.bond_rotaters,
        }

    def _ses_restore(self, data):
        self.clear()
        if 'bond_rots' in data:
            # old, non-backwards compatible, session data
            if data['bond_rots']:
                self.session.logger.warning('Bond-rotation data in session is obsolete and not restorable;'
                    " skipping")
            return
        self.bond_rotations = data['rotations']
        self.bond_rotaters = data['rotaters']
        if self.bond_rotations:
            from chimerax.atomic import get_triggers
            self._handler_ID = get_triggers().add_handler('changes', self._changes_cb)
