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

class BondRotationError(Exception):
    pass

from chimerax.core.state import StateManager
class BondRotationManager(StateManager):
    """Manager for bond rotations"""
    CREATED, MODIFIED, REVERSED, DELETED = trigger_names = ("created", "modified",
        "reversed", "deleted")
    # so you don't have to know how to import it...
    BondRotationError = BondRotationError

    def __init__(self, session, bundle_info):
        self.bond_rotations = {} # bond -> BondRotation
        self.session = session
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        for trig_name in self.trigger_names:
            self.triggers.add_trigger(trig_name)

    def delete_rotation(self, rotater):
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
        for br in self.bond_rotations.values():
            for rotater in br.rotaters:
                self.triggers.activate_trigger(self.DELETED, rotater)
            br.rotaters = []
        if self.bond_rotations:
            from chimerax.atomic import get_triggers
            get_triggers().remove_handler(self._handler_ID)
            self.bond_rotations.clear()

    clear = delete_all_rotations

    def new_rotation(self, bond, move_smaller_side=True, one_shot=True):
        """Create bond rotation for 'bond'

        Parameters
        ----------
        bond : Bond
            Bond to rotate
        move_smaller_side: bool
            If True, move the "smaller" side (side with fewer atoms attached) when the
            bond rotation moves.  Otherwise move the bigger side.
        one_shot: bool
            True if the rotation is going to be used to change the torsion and then immediately
            deleted -- so don't bother firing triggers.

        Returns the created BondRotater
        """
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

        rotater = rotation.new_rotater(moving_side, one_shot)

        if not one_shot:
            self.triggers.activate_trigger(self.CREATED, rotater)
        return rotater

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
            for bond, rotation in self.bond_rotations.items():
                for rotater in rotation.rotaters:
                    if not rotater.one_shot and bond.structure in changed_structures:
                        self.triggers.activate_trigger(self.MODIFIED, rotater)

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
            'version': 2,

            'rotations': self.bond_rotations,
            # need to save rotaters so that they get restored;
            # the rotation doesn't save them to avoid circular dependency
            'rotaters': [rotater for rotation in self.bond_rotations.values()
                            for rotater in rotation.rotaters]
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
        if self.bond_rotations:
            from chimerax.atomic import get_triggers
            self._handler_ID = get_triggers().add_handler('changes', self._changes_cb)
