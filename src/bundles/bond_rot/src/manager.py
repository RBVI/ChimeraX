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

from chimerax.core.state import State
class BondRotationManager(State):
    """Manager for bond rotations"""
    CREATED, MODIFIED, REVERSED, DELETED = trigger_names = ("created", "modified",
        "reversed", "deleted")

    def __init__(self, session, bundle_info):
        self.bond_rots = {}
        self.session = session
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        for trig_name in self.trigger_names:
            self.triggers.add_trigger(trig_name)

    def delete_rotation(self, bond_rot):
        del self.bond_rots[bond_rot.ident]
        if not self.bond_rots:
            from chimera.core.atomic import get_triggers
            get_triggers(self.session).remove_handler(self._handler_ID)
        if not bond_rot.one_shot:
            self.triggers.activate_trigger(self.DELETED, bond_rot)

    def delete_all_rotations(self):
        for bond_rot in self.bond_rots.values():
            self.triggers.activate_trigger(self.DELETED, bond_rot)
        if self.bond_rots:
            from chimera.core.atomic import get_triggers
            get_triggers(self.session).remove_handler(self._handler_ID)
            self.bond_rots.clear()

    clear = delete_all_rotations

    def new_rotation(self, bond, ident=None, move_smaller_side=True, one_shot=True):
        """Create bond rotation for 'bond'

        Note that typically you would use the rotation_for_bond method rather than
        calling this directly, so that funny things don't happen if the rotation is
        already active.

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

        Returns the created BondRotation
        """
        if ident is None:
            ident = 1
            while ident in self.bond_rots:
                ident += 1
        elif ident in self.bond_rots:
            raise BondRotationError("Bond rotation identifier %s already in use" % ident)


        try:
            moving_side = bond.smaller_side
        except ValueError:
            raise BondRotationError("Bond %s is part of a ring/cycle and cannot be rotated" % bond)
        if not move_smaller_side:
            moving_side = bond.other_atom(moving_side)

        from .bond_rot import BondRotation
        bond_rot = BondRotation(self.session, bond, ident, moving_side, one_shot)

        if not self.bond_rots:
            from chimerax.core.atomic import get_triggers
            self._handler_ID = get_triggers(session).add_handler('changes', self._changes_cb)
        self.bond_rots[ident] = bond_rot
        if not one_shot:
            self.triggers.activate_trigger(self.CREATED, bond_rot)
            self.session.logger.status("Bond rotation identifier is %s" % ident, log=True)
        return bond_rot

    def rotation_for_bond(self, bond, create=True, **kw):
        """Used if willing to re-use an existing rotation, e.g. adjusting phi/psi res attrs"""
        for br in self.bond_rots.values():
            if br.bond == bond:
                return br
        if create:
            return self.new_rotation(bond, **kw)
        return None

    def rotation_for_ident(self, ident):
        try:
            return self.bond_rots[ident]
        except KeyError:
            raise BondRotationError("No such bond rotation ident: %s" % ident)

    def _changes_cb(self, trig_name, changes):
        if changes.num_deleted_bonds() > 0:
            for br in list(self.bond_rots.values()):
                if br.bond == bond:
                    self.delete_rotation(br)
        new_bonds = changes.created_bonds(include_new_structures=False)
        if new_bonds:
            nb_structures = new_bonds.unique_structures
            for br in list(self.bond_rots.values()):
                if br.structure in nb_structures:
                    if br.bond.rings(cross_residues=True):
                        # new bond closed a cycle involving bond-rotation bond
                        self.delete_rotation(br)

        changed_structures = set()
        if 'active_coordset changed' in changes.structure_reasons():
            changed_structures.update(changes.modified_structures())
        if 'coordset changed' in changes.coordset_reasons():
            changed_structures.update(changed_coordsets().unique_structures)
        if 'coord changed' in changes.atom_reasons():
            change_structures.update(changed_atoms().unique_structures)
        if changed_structures:
            for br in self.bond_rots.values():
                if not br.one_shot and br.bond.structure in changed_structures:
                    self.triggers.activate_trigger(self.MODIFIED, br)

    # session methods
    def reset_state(self, session):
        for bond_rot in self.bond_rots.values():
            bond_rot._destroy()
        self.bond_rots.clear()

    @staticmethod
    def restore_snapshot(session, data):
        mgr = session.bond_rotations
        mgr._ses_restore(data)
        return mgr

    SESSION_SAVE = True
    
    def take_snapshot(self, session, flags):
        # viewer_info is "session independent"
        return {
            'version': 1,

            'bond_rots': self.bond_rots,
        }

    def _ses_restore(self, data):
        for br in self.bond_rots.values():
            br._destroy()
        self.bond_rots = data['bond_rots']
