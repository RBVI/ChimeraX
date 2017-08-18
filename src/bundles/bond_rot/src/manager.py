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

from chimerax.core.state import State
class BondRotationManager(State):
    """Manager for bond rotations"""
    CREATED, MODIFIED, REVERSED, DELETED = trigger_names = ("created", "modified",
        "reversed", "deleted")

    #TODO: react to changes and fire triggers if not oneshot; decide if we handle incoming
    # trigger andloop through bond rotations and then fire our trigger; or if bond rotations
    # individually react to incoming trigger, manager accumulates non-oneshot changes, and
    # fires trigger when incoming trigger done
    def __init__(self, session, bundle_info):
        self.bond_rots = {}
        self.session = session
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        for trig_name in self.trigger_names:
            self.triggers.add_trigger(trig_name)

    def delete_bond_rot(self, bond_rot):
        del self.bond_rots[bond_rot.ident]
        bond_rot._destroy()

    def new_bond_rot(self, bond, ident=None, one_shot=True):
        """Create bond rotation for 'bond'

        Parameters
        ----------
        bond : Bond
            Bond to rotate
        ident: an integer or None
            Number used to refer to bond rotation in commands.  If None, automatically assign
            one.
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
            raise ValueError("Bond rotation identifier %s already in use" % ident)

        from .bond_rot import BondRotation, BondRotError
        try:
            bond_rot = BondRotation(self.session, bond, ident, one_shot)
        except BondRotError as e:
            raise UserError(str(e))
        self.bond_rots[ident] = bond_rot
        self.session.logger.info("Bond rotation identifier is %s" % ident)
        return bond_rot

    def rotation_for_bond(self, bond):
        """Used if willing to re-use an existing rotation, e.g. adjusting phi/psi res attrs"""
        for br in self.bond_rots.values():
            if br.bond == bond:
                return br
        return None

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
