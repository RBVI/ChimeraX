# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2018 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.state import StateManager

"""
basic_actions.statemgr: Basic Action State Manager
==================================================
The main purpose of this class is to save and restore
named selections.
"""

DEFINE_NAMESEL = 'define named selection'
REMOVE_NAMESEL = 'remove named selection'


class BasicActions(StateManager):

    def __init__(self, session):
        import weakref
        self._session = weakref.ref(session)
        t = session.triggers
        t.add_trigger(DEFINE_NAMESEL)
        t.add_trigger(REMOVE_NAMESEL)
        self._named_sels = {}

    def take_snapshot(self, session, flags):
        from chimerax.core.objects import Objects
        spec = []
        frozen = []
        for name, value in self._named_sels.items():
            if isinstance(value, str):
                spec.append((name, value))
            elif isinstance(value, Objects):
                init_params = {
                    "atoms": value.atoms or None,
                    "bonds": value.bonds or None,
                    "pseudobonds": value.pseudobonds or None,
                    "models": list(value.models) or None,
                }
                frozen.append((name, init_params))
            else:
                session.logger.warning("cannot save named selection %r: "
                                       "unrecognized type" % name)
        data = {"spec":spec, "frozen":frozen}
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import cmd
        from chimerax.core.objects import Objects
        for name, value in data["spec"]:
            cmd.name(session, name, value)
        for name, value in data["frozen"]:
            cmd.name_frozen(session, name, Objects(**value))
        return session.basic_actions

    def reset_state(self, session):
        self._named_sels.clear()

    def define(self, name, value):
        self._named_sels[name] = value
        session = self._session()
        session.triggers.activate_trigger(DEFINE_NAMESEL, name)

    def remove(self, name):
        try:
            del self._named_sels[name]
        except KeyError:
            pass
        else:
            session = self._session()
            session.triggers.activate_trigger(REMOVE_NAMESEL, name)
