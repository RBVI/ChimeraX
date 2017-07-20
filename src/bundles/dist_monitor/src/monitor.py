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
class DistancesMonitor(State):
    """Keep distances pseudobonds up to date"""

    def __init__(self, session, bundle_info):
        self.session = session
        self.monitored_groups = set()

    def add_group(self, group):
        #TODO: callback registration?  destroy when empty?

    def remove_group(self, group):
        self.monitored_groups.discard(group)

    # session methods
    def reset_state(self, session):
        pass

    @staticmethod
    def restore_snapshot(session, data):
        mon = session.pb_dist_monitor
        mon._ses_restore(data)
        return mon

    def take_snapshot(self, session, flags):
        return {
            'version': 1,

            'monitored groups': self.monitored_groups
        }

    def _ses_restore(self, data):
        for grp in list(self.monitored_groups)[:]:
            self.remove_group(grp)
        for grp in data['monitored groups']:
            self.add_group(grp)
