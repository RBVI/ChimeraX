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
        self.update_callbacks = {}
        self._distances_shown = True

    def add_group(self, group, update_callback=None):
        self.monitored_groups.add(group)
        if update_callback:
            self.update_callbacks[group] = update_callback
        if group.num_pseudobonds > 0:
            self._update_distances(group.pseudobonds)

    def _get_distances_shown(self):
        return self._distances_shown

    def _set_distances_shown(self, shown):
        if shown == self._distances_shown:
            return
        self._distances_shown = shown
        self._update_distances()

    def remove_group(self, group):
        self.monitored_groups.discard(group)
        if group in self.update_callbacks:
            del self.update_callbacks[group]

    def _update_distances(self, pseudobonds=None, *args):
        if pseudobonds is None:
            pseudobonds = [pb for mg in self.monitored_groups for pb in mg.pseudobonds]
        by_group = {}
        for pb in pseudobonds:
            by_group.setdefault(pb.group, []).append(pb)

        from chimerax.label.label3d import labels_model, PseudobondLabel
        for grp, pbs in by_group.items():
            lm = labels_model(grp, create=True)
            if self.show_distances:
                from .settings import settings
                fmt = "%%.%df" % settings.precision
                if settings.show_units:
                    fmt += u'\u00C5'
                for pb in pbs:
                    lm.add_labels([pb], PseudobondLabel, self.session.main_view, None,
                        fmt % pb.length, grp.color.uint8x4(), None, None, None)
            else:
                lm.add_labels(pbs, PseudobondLabel, self.session.main_view, None, "",
                    grp.color.uint8x4(), None, None, None)
            if grp in self.update_callbacks:
                self.update_callbacks[group]()

    # session methods
    def reset_state(self, session):
        self.monitored_groups.clear()
        self.update_callbacks.clear()
        self._distances_shown = True

    @staticmethod
    def restore_snapshot(session, data):
        mon = session.pb_dist_monitor
        mon._ses_restore(data)
        return mon

    def take_snapshot(self, session, flags):
        return {
            'version': 1,

            'distances shown': self._distances_shown,
            'monitored groups': self.monitored_groups
        }

    def _ses_restore(self, data):
        for grp in list(self.monitored_groups)[:]:
            self.remove_group(grp)
        self._distances_shown = data['distances shown']
        for grp in data['monitored groups']:
            self.add_group(grp)
