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

from chimerax.core.state import StateManager
class DistancesMonitor(StateManager):
    """Keep distances pseudobonds up to date"""

    def __init__(self, session):
        self.session = session
        self.monitored_groups = set()
        self.update_callbacks = {}
        self._distances_shown = True
        from chimerax.atomic import get_triggers
        triggers = get_triggers()
        triggers.add_handler("changes", self._changes_handler)
        self._already_restored = set()

    def add_group(self, group, update_callback=None, session_restore=False):
        self.monitored_groups.add(group)
        if update_callback:
            self.update_callbacks[group] = update_callback
        if group.num_pseudobonds > 0 and not session_restore:
            self._update_distances(group.pseudobonds)
        if session_restore:
            # there will be a "check for changes" after the session restore,
            # so remember these so they can be skipped then
            self._already_restored.update([pb for pb in group.pseudobonds])
        else:
            self._already_restored.clear()

    def _get_decimal_places(self):
        from .settings import settings
        return settings.decimal_places

    def _set_decimal_places(self, places):
        from .settings import settings
        if places == settings.decimal_places:
            return
        settings.decimal_places = places
        self._update_distances()

    decimal_places = property(_get_decimal_places, _set_decimal_places)

    @property
    def distance_format(self, *, decimal_places=None, show_units=None):
        from .settings import settings
        if decimal_places is None:
            decimal_places = settings.decimal_places
        if show_units is None:
            show_units = settings.show_units
        fmt = "%%.%df" % decimal_places
        if show_units:
            fmt += u'\u00C5'
        return fmt

    def _get_distances_shown(self):
        return self._distances_shown

    def _set_distances_shown(self, shown):
        if shown == self._distances_shown:
            return
        self._distances_shown = shown
        self._update_distances()

    distances_shown = property(_get_distances_shown, _set_distances_shown)

    def remove_group(self, group):
        self.monitored_groups.discard(group)
        if group in self.update_callbacks:
            # can't check if there were pseudobonds since group may be already deleted
            self.update_callbacks[group]()
            del self.update_callbacks[group]

    def _get_show_units(self):
        from .settings import settings
        return settings.show_units

    def _set_show_units(self, show):
        from .settings import settings
        if show == settings.show_units:
            return
        settings.show_units = show
        self._update_distances()

    show_units = property(_get_show_units, _set_show_units)

    def _changes_handler(self, _, changes):
        if changes.num_deleted_pseudobond_groups() > 0:
            for mg in list(self.monitored_groups):
                if mg.deleted:
                    self.remove_group(mg)
        existing_pbs = {}
        for pb in changes.created_pseudobonds():
            if pb in self._already_restored:
                continue
            pbg = pb.group
            if pbg in self.monitored_groups:
                if pbg.group_type == pbg.GROUP_TYPE_COORD_SET and pbg.structure \
                and pbg.structure.num_coordsets > 1:
                    if pbg not in existing_pbs:
                        existing_pbs[pbg] = set(pbg.pseudobonds)
                    if pb not in existing_pbs[pbg]:
                        # prevent showing labels on pseudobonds in non-current coordinate sets
                        continue
                self._update_distances(pseudobonds=[pb])
        self._already_restored.clear()
        if "position changed" in changes.structure_reasons() \
        or len(changes.modified_coordsets()) > 0:
            self._update_distances()
        if "active_coordset changed" in changes.structure_reasons():
            self._update_distances(coordset_changed=True)

    def _update_distances(self, pseudobonds=None, *, coordset_changed=False):
        if pseudobonds is None:
            pseudobonds = [pb for mg in self.monitored_groups for pb in mg.pseudobonds]
            set_color = coordset_changed
        else:
            set_color = True
        by_group = {}
        for pb in pseudobonds:
            by_group.setdefault(pb.group, []).append(pb)

        from chimerax.label.label3d import labels_model, PseudobondLabel
        for grp, pbs in by_group.items():
            if coordset_changed:
                lm = labels_model(grp, create=False)
                if lm:
                    lm.delete()
            lm = labels_model(grp, create=True)
            label_settings = { 'color': grp.color } if set_color else {}
            if self.distances_shown:
                fmt = self.distance_format
                for pb in pbs:
                    label_settings['text'] = fmt % pb.length
                    lm.add_labels([pb], PseudobondLabel, self.session.main_view,
                        settings=label_settings)
            else:
                label_settings['text'] = ""
                lm.add_labels(pbs, PseudobondLabel, self.session.main_view,
                    settings=label_settings)
            if grp in self.update_callbacks:
                self.update_callbacks[grp]()

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
            'version': 3,

            'distances shown': self._distances_shown,
            'monitored groups': self.monitored_groups,
        }

    def _ses_restore(self, data):
        self._already_restored.clear()
        for grp in list(self.monitored_groups)[:]:
            self.remove_group(grp)
        self._distances_shown = data['distances shown']
        for grp in data['monitored groups']:
            self.add_group(grp, session_restore=True)
