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

# -----------------------------------------------------------------------------
# Save and restore map series state.
#
def map_series_states(session):
    from .series import MapSeries
    s = [state_from_series(s, session) for s in session.model_list(type = MapSeries)]
    return s

# -----------------------------------------------------------------------------
#
def restore_map_series(ms_states, session, file_paths, attributes_only = False):
    for ms_state in ms_states:
        from chimerax.map.session import find_volumes_by_session_id
        maps = find_volumes_by_session_id(ms_state['maps'], session)
        from .series import MapSeries
        ms = MapSeries(ms_state['name'], maps, session)
        ms.id = ms_state['id']
        oids = session.object_ids
        if 'session_id' in ms_state:
            oids.set_object_id(ms, ms_state['session_id'])
        session.add_model(ms)

# ---------------------------------------------------------------------------
#
def state_from_series(series, session):
    oids = session.object_ids
    s = {
        'name': series.name,
        'id': series.id,
        'maps': [m.session_volume_id for m in series.maps],
        'session_id': oids.object_id(series),        # Used to reference series in gui session state
    }
    return s
