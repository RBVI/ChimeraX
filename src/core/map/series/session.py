# -----------------------------------------------------------------------------
# Save and restore map series state.
#
def map_series_states(session):
    from .series import Map_Series
    s = [state_from_series(s, session) for s in session.model_list() if isinstance(s, Map_Series)]
    return s

# -----------------------------------------------------------------------------
#
def restore_map_series(ms_states, session, file_paths, attributes_only = False):
    for ms_state in ms_states:
        from ..session import find_volume_by_session_id
        maps = [find_volume_by_session_id(r,session) for r in ms_state['maps']]
        from .series import Map_Series
        ms = Map_Series(ms_state['name'], maps)
        ms.id = ms_state['id']
        session.add_model(ms)

# ---------------------------------------------------------------------------
#
def state_from_series(series, session):
    s = {
        'name': series.name,
        'id': series.id,
        'maps': [m.session_volume_id for m in series.maps],
    }
    return s
    
