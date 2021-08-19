# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class SurfaceUpdaters(State):
    '''
    Keep track of the surface auto update routines so they can be saved in sessions.
    '''
    def __init__(self, updaters = []):
        from weakref import WeakSet
        self._updaters = WeakSet(updaters)

    def add(self, updater):
        '''
        An updater is a callable object taking no arguments that updates a surface.
        '''
        self._updaters.add(updater)
        
    def take_snapshot(self, session, flags):
        updaters = tuple(u for u in self._updaters if _updater_active(u))
        data = {'updaters': updaters,
                'version': 1}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        # Filter out updaters that could not be restored.
        # For example, a surface for a missing Volume might not be restored.
        updaters = [u for u in data['updaters'] if u is not None]
        return SurfaceUpdaters(updaters)

    def clear(self):
        self._updaters.clear()

# -----------------------------------------------------------------------------
#
def _updater_active(u):
    if hasattr(u, 'surface'):
        if u.surface is None or u.surface.deleted:
            return False
    if hasattr(u, 'closed') and u.closed():
        return False
    if hasattr(u, 'active') and not u.active():
        return False   # Some other coloring replaced this one.
    return True
        
# -----------------------------------------------------------------------------
#
def add_updater_for_session_saving(session, updater):
    '''
    An updater is a callable instance taking no arguments that updates a surface.
    It must inherit from State and have take_snapshot() and restore_snapshot()
    methods used by session saving.
    '''
    if not hasattr(session, '_surface_updaters'):
        session._surface_updaters = su = SurfaceUpdaters()
        session.add_state_manager('_surface_updaters', su)
    session._surface_updaters.add(updater)
