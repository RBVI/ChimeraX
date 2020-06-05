# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class SurfaceUpdaters(State):
    '''
    Keep track of the surface auto update routines so they can be saved in sessions.
    '''
    def __init__(self):
        from weakref import WeakSet
        self._updaters = WeakSet()

    def add(self, updater):
        '''
        An updater is a callable object taking no arguments that updates a surface.
        '''
        self._updaters.add(updater)
        
    def take_snapshot(self, session, flags):
        updaters = tuple(u for u in self._updaters
                         if hasattr(u, 'surface')
                         and u.surface is not None
                         and not u.surface.deleted)
        data = {'updaters': updaters,
                'version': 1}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        # Actual updaters are added when each is restored.
        return SurfaceUpdaters()

    def clear(self):
        self._updaters.clear()
        
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
