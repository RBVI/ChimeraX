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

# Session save/restore of place state

def register_place_session_save(session):
    from .place import Place, Places
    methods = {
        Place: PlaceState,
        Places: PlacesState,
    }
    session.register_snapshot_methods(methods)

class PlaceState:
    version = 1

    @staticmethod
    def take_snapshot(place, session, flags):
        data = {'matrix': place.matrix,
                '_is_identity': place._is_identity,
                'version': PlaceState.version,
                }
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import Place
        p = Place(data['matrix'])
        p._is_identity = data['_is_identity']
        return p

class PlacesState:
    version = 1

    @staticmethod
    def take_snapshot(places, session, flags):
        sas = places.shift_and_scale_array()
        data = {'shift_and_scale': sas} if sas is not None else {'array': places.array()}
        data['version'] = PlacesState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import Places
        if 'shift_and_scale' in data:
            p = Places(shift_and_scale = data['shift_and_scale'])
        else:
            pa = data['array']
            from numpy import float32, float64
            if pa.dtype == float32:
                # Fix old sessions that saved array as float32
                pa = pa.astype(float64)
            p = Places(place_array = pa)
        return p
