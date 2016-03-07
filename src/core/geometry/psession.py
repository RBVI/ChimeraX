# Session save/restore of place state

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

    @staticmethod
    def reset_state(place, session):
        pass

class PlacesState:
    version = 1

    @staticmethod
    def take_snapshot(places, session, flags):
        sas = places.shift_and_scale_array()
        data = {'shift_and_scale': sas} if sas else {'array': places.array()}
        data['version'] = PlacesState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import Places
        if 'shift_and_scale' in data:
            p = Places(shift_and_scale = data['shift_and_scale'])
        else:
            p = Places(place_array = data['array'])
        return p

    @staticmethod
    def reset_state(places, session):
        pass
