# -----------------------------------------------------------------------------
#
def surface_area(varray, tarray):

    from ..map_cpp import surface_area
    area = surface_area(varray, tarray)
    return area

# -----------------------------------------------------------------------------
#
def enclosed_volume(varray, tarray):

    from ..map_cpp import enclosed_volume
    vol, hole_count = enclosed_volume(varray, tarray)
    if vol < 0:
        return None, hole_count
    return vol, hole_count
