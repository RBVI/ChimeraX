# vi: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
#
def surface_area(varray, tarray):

    from ._surface import surface_area
    area = surface_area(varray, tarray)
    return area

# -----------------------------------------------------------------------------
#
def enclosed_volume(varray, tarray):

    from ._surface import enclosed_volume
    vol, hole_count = enclosed_volume(varray, tarray)
    if vol < 0:
        return None, hole_count
    return vol, hole_count

# -----------------------------------------------------------------------------
# Calculate volume enclosed by a surface and surface area.
#
def surface_volume_and_area(model):

# TODO: exclude surface caps and outline boxes.
    volume = holes = area = 0
    for d in model.all_drawings():
        varray, tarray = d.geometry
        if not varray is None:
            v, hc = enclosed_volume(varray, tarray)
            volume += 0 if v is None else v
            holes += hc
            area += surface_area(varray, tarray)
    return volume, area, holes
