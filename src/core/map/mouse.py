def mouse_contour_level(mousemodes, event):

    v = mousemodes.view
    if getattr(mousemodes, 'last_contour_frame', None) == v.frame_number:
        return # Handle only one recontour event per frame
    mousemodes.last_contour_frame = v.frame_number

    dx, dy = mousemodes.mouse_motion(event)
    f = -0.001*dy

    for m in mouse_maps(mousemodes.session.models):
        adjust_threshold_level(m, f)
        m.show()

def mouse_maps(models):    
    mall = models.list()
    from .volume import Volume
    mdisp = [m for m in mall if isinstance(m,Volume) and m.display]
    msel = [m for m in mdisp if m.any_part_selected()]
    maps = msel if msel else mdisp
    return maps
    
def wheel_contour_level(mousemodes, event):
    d = mousemodes.wheel_value(event)
    f = d/30
    for m in mouse_maps(mousemodes.session.models):
        adjust_threshold_level(m, f)
        m.show()

def adjust_threshold_level(m, f):
    ms = m.matrix_value_statistics()
    step = f * (ms.maximum - ms.minimum)
    if m.representation == 'solid':
        new_levels = [(l+step,b) for l,b in m.solid_levels]
        l,b = new_levels[-1]
        new_levels[-1] = (max(l,1.01*ms.maximum),b)
        m.set_parameters(solid_levels = new_levels)
    else:
        new_levels = tuple(l+step for l in m.surface_levels)
        m.set_parameters(surface_levels = new_levels)
