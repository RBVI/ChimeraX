#
# Display surface normal lines showing motion from one map time to the next.
#
def show_prickles(drawing, length = 1, color = (255,255,255,255), surface = None, children = True):
    va, ta = drawing.geometry
    na = drawing.normals
    if not va is None and not na is None:
        n = len(va)
        from numpy import empty, float32, int32, arange
        van = empty((2*n,3), float32)
        van[:n,:] = va
        for a in range(3):
            van[n:,a] = va[:,a] + length*na[:,a]
        tan = empty((n,2), int32)
        tan[:,0] = arange(n)
        tan[:,1] = tan[:,0] + n
        d = surface.new_drawing('prickles')
        d.geometry = van, tan
        d.display_style = d.Mesh
        d.color = color
        d.use_lighting = False
    if children:
        # TODO: have to apply position matrix
        for d in drawing.child_drawings():
            show_prickles(d, length, color, surface)

def show_motion_lines(drawing, motion_to_map, scale = 1, color = (255,255,255,255), surface = None, steps = 10):

    # TODO: handle position transforms
    va = drawing.vertices
    na = drawing.normals
    if not va is None and not na is None:
        step_size = min(motion_to_map.data.step)
        level = min(motion_to_map.surface_levels)
        n = len(va)
        from numpy import ones, bool, zeros, float32, logical_and
        vinside = ones((n,), bool)
        vlen = zeros((n,), float32)
        for step in range(steps):
            s = step*step_size
            mval = motion_to_map.interpolated_values(va + s*na)
            logical_and(vinside, mval > level, vinside)
            vlen[vinside] = s
        vlen *= scale
        show_prickles(drawing, vlen, color, surface, children = False)
    for d in drawing.child_drawings():
        show_motion_lines(d, motion_to_map, scale, color, surface, steps)
