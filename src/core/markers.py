# Mouse mode to place markers on surfaces
from .ui import MouseMode
class MarkerMouseMode(MouseMode):

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self.mode_name = 'place markers'
        self.bound_button = None

        self.center = False             # Place at centroid of surface

        self._marker_molecule = None
        self._next_marker_num = 1
        self._marker_chain_id = 'M'

    def marker_molecule(self):
        m = self._marker_molecule
        if m is None:
            from . import structure
            self._marker_molecule = m = structure.AtomicStructure('markers')
            self.session.models.add([m])
        return m

    def mouse_down(self, event):
        x,y = event.position()
        s = self.session
        v = s.main_view
        p = v.first_intercept(x,y)
        if p is None:
            c = None
        elif self.center and hasattr(p, 'triangle_pick'):
            c = connected_center(p.triangle_pick)
        else:
            c = p.position
        log = s.logger
        if c is None:
            log.status('No marker placed')
            return
        m = self.marker_molecule()
        a = m.new_atom('', 'H')
        a.coord = c
        a.radius = 3
        a.color = (255,255,0,255)
        r = m.new_residue('marker', self._marker_chain_id, self._next_marker_num)
        r.add_atom(a)
        self._next_marker_num += 1
        m.new_atoms()
        log.status('Placed marker')

    def mouse_drag(self, event):
        pass

    def mouse_up(self, event):
        pass

def connected_center(triangle_pick):
    d = triangle_pick.drawing()
    t = triangle_pick.triangle_number
    va, ta = d.vertices, d.triangles
    from . import surface
    ti = surface.connected_triangles(ta, t)
    tc = ta[ti,:]
    varea = surface.vertex_areas(va, tc)
    a = varea.sum()
    c = varea.dot(va)/a
    # TODO: Apply drawing transform to map to global coordinates
    return c
    
