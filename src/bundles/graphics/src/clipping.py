# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

class ClipPlanes:
    '''
    Manage multiple clip planes and track when any change so that redrawing is done.
    '''
    def __init__(self):
        self._clip_planes = []		# List of ClipPlane
        self._changed = False

    def planes(self):
        return self._clip_planes

    def add_plane(self, p):
        self._clip_planes.append(p)
        self._changed = True

    def find_plane(self, name):
        np = [p for p in self._clip_planes if p.name == name]
        return np[0] if len(np) == 1 else None

    def replace_planes(self, planes):
        self._clip_planes = list(planes)
        self._changed = True

    def remove_plane(self, name):
        self._clip_planes = [p for p in self._clip_planes if p.name != name]
        self._changed = True

    def _get_changed(self):
        return self._changed or len([p for p in self._clip_planes if p._changed]) > 0
    def _set_changed(self, changed):
        self._changed  = changed
        for p in self._clip_planes:
            p._changed = changed
    changed = property(_get_changed, _set_changed)

    def have_camera_plane(self):
        for p in self._clip_planes:
            if isinstance(p, CameraClipPlane):
                return True
        return False

    def have_scene_plane(self):
        for p in self._clip_planes:
            if isinstance(p, SceneClipPlane):
                return True
        return False

    def clear(self):
        self._clip_planes = []
        self._changed = True

    def set_clip_position(self, name, point, view):
        p = self.find_plane(name)
        if p:
            p.plane_point = point
        elif name in ('near', 'far'):
            camera_normal = (0,0,(-1 if name == 'near' else 1))
            camera_point = view.camera.position.inverse() * point
            p = CameraClipPlane(name, camera_normal, camera_point, view)
            self.add_plane(p)
        else:
            normal = view.camera.view_direction()
            p = SceneClipPlane(name, normal, point)
            self.add_plane(p)

    def enable_clip_plane_graphics(self, render, camera_position):
        cp = self._clip_planes
        if cp:
            render.enable_capabilities |= render.SHADER_CLIP_PLANES
            planes = tuple(p.opengl_vec4() for p in cp)
            render.set_clip_parameters(planes)
        else:
            render.enable_capabilities &= ~render.SHADER_CLIP_PLANES

class ClipPlane:
    '''
    Clip plane that is either fixed in scene coordinates or camera coordinates (near/far planes).
    Subclasses must define normal and plane_point settable properties that use scene coordinates.
    Subclasses mus define a copy method.
    '''

    def __init__(self, name):
        self.name = name
        self._changed = False	# Used to know when graphics update needed.

    def offset(self, point):
        '''Return distance of a point to the plane (signed).'''
        from chimerax.geometry import inner_product
        return inner_product(self.plane_point - point, self.normal)

    def opengl_vec4(self):
        from chimerax.geometry import inner_product
        nx,ny,nz = n = self.normal
        c0 = inner_product(n, self.plane_point)
        return (nx, ny, nz, -c0)

class SceneClipPlane(ClipPlane):
    '''
    Clip plane that is fixed in scene coordinates.
    Defining normal vector and plane point are in scene coordinates.
    '''
    def __init__(self, name, normal, plane_point):
        ClipPlane.__init__(self, name)
        self._normal = normal		# Scene coordinates
        self._plane_point = plane_point	# Scene coordinates

    def _get_normal(self):
        return self._normal
    def _set_normal(self, normal):
        self._normal = normal
        self._changed = True
    normal = property(_get_normal, _set_normal)

    def _get_plane_point(self):
        return self._plane_point
    def _set_plane_point(self, plane_point):
        self._plane_point = plane_point
        self._changed = True
    plane_point = property(_get_plane_point, _set_plane_point)

    def copy(self):
        return SceneClipPlane(self.name, self.normal, self.plane_point)

class CameraClipPlane(ClipPlane):
    '''
    Clip plane that is fixed in camera coordinates.
    Plane is specified by normal (usually +/-z) and point in camera coordinates.
    '''
    def __init__(self, name, normal, plane_point, view):
        ClipPlane.__init__(self, name)
        self._camera_normal = normal		# Camera coordinates
        self._camera_plane_point = plane_point	# Camera coordinates
        self._view = view

    @property
    def _camera_position(self):
        return self._view.camera.position

    def _get_normal(self):
        return self._camera_position.transform_vector(self._camera_normal)
    def _set_normal(self, normal):
        self._camera_normal = self._camera_position.inverse().transform_vector(normal)
        self._changed = True
    normal = property(_get_normal, _set_normal)

    def _get_plane_point(self):
        return self._camera_position * self._camera_plane_point
    def _set_plane_point(self, plane_point):
        self._camera_plane_point = self._camera_position.inverse() * plane_point
        self._changed = True
    plane_point = property(_get_plane_point, _set_plane_point)

    def copy(self):
        return CameraClipPlane(self.name, self._camera_normal, self._camera_plane_point, self._view)
