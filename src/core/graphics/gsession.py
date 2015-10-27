# Session save/restore of graphics state

from ..session import State

class ViewState(State):
    VIEW_STATE_VERSION = 1
    def __init__(self, view):
        self.view = view

    def take_snapshot(self, session, phase, flags):
        v = self.view
        cs = CameraState(v.camera)
        if phase == State.SAVE_PHASE:
            data = [v.center_of_rotation, v.window_size,
                    v.background_color,
                    cs.take_snapshot(session, phase, flags)]
            return [self.VIEW_STATE_VERSION, data]
        if phase == State.CLEANUP_PHASE:
            cs.take_snapshot(session, phase, flags)

    def restore_snapshot(self, phase, session, version, data):
        if version != self.VIEW_STATE_VERSION or len(data) == 0:
            from ..session import RestoreError
            raise RestoreError("Unexpected version or data")
        v = self.view
        if phase == State.CREATE_PHASE:
            (v.center_of_rotation, v.window_size,
             v.background_color) = data[:3]
            from .camera import MonoCamera
            v.camera = MonoCamera()
        CameraState(v.camera).restore_snapshot(phase, session, data[3][0], data[3][1])

    def reset_state(self):
        """Reset state to data-less state"""
        from numpy import array, float32
        v = self.view
        v.center_of_rotation_method = 'front center'
        # v.window_size = ?
        v.background_color = (0, 0, 0, 1)

class CameraState(State):
    CAMERA_STATE_VERSION = 1 
    def __init__(self, camera):
        self.camera = camera

    def take_snapshot(self, session, phase, flags): 
        if phase != State.SAVE_PHASE: 
            return 
        c = self.camera
        data = [c.position, c.field_of_view]
        return [self.CAMERA_STATE_VERSION, data] 

    def restore_snapshot(self, phase, session, version, data): 
        if version != self.CAMERA_STATE_VERSION or len(data) == 0: 
            from ..session import RestoreError 
            raise RestoreError("Unexpected version or data") 
        if phase != State.CREATE_PHASE: 
            return 
        c = self.camera
        (c.position, c.field_of_view) = data 

    def reset_state(self): 
        # delay implementing until needed 
        raise NotImplemented() 

class DrawingState(State):
    DRAWING_STATE_VERSION = 1
    def __init__(self, drawing):
        self.drawing = drawing

    def take_snapshot(self, session, phase, flags):
        d = self.drawing
        if phase == State.CLEANUP_PHASE:
            for c in d.child_drawings():
                DrawingState(c).take_snapshot(session, phase, flags)
            return
        if phase != State.SAVE_PHASE:
            return
        # all drawing objects should have the same version
        data = {
            'children': [c.take_snapshot(session, phase, flags)[1]
                         for c in d.child_drawings()],
            'name': d.name,
            'vertices': d.vertices,
            'triangles': d.triangles,
            'normals': d.normals,
            'vertex_colors': d.vertex_colors,
            'triangle_mask': d._triangle_mask,
            'edge_mask': d._edge_mask,
            'display_style': d.display_style,
            'texture': d.texture,
            'ambient_texture ': d.ambient_texture,
            'ambient_texture_transform': d.ambient_texture_transform,
            'use_lighting': d.use_lighting,

            'display': d.display,
            'display_positions': d.display_positions,
            'selected': d.selected,
            'selected_positions': d.selected_positions,
            'position': d.position,
            'positions': d.positions,
            'selected_triangles_mask': d.selected_triangles_mask,
            'color': d.color,
            'colors': d.colors,
            'geometry': d.geometry,
        }
        return self.DRAWING_STATE_VERSION, data

    def restore_snapshot(self, phase, session, version, data):
        if version != self.DRAWING_STATE_VERSION:
            from ..session import RestoreError
            raise RestoreError("Unexpected version or data")
        if phase != State.CREATE_PHASE:
            return
        d = self.drawing
        for child_data in data['children']:
            child = d.new_drawing()
            DrawingState(child).restore_snapshot(phase, session, version, child_data)
        d.name = data['name']
        d.vertices = data['vertices']
        d.triangles = data['triangles']
        d.normals = data['normals']
        d.vertex_colors = data['vertex_colors']
        d._triangle_mask = data['triangle_mask']
        d._edge_mask = data['edge_mask']
        d.display_style = data['display_style']
        d.texture = data['texture']
        d.ambient_texture = data['ambient_texture ']
        d.ambient_texture_transform = data['ambient_texture_transform']
        d.use_lighting = data['use_lighting']

        d.display = data['display']
        d.selected = data['selected']
        d.position = data['position']
        d.positions = data['positions']
        d.selected_triangles_mask = data['selected_triangles_mask']
        d.color = data['color']
        d.colors = data['colors']
        d.geometry = data['geometry']
        d.display_positions = data['display_positions']
        d.selected_positions = data['selected_positions']

    def reset_state(self):
        # delay implementing until needed
        raise NotImplemented()
