# Session save/restore of graphics state

from ..state import State, CORE_STATE_VERSION


class ViewState(State):

    def __init__(self, session, attribute):
        self.view_attr = attribute

    def take_snapshot(self, session, flags):
        v = getattr(session, self.view_attr)
        cs = CameraState(v.camera)
        data = [self.view_attr, v.center_of_rotation, v.window_size,
                v.background_color, cs.take_snapshot(session, flags)]
        return CORE_STATE_VERSION, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        self.view_attr = data[0]
        v = getattr(session, self.view_attr)
        (v.center_of_rotation, _,   # TODO: don't skip v.window_size
         v.background_color) = data[1:4]
        # Root drawing had redraw callback set to None.  Restore callback.
        v.drawing.set_redraw_callback(v._drawing_manager)
        from .camera import MonoCamera
        v.camera = MonoCamera()
        cam_version, cam_data = data[4]
        CameraState(v.camera).restore_snapshot_init(
            session, bundle_info, cam_version, cam_data)

    def reset_state(self, session):
        """Reset state to data-less state"""
        v = getattr(session, self.view_attr)
        v.center_of_rotation_method = 'front center'
        # v.window_size = ?
        v.background_color = (0, 0, 0, 1)


class CameraState(State):

    def __init__(self, camera):
        self.camera = camera

    def take_snapshot(self, session, flags):
        c = self.camera
        data = [c.position, c.field_of_view]
        return CORE_STATE_VERSION, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        c = self.camera
        (c.position, c.field_of_view) = data

    def reset_state(self, session):
        # delay implementing until needed
        raise NotImplemented()


class DrawingState(State):

    def __init__(self, drawing):
        self.drawing = drawing

    def take_snapshot(self, session, flags):
        d = self.drawing
        # all drawing objects should have the same version
        data = {
            'children': [c.take_snapshot(session, flags) for c in d.child_drawings()],
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
        return CORE_STATE_VERSION, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        d = self.drawing
        for child_version, child_data in data['children']:
            child = d.new_drawing()
            DrawingState(child).restore_snapshot(session, bundle_info, child_version, child_data)
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

    def reset_state(self, session):
        # delay implementing until needed
        raise NotImplemented()
