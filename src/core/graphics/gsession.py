# Session save/restore of graphics state

from ..state import State


class ViewState(State):

    version = 1
    save_attrs = ['center_of_rotation', 'window_size', 'background_color',
                  'silhouettes', 'silhouette_thickness', 'silhouette_color',
                  'silhouette_depth_jump']

    def __init__(self, view):
        self.view = view

    def take_snapshot(self, session, flags):
        v = self.view
        data = {a:getattr(v,a) for a in self.save_attrs}
        data['camera_state'] = CameraState(v.camera)
        data['lighting_state'] = LightingState(v.lighting)
        return self.version, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        # Restores session.main_view
        self.view = v = session.main_view
        for k,value in data.items():
            setattr(v, k, value)

        # Root drawing had redraw callback set to None.  Restore callback.
        v.drawing.set_redraw_callback(v._drawing_manager)

        # Restore camera
        cs = data['camera_state']
        v.camera = cs.camera

        # Restore lighting
        ls = data['lighting_state']
        v.lighting = ls.lighting

        # Restore window size
        from ..commands.windowsize import window_size
        width, height = v.window_size
        window_size(session, width, height)

    def reset_state(self, session):
        pass


class CameraState(State):

    version = 1
    save_attrs = ['position', 'field_of_view']

    def __init__(self, camera):
        self.camera = camera

    def take_snapshot(self, session, flags):
        c = self.camera
        from .camera import MonoCamera
        if isinstance(c, MonoCamera):
            data = {a:getattr(c,a) for a in self.save_attrs}
        else:
            # TODO: Restore other camera modes.
            session.logger.info('"%s" camera not currently saved in sessions' % c.name())
            data = None
            
        return self.version, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        from .camera import MonoCamera
        self.camera = c = MonoCamera()
        if data is not None:
            for k,v in data.items():
                setattr(c, k, v)

    def reset_state(self, session):
        raise NotImplemented()


class LightingState(State):

    version = 1
    save_attrs = [
        'key_light_direction',
        'key_light_color',
        'key_light_intensity',
        'fill_light_direction',
        'fill_light_color',
        'fill_light_intensity',
        'ambient_light_color',
        'ambient_light_intensity',
        'depth_cue_start',
        'depth_cue_end',
        'depth_cue_color',
        'move_lights_with_camera',
        'shadows',
        'shadow_map_size',
        'shadow_depth_bias',
        'multishadow',
        'multishadow_map_size',
        'multishadow_depth_bias',
        ]
    def __init__(self, lighting):
        self.lighting = lighting

    def take_snapshot(self, session, flags):
        l = self.lighting
        data = {a:getattr(l,a) for a in self.save_attrs}
        return self.version, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        from . import Lighting
        self.lighting = l = Lighting()
        for k,v in data.items():
            setattr(l, k, v)

    def reset_state(self, session):
        raise NotImplemented()


class DrawingState(State):

    version = 1
    save_attrs = ['name', 'vertices', 'triangles', 'normals', 'vertex_colors', 
                  'triangle_mask', 'edge_mask', 'display_style', 'texture', 
                  'ambient_texture', 'ambient_texture_transform', 
                  'use_lighting', 'positions', 'display_positions', 
                  'selected_positions', 'selected_triangles_mask', 'colors']

    def __init__(self, drawing):
        self.drawing = drawing

    def take_snapshot(self, session, flags):
        d = self.drawing
        data = {a:getattr(d,a) for a in self.save_attrs}
        data['children'] = [DrawingState(c) for c in d.child_drawings()]
        return self.version, data

    def restore_snapshot_init(self, session, bundle_info, version, data):
        if not hasattr(self, 'drawing'):
            from . import Drawing
            self.drawing = Drawing('')
        d = self.drawing
        for k,v in data.items():
            setattr(d, k, v)
        for child_state in data['children']:
            d.add_drawing(child_state.drawing)


    def reset_state(self, session):
        raise NotImplemented()
