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
        data['clip_plane_states'] = [ClipPlaneState(cp) for cp in v.clip_planes.planes()]
        data['version'] = self.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        # Restores session.main_view
        vs = ViewState(session.main_view)
        vs.set_state_from_snapshot(session, data)
        return vs

    def set_state_from_snapshot(self, session, data):
        v = self.view
        for k in self.save_attrs:
            if k in data and k != 'window_size' and k != 'version':
                setattr(v, k, data[k])

        # Root drawing had redraw callback set to None.  Restore callback.
        v.drawing.set_redraw_callback(v._drawing_manager)

        # Restore camera
        cs = data['camera_state']
        v.camera = cs.camera

        # Restore lighting
        ls = data['lighting_state']
        v.lighting = ls.lighting
        v.update_lighting = True

        # Restore clip planes
        v.clip_planes.replace_planes([cps.clip_plane for cps in data['clip_plane_states']])

        # Restore window size
        from ..commands.windowsize import window_size
        width, height = data['window_size']
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
            session.logger.info('"%s" camera settings not currently saved in sessions' % c.name())
            data = {'position': c.position}
        data['version'] = self.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from .camera import MonoCamera
        cs = CameraState(MonoCamera())
        cs.set_state_from_snapshot(session, data)
        return cs

    def set_state_from_snapshot(self, session, data):
        c = self.camera
        if data is not None:
            for k,v in data.items():
                if k != 'version':
                    setattr(c, k, v)

    def reset_state(self, session):
        pass


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
        data['version'] = self.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import Lighting
        ls = LightingState(Lighting())
        ls.set_state_from_snapshot(session, data)
        return ls

    def set_state_from_snapshot(self, session, data):
        l = self.lighting
        for k,v in data.items():
            if k != 'version':
                setattr(l, k, v)

    def reset_state(self, session):
        pass


class ClipPlaneState(State):

    version = 1
    save_attrs = [
        'name',
        'normal',
        'plane_point',
        'camera_normal',
    ]

    def __init__(self, clip_plane):
        self.clip_plane = clip_plane

    def take_snapshot(self, session, flags):
        cp = self.clip_plane
        data = {a:getattr(cp,a) for a in self.save_attrs}
        data['version'] = self.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import ClipPlane
        cp = ClipPlane(data['name'], data['normal'], data['plane_point'], data['camera_normal'])
        return ClipPlaneState(cp)

    def reset_state(self, session):
        pass


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
        data['version'] = self.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        ds = DrawingState(Drawing(''))
        ds.set_state_from_stanpshot(session, data)
        return ds

    def set_state_from_snapshot(self, session, data):
        d = self.drawing
        for k in self.save_attrs:
            if k in data:
                setattr(d, k, data[k])
        for child_state in data['children']:
            d.add_drawing(child_state.drawing)

    def reset_state(self, session):
        pass
