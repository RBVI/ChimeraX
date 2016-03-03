# Session save/restore of graphics state

class ViewState:

    version = 1
    save_attrs = ['center_of_rotation', 'window_size', 'background_color',
                  'silhouettes', 'silhouette_thickness', 'silhouette_color',
                  'silhouette_depth_jump']

    @staticmethod
    def take_snapshot(view, session, flags):
        v = view
        data = {a:getattr(v,a) for a in ViewState.save_attrs}

        # TODO: Handle cameras other than MonoCamera
        c = v.camera
        from . import MonoCamera
        if not isinstance(c, MonoCamera):
            p = c.position
            c = MonoCamera()
            c.position = p
            
        data['camera'] = v.camera
        data['lighting'] = v.lighting
        data['clip_planes'] = v.clip_planes.planes()
        data['version'] = ViewState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        # Restores session.main_view
        v = session.main_view
        ViewState.set_state_from_snapshot(v, session, data)
        return v

    @staticmethod
    def set_state_from_snapshot(view, session, data):
        v = view
        for k in ViewState.save_attrs:
            if k in data and k != 'window_size':
                setattr(v, k, data[k])

        # Root drawing had redraw callback set to None.  Restore callback.
        v.drawing.set_redraw_callback(v._drawing_manager)

        # Restore camera
        v.camera = data['camera']

        # Restore lighting
        v.lighting = data['lighting']
        v.update_lighting = True

        # Restore clip planes
        v.clip_planes.replace_planes(data['clip_planes'])

        # Restore window size
        from ..commands.windowsize import window_size
        width, height = data['window_size']
        window_size(session, width, height)

    @staticmethod
    def reset_state(view, session):
        pass


class CameraState:

    version = 1
    save_attrs = ['position', 'field_of_view']

    @staticmethod
    def take_snapshot(camera, session, flags):
        c = camera
        from .camera import MonoCamera
        if isinstance(c, MonoCamera):
            data = {a:getattr(c,a) for a in CameraState.save_attrs}
        else:
            # TODO: Restore other camera modes.
            session.logger.info('"%s" camera settings not currently saved in sessions' % c.name())
            data = {'position': c.position}
        data['version'] = CameraState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from .camera import MonoCamera
        c = MonoCamera()
        CameraState.set_state_from_snapshot(c, session, data)
        return c

    @staticmethod
    def set_state_from_snapshot(camera, session, data):
        for k in CameraState.save_attrs:
            if k in data:
                setattr(camera, k, data[k])

    @staticmethod
    def reset_state(camera, session):
        pass


class LightingState:

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

    @staticmethod
    def take_snapshot(lighting, session, flags):
        l = lighting
        data = {a:getattr(l,a) for a in LightingState.save_attrs}
        data['version'] = LightingState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import Lighting
        l = Lighting()
        LightingState.set_state_from_snapshot(l, session, data)
        return l

    @staticmethod
    def set_state_from_snapshot(lighting, session, data):
        l = lighting
        for k in DrawingState.save_attrs:
            if k in data:
                setattr(l, k, data[k])

    @staticmethod
    def reset_state(lighting, session):
        pass


class ClipPlaneState:

    version = 1
    save_attrs = [
        'name',
        'normal',
        'plane_point',
        'camera_normal',
    ]

    @staticmethod
    def take_snapshot(clip_plane, session, flags):
        cp = clip_plane
        data = {a:getattr(cp,a) for a in ClipPlaneState.save_attrs}
        data['version'] = ClipPlaneState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import ClipPlane
        cp = ClipPlane(data['name'], data['normal'], data['plane_point'], data['camera_normal'])
        return cp

    @staticmethod
    def reset_state(clip_plane, session):
        pass


class DrawingState:

    version = 1
    save_attrs = ['name', 'vertices', 'triangles', 'normals', 'vertex_colors', 
                  'triangle_mask', 'edge_mask', 'display_style', 'texture', 
                  'ambient_texture', 'ambient_texture_transform', 
                  'use_lighting', 'positions', 'display_positions', 
                  'selected_positions', 'selected_triangles_mask', 'colors']

    @staticmethod
    def take_snapshot(drawing, session, flags):
        d = drawing
        data = {a:getattr(d,a) for a in DrawingState.save_attrs}
        data['children'] = d.child_drawings()
        data['version'] = DrawingState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        d = Drawing('')
        DrawingState.set_state_from_stanpshot(d, session, data)
        return d

    @staticmethod
    def set_state_from_snapshot(drawing, session, data):
        d = drawing
        for k in DrawingState.save_attrs:
            if k in data:
                setattr(d, k, data[k])
        for c in data['children']:
            d.add_drawing(c)

    @staticmethod
    def reset_state(drawing, session):
        pass
