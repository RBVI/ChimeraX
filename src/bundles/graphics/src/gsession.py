# vim: set expandtab shiftwidth=4 softtabstop=4:
import numpy as np
from chimerax.geometry.psession import PlaceState


# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Session save/restore of graphics state

def register_graphics_session_save(session):
    from chimerax import graphics as g
    methods = {
        g.View: ViewState,
        g.MonoCamera: CameraState,
        g.OrthographicCamera: CameraState,
        g.Lighting: LightingState,
        g.Material: MaterialState,
        g.ClipPlane: ClipPlaneState,
        g.SceneClipPlane: SceneClipPlaneState,
        g.CameraClipPlane: CameraClipPlaneState,
        g.Drawing: DrawingState,
    }
    session.register_snapshot_methods(methods)

class ViewState:

    version = 1
    save_attrs = ['camera', 'lighting', 'material',
                  'center_of_rotation', 'center_of_rotation_method',
                  'background_color', 'highlight_color', 'highlight_thickness']
    silhouette_attrs = ['enabled', 'thickness', 'color', 'depth_jump']

    @staticmethod
    def take_snapshot(view, session, flags):
        v = view
        data = {a:getattr(v,a) for a in ViewState.save_attrs}

        # TODO: Handle cameras other than MonoCamera
        c = v.camera
        from . import MonoCamera, OrthographicCamera
        if not isinstance(c, (MonoCamera, OrthographicCamera)):
            p = c.position
            c = MonoCamera()
            c.position = p
            data['camera'] = c

        data['silhouettes'] = {attr:getattr(v.silhouette, attr) for attr in ViewState.silhouette_attrs}
        data['window_size'] = v.window_size
        data['clip_planes'] = v.clip_planes.planes()
        from chimerax.surface.settings import settings as surf_settings
        data['clipping_surface_caps'] = surf_settings.clipping_surface_caps
        data['clipping_cap_offset'] = surf_settings.clipping_cap_offset

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
        restore_camera = session.restore_options.get('restore camera')
        for k in ViewState.save_attrs:
            if not restore_camera and k == 'camera':
                continue
            if k in data:
                setattr(v, k, data[k])

        # Root drawing has redraw callback set to None -- fix it.
        v.drawing.set_redraw_callback(v._drawing_manager)

        # Restore clip planes
        cplist = data['clip_planes']
        cplist = [cp for cp in cplist if cp is not None]	# Fix old session files.
        ClipPlaneState._fix_plane_points(cplist, v.camera.position)	# Fix old session files.
        v.clip_planes.replace_planes(cplist)

        # Restore clip caps
        if 'clipping_surface_caps' in data:
            from chimerax.surface.settings import settings as surf_settings
            surf_settings.clipping_surface_caps = data['clipping_surface_caps']
            surf_settings.clipping_cap_offset = data['clipping_cap_offset']

        # Restore silhouette edge settings.
        if 'silhouettes' in data:
            sil = data['silhouettes']
            if isinstance(sil, dict):
                for attr, value in sil.items():
                    setattr(v.silhouette, attr, value)

        # Restore window size
        resize = session.restore_options.get('resize window')
        if resize is None:
            from chimerax.core.core_settings import settings
            resize = settings.resize_window_on_session_restore
        if resize:
            ui = session.ui
            maximized = (ui.is_gui and ui.main_window.window_maximized())
            if maximized:
                resize = False
        if resize:
            from .windowsize import window_size
            width, height = data['window_size']
            window_size(session, width, height)

    @staticmethod
    def reset_state(view, session):
        if not session.restore_options.get('restore camera'):
            return
        view.set_default_parameters()
        from chimerax.surface.settings import settings as surf_settings
        surf_settings.reset() # Reset clip cap settings

    @staticmethod
    def include_state(view):
        return True

    @staticmethod
    def interpolate(view, scene1, scene2, frac):
        """
        Interpolate view at frac of the way between two scene datasets.
        :param view: View object to interpolate
        :param scene1: Starting ViewState scene data
        :param scene2: Ending ViewState scene data
        :param frac: Fraction of the way between scene1 and scene2 to interpolate

        save_attrs = ['camera', 'lighting', 'material',
                  'center_of_rotation', 'center_of_rotation_method',
                  'background_color', 'highlight_color', 'highlight_thickness']
        silhouette_attrs = ['enabled', 'thickness', 'color', 'depth_jump']
        """

        for view_attr in ViewState.save_attrs:
            if view_attr in scene1 and view_attr in scene2:
                # Setting the center_of_rotation resets the center_of_rotation method to fixed. Changing the other
                # center of rotation methods automatically calculate the center of rotation.
                # For simplicity use a threshold lerp for center_of_rotation and center_of_rotation_method.

                if view_attr == "center_of_rotation":
                    # Center of rotation is interpolated first according to save_attrs order, settattr will
                    # automatically switch the center of rotation method to fixed because we set its value.
                    lerp_val = threshold_frac_lerp(scene1[view_attr], scene2[view_attr], frac)
                    setattr(view, view_attr, lerp_val)
                elif view_attr == "center_of_rotation_method":
                    # center_of_rotation method will be interpolated second according to save_attrs order. Since we set
                    # the center of rotation first, the method will be set to fixed. If it is not supposed to be fixed
                    # we will overwrite it with the correct method here, and it will automatically recalculate the
                    # appropriate center of rotation.
                    lerp_val = threshold_frac_lerp(scene1[view_attr], scene2[view_attr], frac)
                    setattr(view, view_attr, lerp_val)
                elif view_attr == "background_color":
                    lerp_val = list_frac_lerp(scene1[view_attr], scene2[view_attr], frac)
                    setattr(view, view_attr, lerp_val)
                elif view_attr == "highlight_color":
                    # Highlight color is the select option color
                    lerp_val = list_frac_lerp(scene1[view_attr], scene2[view_attr], frac)
                    setattr(view, view_attr, lerp_val)
                elif view_attr == "highlight_thickness":
                    # Highlight thickness only changes on the whole number. Round to avoid only potential issues
                    lerp_val = round(num_frac_lerp(scene1[view_attr], scene2[view_attr], frac))
                    setattr(view, view_attr, lerp_val)
                elif view_attr == "lighting":
                    LightingState.interpolate(view.lighting, scene1[view_attr], scene2[view_attr], frac)
                elif view_attr == "material":
                    MaterialState.interpolate(view.material, scene1[view_attr], scene2[view_attr], frac)

        s_data1 = scene1['silhouettes']
        s_data2 = scene2['silhouettes']
        for silhouette_attr in ViewState.silhouette_attrs:
            if silhouette_attr in s_data1 and silhouette_attr in s_data2:
                if silhouette_attr == "enabled":
                    # Enabled is a boolean
                    lerp_val = threshold_frac_lerp(s_data1[silhouette_attr], s_data2[silhouette_attr], frac)
                    setattr(view.silhouette, silhouette_attr, lerp_val)
                elif silhouette_attr == "thickness":
                    # Not a fluid interpolation. Some value steps don't display changes like view.highlight_thickness
                    lerp_val = num_frac_lerp(s_data1[silhouette_attr], s_data2[silhouette_attr], frac)
                    setattr(view.silhouette, silhouette_attr, lerp_val)
                elif silhouette_attr == "color":
                    lerp_val = list_frac_lerp(s_data1[silhouette_attr], s_data2[silhouette_attr], frac)
                    setattr(view.silhouette, silhouette_attr, lerp_val)
                elif silhouette_attr == "depth_jump":
                    lerp_val = num_frac_lerp(s_data1[silhouette_attr], s_data2[silhouette_attr], frac)
                    setattr(view.silhouette, silhouette_attr, lerp_val)

        view.update_lighting = True
        view.redraw_needed = True


class CameraState:

    version = 1
    save_attrs = ['name', 'position', 'field_of_view', 'field_width']

    @staticmethod
    def take_snapshot(camera, session, flags):
        c = camera
        from .camera import MonoCamera, OrthographicCamera
        if isinstance(c, (MonoCamera, OrthographicCamera)):
            data = {a:getattr(c,a) for a in CameraState.save_attrs if hasattr(c,a)}
        else:
            # TODO: Restore other camera modes.
            session.logger.info('"%s" camera settings not currently saved in sessions' % c.name)
            data = {'position': c.position}
        data['version'] = CameraState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        cname = data.get('name', 'mono')
        from .camera import MonoCamera, OrthographicCamera
        if cname == 'mono':
            c = MonoCamera()
        elif cname == 'orthographic':
            c = OrthographicCamera()
        else:
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
        'depth_cue',
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
        for k in LightingState.save_attrs:
            if k in data:
                setattr(l, k, data[k])

    @staticmethod
    def reset_state(lighting, session):
        pass

    @staticmethod
    def interpolate(lighting, scene1, scene2, frac):
        """
        save_attrs = ['key_light_direction', 'key_light_color', 'key_light_intensity', 'fill_light_direction',
        'fill_light_color', 'fill_light_intensity', 'ambient_light_color', 'ambient_light_intensity', 'depth_cue',
        'depth_cue_start', 'depth_cue_end', 'depth_cue_color', 'move_lights_with_camera', 'shadows',
        'shadow_map_size', 'shadow_depth_bias', 'multishadow', 'multishadow_map_size', 'multishadow_depth_bias']
        """
        for light_attr in LightingState.save_attrs:
            if light_attr in scene1 and light_attr in scene2:
                if light_attr == "key_light_direction":
                    # Numpy array
                    lerp_val = list_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "key_light_color":
                    lerp_val = list_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "key_light_intensity":
                    lerp_val = num_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "fill_light_direction":
                    lerp_val = list_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "fill_light_color":
                    lerp_val = list_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "fill_light_intensity":
                    lerp_val = num_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "ambient_light_color":
                    lerp_val = list_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "ambient_light_intensity":
                    lerp_val = num_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "depth_cue":
                    lerp_val = threshold_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "depth_cue_start":
                    lerp_val = num_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "depth_cue_end":
                    # This works, but it could maybe take a 'shorter path' in interpolating to look better.
                    lerp_val = num_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "depth_cue_color":
                    lerp_val = list_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "move_lights_with_camera":
                    lerp_val = threshold_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "shadows":
                    lerp_val = threshold_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "shadow_map_size":
                    # Shadow map size needs to be a whole number
                    lerp_val = round(num_frac_lerp(scene1[light_attr], scene2[light_attr], frac))
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "shadow_depth_bias":
                    lerp_val = num_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "multishadow":
                    # Need whole number values
                    lerp_val = round(num_frac_lerp(scene1[light_attr], scene2[light_attr], frac))
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "multishadow_map_size":
                    lerp_val = round(num_frac_lerp(scene1[light_attr], scene2[light_attr], frac))
                    setattr(lighting, light_attr, lerp_val)
                elif light_attr == "multishadow_depth_bias":
                    lerp_val = num_frac_lerp(scene1[light_attr], scene2[light_attr], frac)
                    setattr(lighting, light_attr, lerp_val)


class MaterialState:

    version = 1
    save_attrs = [
        'ambient_reflectivity',
        'diffuse_reflectivity',
        'specular_reflectivity',
        'specular_exponent',
        'transparent_cast_shadows',
        'meshes_cast_shadows',
        ]

    @staticmethod
    def take_snapshot(material, session, flags):
        m = material
        data = {a:getattr(m,a) for a in MaterialState.save_attrs}
        data['version'] = MaterialState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import Material
        m = Material()
        MaterialState.set_state_from_snapshot(m, session, data)
        return m

    @staticmethod
    def set_state_from_snapshot(material, session, data):
        m = material
        for k in MaterialState.save_attrs:
            if k in data:
                setattr(m, k, data[k])

    @staticmethod
    def reset_state(Material, session):
        pass

    @staticmethod
    def interpolate(material, scene1, scene2, frac):
        """
        save_attrs = [
        'ambient_reflectivity', 'diffuse_reflectivity',
        'specular_reflectivity', 'specular_exponent',
        'transparent_cast_shadows', 'meshes_cast_shadows']
        """

        for mat_attr in MaterialState.save_attrs:
            if mat_attr in scene1 and mat_attr in scene2:
                if mat_attr == "ambient_reflectivity":
                    lerp_val = num_frac_lerp(scene1[mat_attr], scene2[mat_attr], frac)
                    setattr(material, mat_attr, lerp_val)
                elif mat_attr == "diffuse_reflectivity":
                    lerp_val = num_frac_lerp(scene1[mat_attr], scene2[mat_attr], frac)
                    setattr(material, mat_attr, lerp_val)
                elif mat_attr == "specular_reflectivity":
                    lerp_val = num_frac_lerp(scene1[mat_attr], scene2[mat_attr], frac)
                    setattr(material, mat_attr, lerp_val)
                elif mat_attr == "specular_exponent":
                    lerp_val = num_frac_lerp(scene1[mat_attr], scene2[mat_attr], frac)
                    setattr(material, mat_attr, lerp_val)
                elif mat_attr == "transparent_cast_shadows":
                    # TODO find something to test this interpolation with
                    lerp_val = threshold_frac_lerp(scene1[mat_attr], scene2[mat_attr], frac)
                    setattr(material, mat_attr, lerp_val)
                elif mat_attr == "meshes_cast_shadows":
                    # TODO find something to test this interpolation with
                    lerp_val = threshold_frac_lerp(scene1[mat_attr], scene2[mat_attr], frac)
                    setattr(material, mat_attr, lerp_val)


class ClipPlaneState:
    '''This is no longer used for saving sessions but is kept for restoring old sessions.'''
    
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
        camera_normal = data['camera_normal']
        if camera_normal is None:
            from . import SceneClipPlane
            cp = SceneClipPlane(data['name'], data['normal'], data['plane_point'])
        else:
            v = session.main_view
            from . import CameraClipPlane
            cp = CameraClipPlane(data['name'], camera_normal, data['plane_point'], v)
            # Camera has not yet been restored, so camera plane point is wrong.
            # Fix it after camera is restored with _fix_plane_points() call.
            cp._session_restore_fix_plane_point = True
        return cp

    @staticmethod
    def _fix_plane_points(clip_planes, camera_pos):
        # Fix old session files clip plane state now that camera has been restored.
        for cp in clip_planes:
            if hasattr(cp, '_session_restore_fix_plane_point'):
                cp._camera_plane_point = camera_pos.inverse() * cp._camera_plane_point

    @staticmethod
    def reset_state(clip_plane, session):
        pass


class SceneClipPlaneState:
    
    version = 1
    save_attrs = [
        'name',
        'normal',
        'plane_point',
    ]

    @staticmethod
    def take_snapshot(clip_plane, session, flags):
        cp = clip_plane
        data = {a:getattr(cp,a) for a in SceneClipPlaneState.save_attrs}
        data['version'] = SceneClipPlaneState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import SceneClipPlane
        cp = SceneClipPlane(data['name'], data['normal'], data['plane_point'])
        return cp

    @staticmethod
    def reset_state(clip_plane, session):
        pass


class CameraClipPlaneState:
    
    version = 1
    save_attrs = [
        'name',
        '_camera_normal',
        '_camera_plane_point',
    ]

    @staticmethod
    def take_snapshot(clip_plane, session, flags):
        cp = clip_plane
        data = {a:getattr(cp,a) for a in CameraClipPlaneState.save_attrs}
        data['version'] = CameraClipPlaneState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        from . import CameraClipPlane
        cp = CameraClipPlane(data['name'], data['_camera_normal'], data['_camera_plane_point'],
                             session.main_view)
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
                  'highlighted_positions', 'highlighted_triangles_mask', 'colors',
                  'allow_depth_cue', 'allow_clipping', 'accept_shadow', 'accept_multishadow']

    @staticmethod
    def take_snapshot(drawing, session, flags, include_children = True):
        d = drawing
        data = {a:getattr(d,a) for a in DrawingState.save_attrs}
        data['children'] = d.child_drawings() if include_children else []
        data['version'] = DrawingState.version
        return data

    @staticmethod
    def restore_snapshot(session, data):
        d = Drawing('')
        DrawingState.set_state_from_snapshot(d, session, data)
        return d

    @staticmethod
    def set_state_from_snapshot(drawing, session, data):
        d = drawing
        setattr(d, 'name', data['name'])
        # need to do vertices, normals, triangles first
        d.set_geometry(data['vertices'], data['normals'], data['triangles'])
        for k in DrawingState.save_attrs[4:]:
            if k in data:
                setattr(d, k, data[k])
        for c in data['children']:
            d.add_drawing(c)

    @staticmethod
    def reset_state(drawing, session):
        pass


def num_frac_lerp(value1, value2, fraction):
    """
    Linear interpolation between two values based on a fraction.
    Supported Types: number types, np.ndarray
    """
    return value1 + fraction * (value2 - value1)


def list_frac_lerp(value1, value2, fraction):
    """
    Linear interpolation between two list-like types based on a fraction.
    Supported Types: list, tuple ...
    """
    return [value1[i] + fraction * (value2[i] - value1[i]) for i in range(len(value1))]


def threshold_frac_lerp(value1, value2, fraction):
    """
    Either value1 or value2 based on a threshold.
    Supported Types: bool, str
    """
    return value1 if fraction < 0.5 else value2