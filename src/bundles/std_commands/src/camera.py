# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def camera(session, type=None, field_of_view=None,
           eye_separation=None, pixel_eye_separation=None):
    '''Change camera parameters.

    Parameters
    ----------
    type : string
        Controls type of projection, currently "mono", "360", "360tb" (stereoscopic top-bottom layout),
        "360sbs" (stereoscopic side-by-side layout), "stereo", "sbs" (side by side stereo), "tb" (top bottom stereo)
    field_of_view : float
        Horizontal field of view in degrees.
    eye_separation : float
        Distance between left/right eye cameras for stereo camera modes in scene distance units.
    pixel_eye_separation : float
        Physical distance between viewer eyes for stereo camera modes in screen pixels.
        This is needed for shutter glasses stereo so that an object very far away appears
        has left/right eye images separated by the viewer's physical eye spacing.
        Usually this need not be set and will be figured out from the pixels/inch reported
        by the display.  But for projectors the size of the displayed image is unknown and
        it is necessary to set this option to get comfortable stereoscopic viewing.
    '''
    view = session.main_view
    has_arg = False
    if type is not None:
        has_arg = True
        if (type == 'stereo') != view.render.opengl_context.stereo:
            # Need to switch stereo mode of OpenGL context.
            if not session.ui.main_window.enable_stereo(type == 'stereo'):
                from chimerax.core.errors import UserError
                raise UserError('Could not switch graphics mode.  '
                                'Graphics driver did not create OpenGL context.')
            # Close side view since it must be restarted to use new OpenGL context
            for t in tuple(session.tools.list()):
                if t.tool_name == 'Side View':
                    t.delete()
                    session.logger.info('Restarting side view because stereo mode switched')
                    from chimerax.core.commands import run
                    run(session, 'toolshed show "Side View"')
        camera = None
        if type == 'mono':
            from chimerax.core.graphics import MonoCamera
            camera = MonoCamera()
        elif type == 'ortho':
            from chimerax.core.graphics import OrthographicCamera
            w = view.camera.view_width(view.center_of_rotation)
            camera = OrthographicCamera(w)
        elif type == '360':
            from chimerax.core.graphics import Mono360Camera
            camera = Mono360Camera()
        elif type == '360tb':
            from chimerax.core.graphics import Stereo360Camera
            camera = Stereo360Camera()
        elif type == '360sbs':
            from chimerax.core.graphics import Stereo360Camera
            camera = Stereo360Camera(layout = 'side-by-side')
        elif type == 'stereo':
            from chimerax.core.graphics import StereoCamera
            camera = StereoCamera()
            b = view.drawing_bounds()
            if b:
                camera.position = view.camera.position
                camera.set_focus_depth(b.center(), view.window_size[0])
        elif type == 'sbs':
            from chimerax.core.graphics import SplitStereoCamera
            camera = SplitStereoCamera()
        elif type == 'tb':
            from chimerax.core.graphics import SplitStereoCamera
            camera = SplitStereoCamera(layout = 'top-bottom')

        if camera is not None:
            camera.position = view.camera.position  # Preserve current camera position
            view.camera.delete()
            view.camera = camera

    cam = session.main_view.camera
    if field_of_view is not None:
        has_arg = True
        cam.field_of_view = field_of_view
        cam.redraw_needed = True
    if eye_separation is not None:
        has_arg = True
        cam.eye_separation_scene = eye_separation
        cam.redraw_needed = True
    if pixel_eye_separation is not None:
        has_arg = True
        cam.eye_separation_pixels = pixel_eye_separation
        cam.redraw_needed = True
        b = view.drawing_bounds()
        if b:
            cam.set_focus_depth(b.center(), view.window_size[0])

    if not has_arg:
        lines = [
            'Camera parameters:',
            '    type: %s' % cam.name,
            '    position: %.5g %.5g %.5g' % tuple(cam.position.origin()),
            '    view direction: %.5g %.5g %.5g' % tuple(cam.view_direction())
            ]
        if hasattr(cam, 'field_of_view'):
            lines.append('    field of view: %.5g degrees' % cam.field_of_view)
        if hasattr(cam, 'field_width'):
            lines.append('    field width: %.5g' % cam.field_width)
        if hasattr(cam, 'eye_separation_scene'):
            lines.append('    eye separation in scene: %.5g' % cam.eye_separation_scene)
        if hasattr(cam, 'eye_separation_pixels'):
            lines.append('    eye separation in screen pixels: %.5g' % cam.eye_separation_pixels)
        session.logger.info('\n'.join(lines))

        fields = ['%s camera' % cam.name]
        if hasattr(cam, 'field_of_view'):
            fields.append('%.5g degree field of view' % cam.field_of_view)
        session.logger.status(', '.join(fields))


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, EnumOf
    types = EnumOf(('mono', 'ortho', '360', '360tb', '360sbs', 'stereo', 'sbs', 'tb'))
    desc = CmdDesc(
        optional = [('type', types)],
        keyword = [('field_of_view', FloatArg),
                   ('eye_separation', FloatArg),
                   ('pixel_eye_separation', FloatArg)],
        synopsis='adjust camera parameters'
    )
    register('camera', desc, camera, logger=logger)
