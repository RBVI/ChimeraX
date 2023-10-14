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

def camera(session, type=None, field_of_view=None,
           eye_separation=None, pixel_eye_separation=None, convergence=None,
           cube_pixels=1024):
    '''Change camera parameters.

    Parameters
    ----------
    type : string
        Controls type of projection, currently "mono", "360", "dome", "360tb" (stereoscopic top-bottom layout),
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
    cube_pixels : int
        Controls resolution of 360 and dome modes as the width of the cube map faces in
        pixels.  For best appearance this value should be about as large as the window size.
        Default is 1024.
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
                    run(session, 'ui tool show "Side View"')
        camera = None
        from chimerax import graphics
        if type == 'mono':
            camera = graphics.MonoCamera()
        elif type == 'ortho':
            w = view.camera.view_width(view.center_of_rotation)
            camera = graphics.OrthographicCamera(w)
        elif type == 'crosseye':
            camera = graphics.SplitStereoCamera(swap_eyes = True, convergence = 10, eye_separation_scene = 50)
        elif type == 'walleye':
            camera = graphics.SplitStereoCamera(convergence = -5)
        elif type == '360':
            camera = graphics.Mono360Camera(cube_face_size = cube_pixels)
        elif type == 'dome':
            camera = graphics.DomeCamera(cube_face_size = cube_pixels)
        elif type == '360tb':
            camera = graphics.Stereo360Camera(cube_face_size = cube_pixels)
        elif type == '360sbs':
            camera = graphics.Stereo360Camera(layout = 'side-by-side', cube_face_size = cube_pixels)
        elif type == 'stereo':
            camera = graphics.StereoCamera()
            b = view.drawing_bounds()
            if b:
                camera.position = view.camera.position
                camera.set_focus_depth(b.center(), view.window_size[0])
        elif type == 'sbs':
            camera = graphics.SplitStereoCamera()
        elif type == 'tb':
            camera = graphics.SplitStereoCamera(layout = 'top-bottom')

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
        if cam.name != 'stereo':
            from chimerax.core.errors import UserError
            raise UserError('camera pixelEyeSeparation option only applies to stereo camera mode.')
        has_arg = True
        cam.eye_separation_pixels = pixel_eye_separation
        cam.redraw_needed = True
        b = view.drawing_bounds()
        if b:
            cam.set_focus_depth(b.center(), view.window_size[0])
    if convergence is not None:
        has_arg = True
        cam.convergence = convergence
        cam.redraw_needed = True
        
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
        if hasattr(cam, 'convergence'):
            lines.append('    convergence (degrees): %.5g' % cam.convergence)
        session.logger.info('\n'.join(lines))

        fields = ['%s camera' % cam.name]
        if hasattr(cam, 'field_of_view'):
            fields.append('%.5g degree field of view' % cam.field_of_view)
        session.logger.status(', '.join(fields))


def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, EnumOf, IntArg
    types = EnumOf(('mono', 'ortho', 'crosseye', 'walleye', '360', 'dome', '360tb', '360sbs', 'stereo', 'sbs', 'tb'))
    desc = CmdDesc(
        optional = [('type', types)],
        keyword = [('field_of_view', FloatArg),
                   ('eye_separation', FloatArg),
                   ('pixel_eye_separation', FloatArg),
                   ('convergence', FloatArg),
                   ('cube_pixels', IntArg)],
        synopsis='adjust camera parameters'
    )
    register('camera', desc, camera, logger=logger)
