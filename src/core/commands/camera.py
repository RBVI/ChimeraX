# vi: set expandtab shiftwidth=4 softtabstop=4:


def camera(session, type=None, field_of_view=None,
           eye_separation=None, pixel_eye_separation=None):
    '''Change camera parameters.

    Parameters
    ----------
    type : string
        Controls type of projection, currently "mono", "360" or "360s" (stereoscopic)
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
    cam = session.main_view.camera
    has_arg = False
    if type is not None:
        has_arg = True
        if type == 'mono':
            from ..graphics import MonoCamera
            view.camera = MonoCamera()
        elif type == '360':
            from ..graphics import Mono360Camera
            view.camera = Mono360Camera()
        elif type == '360s':
            from ..graphics import Stereo360Camera
            view.camera = Stereo360Camera()

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

    if not has_arg:
        msg = (
            'Camera parameters:\n' +
            '    position: %.5g %.5g %.5g\n' % tuple(cam.position.origin()) +
            '    view direction: %.6f %.6f %.6f\n' %
            tuple(cam.view_direction()) +
            '    field of view: %.5g degrees\n' % cam.field_of_view +
            '    mode: %s\n' % cam.mode.name()
        )
        session.logger.info(msg)
        msg = (cam.mode.name() +
               ', %.5g degree field of view' % cam.field_of_view)
        session.logger.status(msg)


def register_command(session):
    from . import CmdDesc, register, FloatArg, EnumOf
    desc = CmdDesc(
        optional=[
            ('type', EnumOf(('mono', '360', '360s'))),
            ('field_of_view', FloatArg),
            ('eye_separation', FloatArg),
            ('pixel_eye_separation', FloatArg),
        ],
        synopsis='adjust camera parameters'
    )
    register('camera', desc, camera)
