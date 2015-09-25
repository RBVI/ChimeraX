# vi: set expandtab shiftwidth=4 softtabstop=4:


def camera(session, mode=None, field_of_view=None, eye_separation=None,
           screen_width=None, depth_scale=None):
    '''Change camera parameters.

    Parameters
    ----------
    mode : string
        Controls type of projection, currently "mono" or "360"
    field_of_view : float
        Horizontal field of view in degrees.
    eye_separation : float
        Distance between eyes for stereo camera modes.  Can use any units and must specify
        screen width too.  Only eye_separation divided by screen width is used.
    screen_width : float
        Width of screen in same units as eye separation.  Only used for stereo camera modes.
    depth_scale : float
        Scale the eye separation by this factor.  Only used in stereo camera modes.
    '''
    view = session.main_view
    cam = session.main_view.camera
    has_arg = False
    if mode is not None:
        if mode == 'mono':
            from ..graphics import mono_camera_mode
            cam.mode = mono_camera_mode
        elif mode == '360':
            from ..graphics.camera360 import mono_360_camera_mode
            mono_360_camera_mode.set_camera_mode(cam)
        has_arg = True
        # TODO
    if field_of_view is not None:
        has_arg = True
        cam.field_of_view = field_of_view
        cam.redraw_needed = True
    if eye_separation is not None or screen_width is not None:
        has_arg = True
        if eye_separation is None or screen_width is None:
            from ..errors import UserError
            raise UserError("Must specifiy both eye-separation and"
                            " screen-width -- only ratio is used")
        cam.eye_separation_pixels = (eye_separation / screen_width) * \
            view.screen().size().width()
        cam.redraw_needed = True
    if depth_scale is not None:
        has_arg = True
        cam.eye_separation_pixels *= depth_scale
        cam.eye_separation_scene *= depth_scale
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
            ('mode', EnumOf(('mono', '360'))),
            ('field_of_view', FloatArg),
            ('eye_separation', FloatArg),
            ('screen_width', FloatArg),
            ('depth_scale', FloatArg),
        ],
        synopsis='adjust camara parameters'
    )
    register('camera', desc, camera)
