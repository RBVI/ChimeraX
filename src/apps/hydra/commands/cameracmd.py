def camera_command(cmdname, args, session):

    from .parse import float_arg, floats_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = ()
    kw_args = (('mono', no_arg),
               ('stereo', no_arg),
               ('oculus', no_arg),
               ('fieldOfView', float_arg),      # degrees, width
               ('eyeSeparation', float_arg),    # physical units
               ('screenWidth', float_arg),      # physical units
               ('sEyeSeparation', float_arg),   # scene units
               ('middleDistance', no_arg),      # Adjust scene eye sep so models at screen depth.
               ('depthScale', float_arg),       # Scale scene and pixel eye separations
               ('nearFarClip', floats_arg, {'allowed_counts':(2,)}),     # scene units
               ('report', no_arg),
           )

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    camera(session, **kw)

def camera(session, mono = None, stereo = None, oculus = None, fieldOfView = None, 
           eyeSeparation = None, screenWidth = None, sEyeSeparation = None,
           middleDistance = False, depthScale = None, report = False):

    v = session.view
    c = v.camera
    
    if mono:
        from ..graphics import mono_camera_mode
        c.mode = mono_camera_mode
    elif stereo:
        from ..graphics import stereo_camera_mode
        c.mode = stereo_camera_mode
    elif oculus:
        from ..devices.oculus import OculusRiftCameraMode
        c.mode = OculusRiftCameraMode()

    if not fieldOfView is None:
        c.field_of_view = fieldOfView
        c.redraw_needed = True
    if not eyeSeparation is None or not screenWidth is None:
        if eyeSeparation is None or screenWidth is None:
            from .parse import CommandError
            raise CommandError('Must specify eyeSeparation and screenWidth, only ratio is used')
        c.eye_separation_pixels = (eyeSeparation / screenWidth) * v.screen().size().width()
        c.redraw_needed = True
    if not sEyeSeparation is None:
        c.eye_separation_scene = sEyeSeparation
        c.redraw_needed = True
    if middleDistance:
        center = session.bounds().center()
        wscene = c.view_width(center)
        wpixels = v.window_size[0]
        c.eye_separation_scene = wscene * c.eye_separation_pixels / wpixels
        c.redraw_needed = True
    if not depthScale is None:
        # This scales the apparent depth while leaving apparent distance to models the same.
        c.eye_separation_pixels *= depthScale
        c.eye_separation_scene *= depthScale
        c.redraw_needed = True
    if report:
        msg = ('Camera\n' +
               'position %.5g %.5g %.5g\n' % tuple(c.position.origin()) +
               'view direction %.6f %.6f %.6f\n' % tuple(c.view_direction()) +
               'field of view %.5g degrees\n' % c.field_of_view +
               'mode %s\n' % c.mode.name())
        if hasattr(c, 'eye_separation_pixels') and hasattr(c, 'eye_separation_scene'):
            msg += 'eye separation pixels %.5g, scene %.5g' % (c.eye_separation_pixels, c.eye_separation_scene)
        session.show_info(msg)
        smsg = 'Camera mode %s, field of view %.4g degrees' % (c.mode.name(), c.field_of_view)
        session.show_status(smsg)
