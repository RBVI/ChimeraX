#
# Turn command to rotate models.
#
def turn_command(cmdname, args, session):

    from .parse import float_arg, int_arg, axis_arg, parse_arguments
    req_args = (('axis', axis_arg),
                ('angle', float_arg),)
    opt_args = (('frames', int_arg),)
    kw_args = ()
    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    turn(**kw)

def turn(axis, angle, frames = 1, session = None):

    v = session.view
    c = v.camera
    cv = c.view()
    saxis = cv.apply_without_translation(axis)  # Convert axis from camera to scene coordinates
    center = v.center_of_rotation
    from ..geometry.place import rotation
    r = rotation(saxis, -angle, center)
    if frames == 1:
        c.set_view(r*cv)
    else:
        def rotate(r=r,c=c):
            c.set_view(r*c.view())
        from . import motion
        motion.call_for_n_frames(rotate, frames, session)
