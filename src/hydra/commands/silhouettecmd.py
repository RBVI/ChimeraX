def silhouette_command(cmdname, args, session):

    from .parse import float_arg, color_arg, bool_arg, parse_arguments
    req_args = (('enable', bool_arg),)
    opt_args = ()
    kw_args = (('thickness', float_arg),
               ('color', color_arg),
               ('depthJump', float_arg),
           )

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    silhouette(session, **kw)

def silhouette(session, enable, thickness = None, color = None, depthJump = None):

    v = session.view
    v.silhouettes = enable

    if not thickness is None:
        v.silhouette_thickness = thickness
    if not color is None:
        v.silhouette_color = color
    if not depthJump is None:
        v.silhouette_depth_jump = depthJump

    v.redraw_needed = True
