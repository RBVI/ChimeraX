# vim: set expandtab ts=4 sw=4:

def window_size_command(cmdname, args, session):

    from ..commands.parse import int_arg, parse_arguments
    req_args = ()
    opt_args = ((('width', int_arg),
                 ('height', int_arg),))
    kw_args = ()

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    from . import set_window_size
    set_window_size(session, **kw)
