def color_command(cmdname, args, session):

    from ..ui.commands import atoms_arg, color_arg, no_arg, parse_arguments
    req_args = (('what', atoms_arg),
                ('color', color_arg))
    opt_args = ()
    kw_args = (('atoms', no_arg),
               ('ribbons', no_arg))

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    color(**kw)

def color(what = None, color = (1,1,1,1), atoms = False, ribbons = False, session = None):

    if not atoms and not ribbons:
        atoms = True
        ribbons = True

    if what is None:
        what = session.all_atoms()

    c8 = tuple(int(255*r) for r in color)       # Molecules require 0-255 color values
    if atoms:
        what.color_atoms(c8)
    if ribbons:
        what.color_ribbon(c8)
