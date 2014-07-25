def show_command(cmdname, args, session):

    from ..ui.commands import atoms_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = (('what', atoms_arg),)
    kw_args = (('atoms', no_arg),
               ('ribbons', no_arg),
               ('only', no_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    show(**kw)

def show(what = None, atoms = False, ribbons = False, only = False, session = None):

    if not atoms and not ribbons:
        atoms = True

    if what is None:
        what = session.all_atoms()

    if atoms:
        what.show_atoms(only)
    if ribbons:
        what.show_ribbon(only)

def hide_command(cmdname, args, session):

    from ..ui.commands import atoms_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = (('what', atoms_arg),)
    kw_args = (('atoms', no_arg),
               ('ribbons', no_arg),)

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    hide(**kw)

def hide(what = None, atoms = False, ribbons = False, session = None):

    if not atoms and not ribbons:
        atoms = True
        ribbons = True

    if what is None:
        what = session.all_atoms()

    if atoms:
        what.hide_atoms()
    if ribbons:
        what.hide_ribbon()

def color_command(cmdname, args, session):

    from ..ui.commands import specifier_arg, color_arg, no_arg, parse_arguments
    req_args = (('what', specifier_arg),
                ('color', color_arg))
    opt_args = ()
    kw_args = (('atoms', no_arg),
               ('ribbons', no_arg))

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    color(**kw)

def color(what = None, color = (1,1,1,1), atoms = False, ribbons = False, session = None):

    a = session.all_atoms() if what is None else what.atom_set()
    if a.count() > 0:
        color_molecule(a, color, atoms, ribbons)

    maps = session.maps() if what is None else what.maps()
    for m in maps:
        m.set_color(color)

    surfs = session.surfaces() if what is None else what.surfaces()
    for s in surfs:
        color_surface(s, color)

def color_surface(surf, color):

    c8 = tuple(int(255*r) for r in color)       # Require 0-255 color values
    for d in surf.all_drawings():
        d.vertex_colors = None
        d.color = c8

def color_molecule (atoms, color = (1,1,1,1), color_atoms = False, color_ribbons = False):

    if not color_atoms and not color_ribbons:
        color_atoms = True
        color_ribbons = True

    c8 = tuple(int(255*r) for r in color)       # Molecules require 0-255 color values
    if color_atoms:
        atoms.color_atoms(c8)
    if color_ribbons:
        atoms.color_ribbon(c8)

def style_command(cmdname, args, session):

    from ..ui.commands import atoms_arg, enum_arg, parse_arguments
    req_args = (('atoms', atoms_arg),
                ('style', enum_arg, {'values':('sphere', 'stick', 'ballstick'), 'abbrev':True}),)
    opt_args = ()
    kw_args = ()

    kw = parse_arguments(cmdname, args, session, req_args, opt_args, kw_args)
    kw['session'] = session
    style(**kw)

def style(atoms = None, style = None, session = None):

    s = {'sphere':atoms.SPHERE_STYLE,
         'stick':atoms.STICK_STYLE,
         'ballstick':atoms.BALL_STICK_STYLE,
        }[style]
    atoms.set_atom_style(s)
