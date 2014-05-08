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
        from . import molecule
        what = molecule.all_atoms(session)

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
        from . import molecule
        what = molecule.all_atoms(session)

    if atoms:
        what.hide_atoms()
    if ribbons:
        what.hide_ribbon()
