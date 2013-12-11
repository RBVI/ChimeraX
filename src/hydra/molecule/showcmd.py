def show_command(cmdname, args):

    from ..ui.commands import atoms_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = (('what', atoms_arg),)
    kw_args = (('atoms', no_arg),
               ('ribbons', no_arg),
               ('only', no_arg),)

    kw = parse_arguments(cmdname, args, req_args, opt_args, kw_args)
    show(**kw)

def show(what = None, atoms = False, ribbons = False, only = False):

    if not atoms and not ribbons:
        atoms = True
        ribbons = True

    if what is None:
        from . import molecule
        what = molecule.all_atoms()

    if atoms:
        what.show_atoms(only)
    if ribbons:
        what.show_ribbon(only)

def hide_command(cmdname, args):

    from ..ui.commands import atoms_arg, no_arg, parse_arguments
    req_args = ()
    opt_args = (('what', atoms_arg),)
    kw_args = (('atoms', no_arg),
               ('ribbons', no_arg),)

    kw = parse_arguments(cmdname, args, req_args, opt_args, kw_args)
    hide(**kw)

def hide(what = None, atoms = False, ribbons = False):

    if not atoms and not ribbons:
        atoms = True
        ribbons = True

    if what is None:
        from . import molecule
        what = molecule.all_atoms()

    if atoms:
        what.hide_atoms()
    if ribbons:
        what.hide_ribbon()
