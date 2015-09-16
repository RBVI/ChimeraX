# vi: set expandtab shiftwidth=4 softtabstop=4:


def ribbon(session, spec=None):
    '''Display ribbons for specified residues.

    Parameters
    ----------
    spec : atom specifier
        Show ribbons for the specified residues. If no atom specifier is given then ribbons are shown
        for all residues.  Residues that are already shown as ribbons remain shown as ribbons.
    '''
    if spec is None:
        from . import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.residues.ribbon_displays = True


def unribbon(session, spec=None):
    '''Undisplay ribbons for specified residues.

    Parameters
    ----------
    spec : atom specifier
        Hide ribbons for the specified residues. If no atom specifier is given then all ribbons are hidden.
    '''
    if spec is None:
        from . import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.residues.ribbon_displays = False


def register_command(session):
    from . import cli
    from . import atomspec
    desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                       synopsis='display ribbon for specified residues')
    cli.register('ribbon', desc, ribbon)
    desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                       synopsis='display ribbon for specified residues')
    cli.register('~ribbon', desc, unribbon)
