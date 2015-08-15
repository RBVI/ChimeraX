# vi: set expandtab shiftwidth=4 softtabstop=4:

def display(session, spec=None):
    '''Display specified atoms.

    Parameters
    ----------
    spec : atom specifier
        Show the specified atoms. If no atom specifier is given then all atoms are shown.
        Atoms that are already shown remain shown.
    '''
    if spec is None:
        from . import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.displays = True

def undisplay(session, spec=None):
    '''Hide specified atoms.

    Parameters
    ----------
    spec : atom specifier
        Hide the specified atoms. If no atom specifier is given then all atoms are hidden.
    '''
    if spec is None:
        from . import atomspec
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.displays = False

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                       synopsis='display specified atoms')
    cli.register('display', desc, display)
    desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                              synopsis='undisplay specified atoms')
    cli.register('~display', desc, undisplay)

