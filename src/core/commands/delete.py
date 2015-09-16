# vi: set expandtab shiftwidth=4 softtabstop=4:


def delete(session, atoms):
    '''Delete atoms.

    Parameters
    ----------
    atoms : Atoms collection
        Delete these atoms.  If all atoms of a model are closed then the model is closed.
    '''
    atoms.delete()


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(required=[('atoms', cli.AtomsArg)],
                       synopsis='delete atoms')
    cli.register('delete', desc, delete)
