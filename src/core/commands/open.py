# vi: set expandtab shiftwidth=4 softtabstop=4:


def open(session, filename, id=None, as_=None):
    '''Open a file.

    Parameters
    ----------
    filename : string
        A path to a file (relative to the current directory), or a database id code to
        fetch prefixed by the database name, for example, pdb:1a0m, mmcif:1jj2, emdb:1080.
        A 4-letter id that is not a local file is interpreted as an mmCIF fetch.
    id : tuple of integer
        The model id number to use for this data set.
    as_ : user-supplied name (as opposed to the filename)
    '''
    try:
        return session.models.open(filename, id=id, as_=as_)
    except OSError as e:
        from ..errors import UserError
        raise UserError(e)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(required=[('filename', cli.StringArg)],
                       keyword=[('id', cli.ModelIdArg),
                                ('as_a', cli.StringArg),
                                ('label', cli.StringArg)],
                       synopsis='read and display data')
    cli.register('open', desc, open)
