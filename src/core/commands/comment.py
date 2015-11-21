# vim: set expandtab shiftwidth=4 softtabstop=4:


def comment(session, comment=''):
    '''Comment string.

    Parameters
    ----------
    comment : string
        The comment text to ignore.
    '''
    pass


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(optional=[('comment', cli.WholeRestOfLine)],
                       non_keyword=['comment'],
                       synopsis='placeholder for a comment')
    cli.register('#', desc, comment)
