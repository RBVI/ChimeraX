# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

def cd(session, directory=None):
    '''Change working directory.'''
    import os
    if directory is None:
        directory = os.path.expanduser('~')
        if directory == '~':
            from ..errors import UserError
            raise UserError('Unable to figure out home directory')
    try:
        os.chdir(directory)
    except OSError as e:
        from ..errors import UserError
        raise UserError(e)
    from . import pwd
    pwd.pwd(session)


def register_command(session):
    from . import register, CmdDesc, OpenFolderNameArg
    desc = CmdDesc(
        optional=[('directory', OpenFolderNameArg)],
        synopsis='Change the current working directory')
    register('cd', desc, cd, logger=session.logger)
