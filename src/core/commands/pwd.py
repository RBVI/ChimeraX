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

def pwd(session):
    '''Report the current directory to the log.'''
    import os
    directory = os.getcwd()
    session.logger.info('Current working directory is: %s' % directory)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(synopsis='print current working directory')
    cli.register('pwd', desc, pwd)
