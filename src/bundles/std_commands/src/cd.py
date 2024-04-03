# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def cd(session, directory=None):
    '''Change working directory.'''
    import os
    if directory is None:
        directory = os.path.expanduser('~')
        if directory == '~':
            from chimerax.core.errors import UserError
            raise UserError('Unable to figure out home directory')
    try:
        os.chdir(directory)
    except OSError as e:
        from chimerax.core.errors import UserError
        raise UserError(e)
    from . import pwd
    pwd.pwd(session)


def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, OpenFolderNameArg
    desc = CmdDesc(
        optional=[('directory', OpenFolderNameArg)],
        synopsis='Change the current working directory')
    register('cd', desc, cd, logger=logger)
