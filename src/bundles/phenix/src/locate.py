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

# ---------------------------------------------------------------------------------------
#
def find_phenix_command(session, program_name, phenix_location = None):
    bin_dir = 'build/bin'	# For Python 2 Phenix
    bin_dir = 'bin'		# For Python 3 Phenix
    settings = _phenix_settings(session)
    from os.path import isfile, isdir, join
    if phenix_location is None:
        if settings.phenix_location:
            cmd = join(settings.phenix_location, bin_dir, program_name)
            if isfile(cmd):
                return cmd
            
        phenix_dirs = []
        import sys
        if sys.platform == 'darwin':
            search_dirs = ['/Applications']
        for search_dir in search_dirs:
            if isdir(search_dir):
                from os import listdir
                pdirs = [join(search_dir,f) for f in listdir(search_dir)
                         if f.startswith('phenix') and isdir(join(search_dir,f))]
                pdirs.sort(reverse = True)
                phenix_dirs.extend(pdirs)
        if len(phenix_dirs) == 0:
            from chimerax.core.errors import UserError
            raise UserError('Could not find phenix installation in ' + ', '.join(search_dirs))
        for pdir in phenix_dirs:
            cmd = join(pdir, bin_dir, program_name)
            if isfile(cmd):
                return cmd
    else:
        cmd = join(phenix_location, bin_dir, program_name)
        if not isfile(cmd):
            from chimerax.core.errors import UserError
            raise UserError('Could not find phenix program ' + cmd)
        settings.phenix_location = phenix_location
        settings.save()
        return cmd

# ---------------------------------------------------------------------------------------
#
def _phenix_settings(session):
    settings = getattr(session, '_phenix_settings', None)
    if settings is None:
        from chimerax.core.settings import Settings
        class _PhenixSettings(Settings):
            EXPLICIT_SAVE = {
                'phenix_location': '',
            }
        settings = _PhenixSettings(session, 'phenix')
        session._phenix_settings = settings
    return settings

# ---------------------------------------------------------------------------------------
#
def phenix_location(session, phenix_location = None):
    settings = _phenix_settings(session)
    if phenix_location is None:
        loc = settings.phenix_location
        if loc:
            msg = f'Using Phenix installation {loc}'
        else:
            msg = 'No Phenix installation location set'
    else:
        settings.phenix_location = phenix_location
        settings.save()
        msg = f'Using Phenix installation {phenix_location}'

    session.logger.status(msg, log = True)
    
# ---------------------------------------------------------------------------------------
#
def register_phenix_location_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFolderNameArg
    desc = CmdDesc(
        optional = [('phenix_location', OpenFolderNameArg)],
        synopsis = 'Set the Phenix installation location'
    )
    register('phenix location', desc, phenix_location, logger=logger)
