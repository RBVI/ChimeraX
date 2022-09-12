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
    bin_dirs = ['bin', 'build/bin'] # for Python 3 / Python 2 Phenix respectively
    settings = _phenix_settings(session)
    from os.path import isfile, isdir, join, expanduser
    from chimerax.core.errors import UserError
    if phenix_location is None:
        if settings.phenix_location:
            for bin_dir in bin_dirs:
                cmd = join(settings.phenix_location, bin_dir, program_name)
                if isfile(cmd):
                    return cmd

        phenix_dirs = []
        search_dirs = [expanduser("~")]
        import sys
        if sys.platform == 'darwin':
            search_dirs.append('/Applications')
        for search_dir in search_dirs:
            if isdir(search_dir):
                from os import listdir
                pdirs = [join(search_dir,f) for f in listdir(search_dir)
                         if f.startswith('phenix') and isdir(join(search_dir,f))]
                pdirs.sort(reverse = True)
                phenix_dirs.extend(pdirs)
        if len(phenix_dirs) == 0:
            raise UserError('Could not find Phenix installation in ' + ', '.join(search_dirs)
                + '.\nUse "phenix location" command to specify the location of your Phenix installlation.')
        for pdir in phenix_dirs:
            for bin_dir in bin_dirs:
                cmd = join(pdir, bin_dir, program_name)
                if isfile(cmd):
                    return cmd
        from chimerax.core.commands import commas
        raise UserError('Could not find phenix program %s in %s folder of %s' % (program_name,
            commas(bin_dirs), commas(phenix_dirs)))
    else:
        for bin_dir in bin_dirs:
            cmd = join(phenix_location, bin_dir, program_name)
            if isfile(cmd):
                break
        else:
            raise UserError('Could not find phenix program ' + program_name)
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
