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

from chimerax.core.errors import UserError

def find_phenix_command(session, program_name, phenix_location=None, *, from_root=False):
    if from_root:
        bin_dirs = ['.']
    else:
        bin_dirs = ['bin', 'build/bin'] # for Python 3 / Python 2 Phenix respectively
    from .settings import get_settings
    settings = get_settings(session)
    from os.path import isfile, isdir, join, expanduser
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

def phenix_location(session, phenix_location = None):
    try:
        cmd = find_phenix_command(session, "phenix_env.sh", from_root=True)
    except UserError:
        msg = "No Phenix installation found"
    else:
        # remove trailing /./phenix_env.sh
        import os.path
        dot_path, env = os.path.split(cmd)
        install_path, dot = os.path.split(dot_path)
        msg = "Using Phenix installation %s" % install_path
    session.logger.status(msg, log=True)
