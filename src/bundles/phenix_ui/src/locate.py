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

from chimerax.core.errors import UserError

def find_phenix_command(session, program_name, phenix_location=None, *, verify_installation=False):
    if verify_installation:
        bin_dirs = ['.']
    else:
        bin_dirs = ['phenix_bin', 'bin', 'build/bin'] # for Python 3 / Python 2 Phenix respectively
    import sys
    if sys.platform == 'win32':
        bin_dirs += ['Library', 'Library\\bin']
    from .settings import get_settings
    settings = get_settings(session)
    from os.path import isfile, isdir, join, expanduser
    if phenix_location is None:
        if settings.phenix_location:
            for bin_dir in bin_dirs:
                cmd = join(settings.phenix_location, bin_dir, program_name)
                if sys.platform == 'win32' and not verify_installation:
                    cmd += '.bat'
                if isfile(cmd):
                    return cmd

        phenix_dirs = []
        search_dirs = [expanduser("~")]
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
            if verify_installation and phenix_location != settings.phenix_location:
                from chimerax.ui.ask import ask
                if ask(session, "Confirm Phenix Location", info="%s does not seem to be a Phenix"
                        " installation (no '%s' in top folder), use anyway?" % (phenix_location,
                        program_name), default="no") == "no":
                    raise UserError('Could not find phenix program ' + program_name)
        settings.phenix_location = phenix_location
        settings.save()
        return cmd

def env_file_name():
    import sys
    if sys.platform == "win32":
        suffix = ".bat"
    else:
        suffix = ".sh"
    return "phenix_env" + suffix

def phenix_location(session, phenix_location=None):
    try:
        cmd = find_phenix_command(session, env_file_name(), phenix_location=phenix_location,
            verify_installation=True)
    except UserError:
        msg = "No Phenix installation found"
    else:
        # remove trailing /./phenix_env.sh
        import os.path
        dot_path, env = os.path.split(cmd)
        install_path, dot = os.path.split(dot_path)
        msg = "Using Phenix installation %s" % install_path
    session.logger.status(msg, log=True)
