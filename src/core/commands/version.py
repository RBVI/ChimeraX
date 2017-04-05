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

def version(session, format=None):
    '''Show version information.

    Parameters
    ----------
    format : one of 'verbose', ''bundles', or 'packages'
    '''
    from .. import buildinfo
    from chimerax import app_dirs as ad
    if format is None:
        session.logger.info("%s %s version: %s (%s)" % (ad.appauthor, ad.appname, ad.version, buildinfo.date.split()[0]))
        return
    session.logger.info("%s %s version: %s" % (ad.appauthor, ad.appname, ad.version))
    session.logger.info("date: %s" % buildinfo.date)
    session.logger.info("branch: %s" % buildinfo.branch)
    session.logger.info("commit: %s" % buildinfo.commit)
    import sys
    session.logger.info("Python: %s" % sys.version.split(maxsplit=1)[0])
    if format == 'verbose':
        return
    import os
    if format == 'bundles':
        dists = session.toolshed.bundle_info(session.logger,
                                             installed=True, available=False)
        dists = list(dists)
        dists.sort(key=lambda d: d.name.casefold())
    else:
        import pip
        dists = pip.get_installed_distributions(local_only=True)
        dists = list(dists)
        dists.sort(key=lambda d: d.project_name.casefold())
    if not dists:
        session.logger.error("no version information available")
        return os.EX_SOFTWARE
    if format == 'bundles':
        info = "Installed bundles:"
    else:
        info = "Installed packages:"
    if session.ui.is_gui:
        info += "\n<ul>"
        sep = "<li>"
        from html import escape
    else:
        sep = '  '

        def escape(txt):
            return txt

    for d in dists:
        if format == 'bundles':
            name = d.name
            version = d.version
            if name.startswith('ChimeraX-'):
                name = name[len('ChimeraX-'):]
        else:
            name = d.project_name
            if d.has_version():
                version = d.version
            else:
                version = "unknown"
        info += "\n%s %s: %s" % (sep, escape(name), escape(version))
    if session.ui.is_gui:
        info += "\n</ul>"
    session.logger.info(info, is_html=session.ui.is_gui)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        optional=[('format', cli.EnumOf(['verbose', 'bundles', 'packages']))],
        non_keyword=['format'],
        synopsis='show version information')
    cli.register('version', desc, version, logger=session.logger)
