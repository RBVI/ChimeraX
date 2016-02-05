# vim: set expandtab shiftwidth=4 softtabstop=4:


def version(session, format=None):
    '''Show version information.

    Parameters
    ----------
    format : one of 'verbose', ''bundles', or 'package'
    '''
    from .. import buildinfo
    ad = session.app_dirs
    if format is None:
        session.logger.info("%s %s version: %s (%s)" % (ad.appauthor, ad.appname, ad.version, buildinfo.date.split()[0]))
        return
    session.logger.info("%s %s version: %s" % (ad.appauthor, ad.appname, ad.version))
    session.logger.info("date: %s" % buildinfo.date)
    session.logger.info("branch: %s" % buildinfo.branch)
    session.logger.info("commit: %s" % buildinfo.commit)
    if format == 'verbose':
        return
    import os
    import pip
    dists = pip.get_installed_distributions(local_only=True)
    if not dists:
        session.logger.error("no version information available")
        return os.EX_SOFTWARE
    dists = list(dists)
    dists.sort(key=lambda d: d.key)
    if format == 'bundles':
        info = "Installed bundles:"
    else:
        info ="Installed packages:"
    if session.ui.is_gui:
        info += "\n<ul>"
        sep = "<li>"
        from html import escape
    else:
        sep = '  '
        def escape(txt):
            return txt
    for d in dists:
        key = d.key
        if format == 'bundles':
            if not key.startswith('chimerax.'):
                continue
            key = key[len('chimerax.'):]
        if d.has_version():
            info += "\n%s %s: %s" % (sep, escape(key), escape(d.version))
        else:
            info += "\n%s %s: unknown" % (sep, escape(key))
    if session.ui.is_gui:
        info += "\n</ul>"
    session.logger.info(info, is_html=session.ui.is_gui)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        optional=[('format', cli.EnumOf(['verbose', 'bundles', 'package']))],
        non_keyword=['format'],
        synopsis='show version information')
    cli.register('version', desc, version)
