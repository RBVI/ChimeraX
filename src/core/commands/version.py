# vim: set expandtab shiftwidth=4 softtabstop=4:


def version(session, format='terse'):
    '''Show version information.

    Parameters
    ----------
    format : one of 'terse', ''bundles', or 'package'
    '''
    ad = session.app_dirs
    session.logger.info("%s %s version: %s" % (ad.appauthor, ad.appname, ad.version))
    from .. import buildinfo
    session.logger.info("date: %s" % buildinfo.date)
    session.logger.info("branch: %s" % buildinfo.branch)
    session.logger.info("commit: %s" % buildinfo.commit)
    if format == 'terse':
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
        session.logger.info("Installed bundles:")
    else:
        session.logger.info("Installed packages:")
    for d in dists:
        key = d.key
        if format == 'bundles':
            if not key.startswith('chimerax.'):
                continue
            key = key[len('chimerax.'):]
        if d.has_version():
            session.logger.info("    %s: %s" % (key, d.version))
        else:
            session.logger.info("    %s: unknown" % key)


def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        optional=[('format', cli.EnumOf(['terse', 'bundles', 'package']))],
        non_keyword=['format'],
        synopsis='show version information')
    cli.register('version', desc, version)
