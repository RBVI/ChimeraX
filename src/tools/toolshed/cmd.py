# vim: set expandtab ts=4 sw=4:

from chimerax.core.commands import EnumOf, CmdDesc, StringArg, BoolArg

_bundle_types = EnumOf(["all", "installed", "available"])


def _display_bundles(bi_list, logger):
    for bi in bi_list:
        logger.info(" %s (%s %s): %s" % (bi.display_name, bi.name,
                                         bi.version, bi.synopsis))


def ts_list(session, bundle_type="installed"):
    '''List installed bundles in the log.

    Parameters
    ----------
    bundle_type : string
      Types are "installed", "available", or "all"
    '''
    ts = session.toolshed
    logger = session.logger
    if bundle_type == "installed" or bundle_type == "all":
        bi_list = ts.bundle_info(installed=True, available=False)
        if bi_list:
            logger.info("List of installed bundles:")
            _display_bundles(bi_list, logger)
        else:
            logger.info("No installed bundles found.")
    if bundle_type in ("available", "all"):
        bi_list = ts.bundle_info(installed=False, available=True)
        if bi_list:
            logger.info("List of available bundles:")
            _display_bundles(bi_list, logger)
        else:
            logger.info("No available bundles found.")
ts_list_desc = CmdDesc(optional=[("bundle_type", _bundle_types)],
                       non_keyword=['bundle_type'])


def ts_refresh(session, bundle_type="installed"):
    '''
    Check for new bundles or new bundle vesions on server and display
    them in the toolshed window.

    Parameters
    ----------
    bundle_type : string
      Types are "installed", "available", or "all"
    '''
    ts = session.toolshed
    logger = session.logger
    if bundle_type == "installed":
        ts.reload(logger, rebuild_cache=True, check_remote=False)
    elif bundle_type == "available":
        ts.reload(logger, rebuild_cache=False, check_remote=True)
    elif bundle_type == "all":
        ts.reload(logger, rebuild_cache=True, check_remote=True)
ts_refresh_desc = CmdDesc(optional=[("bundle_type", _bundle_types)])


def _bundle_string(bundle_name, version):
    if version is None:
        return bundle_name
    else:
        return "%s (%s)" % (bundle_name, version)


def ts_install(session, bundle_name, user_only=True, version=None):
    '''
    Install a bundle.

    Parameters
    ----------
    bundle_name : string
    user_only : bool
      Install for this user only, or install for all users.
    version : string
    '''
    ts = session.toolshed
    logger = session.logger
    bi = ts.find_bundle(bundle_name, installed=True, version=version)
    if bi:
        logger.error("\"%s\" is already installed" % bundle_name)
        return
    bi = ts.find_bundle(bundle_name, installed=False, version=version)
    if bi is None:
        logger.error("\"%s\" does not match any bundles"
                     % _bundle_string(bundle_name, version))
        return
    ts.install_bundle(bi, logger, not user_only)
ts_install_desc = CmdDesc(required=[("bundle_name", StringArg)],
                          optional=[("user_only", BoolArg),
                                    ("version", StringArg)])


def ts_remove(session, bundle_name):
    '''
    Remove an installed bundle.

    Parameters
    ----------
    bundle_name : string
    '''
    ts = session.toolshed
    logger = session.logger
    bi = ts.find_bundle(bundle_name, installed=True)
    if bi is None:
        logger.error("\"%s\" does not match any bundles" % bundle_name)
        return
    ts.uninstall_bundle(bi, logger)
ts_remove_desc = CmdDesc(required=[("bundle_name", StringArg)])


def ts_update(session, bundle_name, version=None):
    '''
    Update a bundle to the latest version.

    Parameters
    ----------
    bundle_name : string
    version : string
    '''
    ts = session.toolshed
    logger = session.logger
    new_bi = ts.find_bundle(bundle_name, installed=False, version=version)
    if new_bi is None:
        logger.error("\"%s\" does not match any bundles"
                     % _bundle_string(bundle_name, version))
        return
    bi = ts.find_bundle(bundle_name, installed=True)
    if bi is None:
        logger.error("\"%s\" does not match any installed bundles" % bundle_name)
        return
    if (version is None and not new_bi.newer_than(bi) or
            new_bi.version == bi.version):
        logger.info("\"%s\" is up to date" % bundle_name)
        return
    ts.install_bundle(new_bi, logger)
ts_update_desc = CmdDesc(required=[("bundle_name", StringArg)],
                         optional=[("version", StringArg)])


#
# Commands that deal with GUI (singleton)
#

def ts_start(session, bundle_name):
    '''
    Start a tool in a bundle.

    Parameters
    ----------
    bundle_name : string
    '''
    ts = session.toolshed
    tinfo = ts.find_bundle(bundle_name)
    if tinfo is None:
        from chimerax.core.errors import UserError
        raise UserError('No installed bundle named "%s"' % bundle_name)
    tinfo.start(session)
ts_start_desc = CmdDesc(required=[('bundle_name', StringArg)])


def ts_show(session, bundle_name, _show=True):
    '''
    Show a tool panel, or start one if none is running.

    Parameters
    ----------
    bundle_name : string
    '''
    ts = session.toolshed
    tinfo = ts.find_bundle(bundle_name)
    if tinfo is None:
        from chimerax.core.errors import UserError
        raise UserError('No installed bundle named "%s"' % bundle_name)
    tinst = [t for t in session.tools.list() if t.bundle_info is tinfo]
    for ti in tinst:
        ti.display(_show)
    if len(tinst) == 0:
        tinfo.start(session)
ts_show_desc = CmdDesc(required=[('bundle_name', StringArg)])


def ts_hide(session, bundle_name):
    '''
    Hide tool panels.

    Parameters
    ----------
    bundle_name : string
    '''
    ts_show(session, bundle_name, _show=False)
ts_hide_desc = CmdDesc(required=[('bundle_name', StringArg)])
