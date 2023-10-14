# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
"""
ChimeraX Toolshed Utilities

This package includes parts of the toolshed code that are for
installing and uninstalling bundles.  It is separated out of the core
for ease of updating outside of the core release cycle.

Everything in here is considered private.
"""

__version__ = "1.2.4"

from chimerax.core.toolshed import (
    TOOLSHED_BUNDLE_INSTALLED, TOOLSHED_BUNDLE_UNINSTALLED,
    ToolshedInstalledError,
    BundleAPI, BundleInfo,
    _ChimeraXNamespace
)


class _BootstrapAPI(BundleAPI):

    @staticmethod
    def initialize(session, bundle_info):
        # 'initialize' is called by the toolshed on start up
        from . import cmd
        from chimerax.core import toolshed
        cmd.register_command(session.logger)

        def show_updates(trigger_name, data, *, session=session):
            from . import tool
            session.ui.thread_safe(tool.show, session, tool.OUT_OF_DATE)
        session.toolshed.triggers.add_handler(toolshed.TOOLSHED_OUT_OF_DATE_BUNDLES, show_updates)

    @staticmethod
    def start_tool(session, tool_name):
        from .tool import show
        return show(session)

    @staticmethod
    def get_class(class_name):
        return None


bundle_api = _BootstrapAPI()


def _debug(*args, **kw):
    pass


def _import_bundle(toolshed, bundle_name, logger, install="ask", session=None):
    """Return the module for the bundle with the given name.

    Parameters
    ----------
    bundle_name : str
        Name (internal or display name) of the bundle of interest.
    logger : :py:class:`~chimerax.core.logger.Logger` instance
        Logging object where warning and error messages are sent.
    install: str
        Action to take if bundle is uninstalled but available.
        "ask" (default) means to ask user, if `session` is not `None`;
        "never" means not to install; and
        "always" means always install.
    session : :py:class:`chimerax.core.session.Session` instance.
        Session that is requesting the module.  Defaults to `None`.

    Raises
    ------
    ImportError
        Raised if a module for the bundle cannot be found.
    """
    # If the bundle is installed, return its module.
    bundle = toolshed.find_bundle(bundle_name, logger, installed=True)
    if bundle is not None:
        module = bundle.get_module()
        if module is None:
            raise ImportError("bundle %r has no module" % bundle_name)
        return module
    bundle = toolshed.find_bundle(bundle_name, logger, installed=False)
    if bundle is None:
        raise ImportError("bundle %r not found" % bundle_name)
    return _install_module(bundle, logger, install, session)


def _import_package(toolshed, package_name, logger, install=None, session=None):
    """package of given name if it is associated with a bundle.

    Parameters
    ----------
    module_name : str
        Name of the module of interest.
    logger : :py:class:`~chimerax.core.logger.Logger` instance
        Logging object where warning and error messages are sent.
    install: str
        Action to take if bundle is uninstalled but available.
        "ask" (default) means to ask user, if `session` is not `None`;
        "never" means not to install; and
        "always" means always install.
    session : :py:class:`chimerax.core.session.Session` instance.
        Session that is requesting the module.  Defaults to `None`.

    Raises
    ------
    ImportError
        Raised if a module for the bundle cannot be found.
    """
    for bi in toolshed._installed_bundle_info:
        if bi.package_name == package_name:
            module = bi.get_module()
            if module is None:
                raise ImportError("bundle %r has no module" % package_name)
            return module
    # No installed bundle matches
    from pkg_resources import parse_version
    best_bi = None
    best_version = None
    for bi in toolshed._get_available_bundles(logger):
        if bi.package_name != package_name:
            continue
        if best_bi is None:
            best_bi = bi
            best_version = parse_version(bi.version)
        elif best_bi.name != bi.name:
            raise ImportError("%r matches multiple bundles %s, %s" % (
                package_name, best_bi.name, bi.name))
        else:
            v = parse_version(bi.version)
            if v > best_version:
                best_bi = bi
                best_version = v
    if best_bi is None:
        raise ImportError("bundle %r not found" % package_name)
    return _install_module(best_bi, logger, install, session)


def _install_bundle(toolshed, bundles, logger, *, per_user=True, reinstall=False, session=None,
                    no_deps=False):
    """Install the bundle(s) either by retrieving them from the remote toolshed
    or by from a local wheel.

    Parameters
    ----------
    bundles : string or :py:class:`BundleInfo` instance or sequence of them
        If string, path to wheel installer.
        If instance, should be from the available bundle list.
    per_user : boolean
        True to install bundle only for the current user (default);
        False to install for everyone.
    reinstall : boolean
        True to force reinstall package.
    logger : :py:class:`~chimerax.core.logger.Logger` instance
        Logging object where warning and error messages are sent.

    Raises
    ------
    ToolshedInstalledError
        Raised if a bundle is already installed.

    Notes
    -----
    A :py:const:`TOOLSHED_BUNDLE_INSTALLED` trigger is fired after installation.
    """
    _debug("install_bundle", bundles)
    import os

    # Make sure that our install location is on chimerax module.__path__
    # so that newly installed modules may be found
    import importlib
    import re
    if per_user is None:
        per_user = True
    cx_dir = os.path.join(toolshed._site_dir, _ChimeraXNamespace)
    m = importlib.import_module(_ChimeraXNamespace)
    if cx_dir not in m.__path__:
        m.__path__.append(cx_dir)
    if isinstance(bundles, (str, BundleInfo)):
        bundles = [bundles]
    bundle_names = []
    # TODO: redo this to collect bundles to install and then revise _can_install to
    # check all bundles together.
    # TODO: expose new _can_install, so gui can use it
    all_install_now = True
    for bundle in bundles:
        old_bundle = None
        if isinstance(bundle, str):
            # If the name ends with .whl, it must be a path.
            if not bundle.endswith(".whl"):
                raise ValueError("Can only install wheel files")
            bundle_path = os.path.expanduser(bundle)
            if not os.path.exists(bundle_path):
                raise ValueError("Non-existent wheel file")
            basename = os.path.split(bundle_path)[1]
            name = basename.split('-')[0]
            old_bundle = toolshed.find_bundle(name, logger, installed=True)
            bundle_name = bundle
            from wheel_filename import parse_wheel_filename
            bundle_version = parse_wheel_filename(bundle).version
        elif isinstance(bundle, BundleInfo):
            # If "bundle" is not a string, it must be a Bundle instance.
            old_bundle = toolshed.find_bundle(bundle.name, logger, installed=True)
            bundle_name = bundle
            bundle_version = bundle.version
        else:
            raise ValueError("incorrect bundle argument")
        if old_bundle:
            if not reinstall and bundle_version == old_bundle.version:
                if isinstance(bundle, BundleInfo):
                    bundle_name = bundle.name
                else:
                    bundle_name = bundle
                raise ToolshedInstalledError("bundle %r already installed" % bundle_name)
            install_now = _can_install(old_bundle, session)
            if install_now:
                old_bundle.deregister(logger)
                toolshed._installed_bundle_info.remove(old_bundle)
            else:
                all_install_now = False
        bundle_names.append(bundle_name)
    if not all_install_now:
        logger.info("Deferring bundle installation")
        args = []
        if per_user:
            args.append("--user")
        if no_deps:
            args.append("--no-deps")
        message = "A bundle is currently in use.  ChimeraX must be restarted to finish installing."
        _add_restart_action("install", bundle_names, args, logger, message, session)
        return
    from chimerax.core.commands import plural_form
    logger.status("Installing %s" % plural_form(bundle_names, "bundle"))
    try:
        results = _pip_install(
            toolshed, bundle_names, logger, per_user=per_user, reinstall=reinstall, no_deps=no_deps)
    except PermissionError:
        who = "everyone" if not per_user else "this account"
        logger.error("You do not have permission to install %s for %s" %
                     (bundle_name, who))
        return
    installed = re.findall(r"^\s*Successfully installed.*$", results, re.M)
    if installed:
        logger.info('\n'.join(installed))
    else:
        logger.info('No bundles were installed')
    toolshed.set_install_timestamp(per_user)
    changes = toolshed.reload(logger, rebuild_cache=True, report=True)

    if not toolshed._safe_mode:
        # Initialize managers, notify other managers about newly
        # installed providers, and call custom init.
        # There /may/ be a problem with the order in which we call
        # these if multiple bundles were installed, but we hope for
        # the best.  We do /not/ call initialization functions for
        # bundles that were just updated because we do not want to
        # confuse already initialized bundles.
        try:
            new_bundles = changes["installed"]
        except KeyError:
            pass
        else:
            # managers
            failed = []
            done = set()
            initializing = set()
            for name, version in new_bundles.items():
                bi = toolshed.find_bundle(name, logger, version=version)
                if bi:
                    toolshed._init_bundle_manager(
                        session, bi, done, initializing, failed)
            for name in failed:
                logger.warning("%s: manager initialization failed" % name)

            # providers
            ends_needed = set()
            for name, version in new_bundles.items():
                bi = toolshed.find_bundle(name, logger, version=version)
                if bi:
                    for name, kw in bi.providers.items():
                        mgr_name, pvdr_name = name.split('/', 1)
                        mgr = toolshed._manager_instances.get(mgr_name, None)
                        if mgr:
                            mgr.add_provider(bi, pvdr_name, **kw)
                            ends_needed.add(mgr)
            for mgr in ends_needed:
                mgr.end_providers()

            # custom inits
            failed = []
            done = set()
            initializing = set()
            for name, version in new_bundles.items():
                bi = toolshed.find_bundle(name, logger, version=version)
                if bi:
                    toolshed._init_bundle_custom(
                        session, bi, done, initializing, failed)
            for name in failed:
                logger.warning("%s: custom initialization failed" % name)

    toolshed.triggers.activate_trigger(TOOLSHED_BUNDLE_INSTALLED, bundle_name)


def _can_install(bi, session=None):
    """Check if bundle can be installed (i.e., not in use)."""
    # A custom init means it's currently in use
    if session and not session.minimal and bi.custom_init:
        return False
    # A bundle can be installed if its own package is not in use
    # and does not pull in any dependent bundle that is in use.
    if not bi.imported():
        return True
    # TODO: Figuring out the latter is hard, so we ignore it for now.
    # TODO: possible stragegy: look through bundle dependencies and
    # see that all dependencies on an installed bundle are satisfied
    # TODO: need dependencies to be provided when querying toolshed for
    # available bundles
    return False


def _can_uninstall(bi):
    """Check if bundle can be uninstalled (i.e., not in use)."""
    # A bundle can be uninstalled if it has no library/shared object/DLL
    # loaded.  That is hard to tell, so we err on the side of caution.
    return not bi.imported()


def _add_restart_action(action_type, bundles, extra_args, logger, message, session):
    # Show user a dialog (hence error) so they know something happened.
    # Append to on_restart file so bundle action is done on restart.
    import os
    from chimerax.core.toolshed import restart_action_info
    inst_dir, restart_file = restart_action_info()
    try:
        os.makedirs(inst_dir)
    except FileExistsError:
        pass
    with open(restart_file, "a") as f:
        if action_type == "install" and session is not None:
            print(f"toolshed_url\t{session.toolshed.remote_url}", file=f)
        args = [action_type]
        bundle_args = []
        for bundle in bundles:
            if not isinstance(bundle, str):
                # Must be a BundleInfo instance
                bundle_args.append(f"{bundle.name}=={bundle.version}")
            else:
                # Must be a file
                import shutil
                shutil.copy(bundle, inst_dir)
                bundle_args.append(os.path.split(bundle)[1])
        args.append(' '.join(bundle_args))
        args.extend(extra_args)
        print("\t".join(args), file=f)
    if session is None or not session.ui.is_gui:
        logger.error(message)
    else:
        from Qt.QtWidgets import QMessageBox
        msg_box = QMessageBox(QMessageBox.Question, "Restart ChimeraX?", message)
        msg_box.setInformativeText("Do you want to restart now?")
        yes = msg_box.addButton("Restart Now", QMessageBox.AcceptRole)
        no = msg_box.addButton("Restart Later", QMessageBox.RejectRole)
        msg_box.setDefaultButton(no)
        msg_box.setEscapeButton(no)
        msg_box.exec()
        if msg_box.clickedButton() == yes:
            import sys
            import os
            try:
                os.execv(sys.executable, [sys.executable])
            except Exception as err:
                logger.error("Unable to restart ChimeraX: %s" % err)


def _uninstall_bundle(toolshed, bundle, logger, *, session=None, force_remove=False):
    """Supported API. Uninstall bundle by removing the corresponding Python distribution.

    Parameters
    ----------
    bundle : string or :py:class:`BundleInfo` instance or sequence of them
        If string, path to wheel installer.
        If instance, should be from the available bundle list.
    logger : :py:class:`~chimerax.core.logger.Logger` instance
        Logging object where warning and error messages are sent.

    Raises
    ------
    ToolshedInstalledError
        Raised if the bundle is not installed.

    Notes
    -----
    A :py:const:`TOOLSHED_BUNDLE_UNINSTALLED` trigger is fired after package removal.
    """
    import re
    _debug("uninstall_bundle", bundle)
    if isinstance(bundle, (str, BundleInfo)):
        bundles = [bundle]
    else:
        bundles = bundle
    uninstall_now = []
    uninstall_later = []
    for bundle in bundles:
        if isinstance(bundle, str):
            bundle = toolshed.find_bundle(bundle, logger, installed=True)
        if bundle is None or not bundle.installed:
            raise ToolshedInstalledError("bundle %r not installed" % bundle.name)
        if _can_uninstall(bundle):
            uninstall_now.append(bundle)
        else:
            uninstall_later.append(bundle)
    if not force_remove:
        all_bundles = set()
        all_bundles.update(uninstall_now)
        all_bundles.update(uninstall_later)
        for bi in all_bundles:
            needed_by = bi.dependents(logger)
            needed_by -= all_bundles
            if needed_by:
                from chimerax.core.commands import commas, plural_form
                other = plural_form(needed_by, "another", "other")
                bundles = plural_form(needed_by, "bundles")
                logger.error("Unable to uninstall %s because it is needed by %s %s: %s" % (
                    bi.name, other, bundles, commas((bi.name for bi in needed_by), 'and')))
                return
    if uninstall_now:
        for bundle in uninstall_now:
            bundle.deregister(logger)
            bundle.unload(logger)
        results = _pip_uninstall(uninstall_now, logger)
        uninstalled = re.findall(r"^\s*Successfully uninstalled.*$", results, re.M)
        if uninstalled:
            logger.info('\n'.join(uninstalled))
        toolshed.reload(logger, rebuild_cache=True, report=True)
        toolshed.triggers.activate_trigger(TOOLSHED_BUNDLE_UNINSTALLED, bundle)
    if uninstall_later:
        for bundle in uninstall_later:
            bundle.deregister(logger)
            bundle.unload(logger)
        message = "ChimeraX must be restarted to finish uninstalling."
        _add_restart_action("uninstall", uninstall_later, [], logger, message, session)


def _install_module(toolshed, bundle, logger, install, session):
    # Given a bundle name and *uninstalled* bundle, install it
    # and return the module from the *installed* bundle
    if install == "never":
        raise ImportError("bundle %r is not installed" % bundle.name)
    if install == "ask":
        if session is None:
            raise ImportError("bundle %r is not installed" % bundle.name)
        from chimerax.ui.ask import ask
        answer = ask(session, "Install bundle %r?" % bundle.name,
                     buttons=["install", "cancel"])
        if answer == "cancel":
            raise ImportError("user canceled installation of bundle %r" % bundle.name)
        elif answer == "install":
            per_user = True
        else:
            raise ImportError("installation of bundle %r canceled" % bundle.name)
    # We need to install the bundle.
    toolshed.install_bundle(bundle.name, logger, per_user=per_user)
    # Now find the *installed* bundle.
    bundle = toolshed.find_bundle(bundle.name, logger, installed=True)
    if bundle is None:
        raise ImportError("could not install bundle %r" % bundle.name)
    module = bundle.get_module()
    if module is None:
        raise ImportError("bundle %r has no module" % bundle.name)
    return module


def _pip_install(toolshed, bundles, logger, per_user=True, reinstall=False, no_deps=False):
    # Run "pip" with our standard arguments (index location, update
    # strategy, etc) plus the given arguments.  Return standard
    # output as string.  If there was an error, raise RuntimeError
    # with stderr as parameter.
    command = [
        "install", "-qq", "--extra-index-url", toolshed.remote_url + "/pypi/",
        "--upgrade-strategy", "only-if-needed", "--no-warn-script-location",
        # "--only-binary", ":all:"   # msgpack-python is not binary
    ]
    if per_user:
        command.append("--user")
    if no_deps:
        command.append("--no-deps")
    if reinstall:
        # XXX: Not sure how this interacts with "only-if-needed"
        # For now, prevent --force-reinstall from reinstalling dependencies
        command.extend(["--force-reinstall", "--no-deps"])
    # bundles can be either a file path or a bundle name in repository or a list of them
    if isinstance(bundles, str):
        command.append(bundles)
    else:
        for bundle in bundles:
            if not isinstance(bundle, str):
                bundle = f"{bundle.name}=={bundle.version}"
            command.append(bundle)
    try:
        from chimerax.core.python_utils import run_logged_pip
        results = run_logged_pip(command, logger)
    except (RuntimeError, PermissionError) as e:
        from chimerax.core.errors import UserError
        raise UserError(str(e))
    # _remove_scripts()
    return results


def _pip_uninstall(bundles, logger):
    # Run "pip" and return standard output as string.  If there
    # was an error, raise RuntimeError with stderr as parameter.
    command = ["uninstall", "--yes"]
    command.extend(bundle.name for bundle in bundles)
    from chimerax.core.python_utils import run_logged_pip
    return run_logged_pip(command, logger)


def _remove_scripts():
    # remove pip installed scripts since they have hardcoded paths to
    # python and thus don't work when ChimeraX is installed elsewhere
    from chimerax import app_bin_dir
    import os
    import sys
    if sys.platform.startswith('win'):
        # Windows
        script_dir = os.path.join(app_bin_dir, 'Scripts')
        for dirpath, dirnames, filenames in os.walk(script_dir, topdown=False):
            for f in filenames:
                path = os.path.join(dirpath, f)
                os.remove(path)
            os.rmdir(dirpath)
    else:
        # Linux, Mac OS X
        for filename in os.listdir(app_bin_dir):
            path = os.path.join(app_bin_dir, filename)
            if not os.path.isfile(path):
                continue
            with open(path, 'br') as f:
                line = f.readline()
                if line[0:2] != b'#!' or b'/bin/python' not in line:
                    continue
            # print('removing (pip installed)', path)
            os.remove(path)
