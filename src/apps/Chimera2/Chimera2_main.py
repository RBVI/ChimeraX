# vim: set expandtab shiftwidth=4 softtabstop=4:
# Copyright © 2014 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
import sys
import os

__version__ = "0.1.0a0"     # version of this file -- PEP 440 compatible

app_name = "Chimera2"
app_author = "UCSF"
# remember locale codes are frequently different than country codes
localized_app_name = {
    'af': u'Chimera2',          # Afrikaans
    'cs': u'Přízrak2',          # Czech
    'da': u'Chiemra2',          # Danish
    'de': u'Chimäre2',          # German
    'el': u'Χίμαιρα2',          # Greek
    'en': u'Chimera2',          # English
    'es': u'Quimera2',          # Spanish
    'fi': u'Kauhukuva2',        # Finish
    'fr': u'Chimère2',          # French
    'hr': u'Himera2',           # Croatian
    'in': u'Angan-angan2',      # Indonesian
    'it': u'Chimera2',          # Italian
    'ja': u'キメラ2',           # Japanese
    'ko': u'키메라2',           # Korean
    'nl': u'Chimera2',          # Dutch
    'no': u'Chimera2',          # Norwegian
    'pl': u'Chimera2',          # Polish
    'pt': u'Quimera2',          # Portuguese
    'ro': u'Himeră2',           # Romainian
    'ru': u'Химера2',           # Russian
    'sr': u'Химера2',           # Serbian
    'sk': u'Prízrak2',          # Slovak
    'sv': u'Chimera2',          # Swedish
    'th': u'ความเพ้อฝัน2',        # Thai
    'tr': u'Kuruntu2',          # Turkish
    'uk': u'Химера2',           # Ukrainian
    'zh': u'嵌合體2',           # Chinese
}


def parse_arguments(argv):
    """Initialize Chimera application."""
    import getopt

    if sys.platform.startswith('darwin'):
        # skip extra -psn_ argument on Mac OS X 10.8 and earlier
        import platform
        release = platform.mac_ver()[0]
        if release:
            release = [int(x) for x in release.split('.')]
            if release < [10, 9]:
                for i, arg in enumerate(argv):
                    if i == 0:
                        continue
                    if arg.startswith('-psn_'):
                        del argv[i]
                        break

    class Opts:
        pass
    opts = Opts()
    opts.debug = False
    opts.gui = True
    opts.module = None
    opts.line_profile = False
    opts.list_file_types = False
    opts.load_tools = True
    opts.silent = False
    opts.status = True
    opts.stereo = False
    opts.uninstall = False
    opts.use_defaults = False
    opts.version = 0

    # Will build usage string from list of arguments
    arguments = [
        "--debug",
        "--nogui",
        "--help",
        "--lineprofile",
        "--listfiletypes",
        "--silent",
        "--nostatus",
        "--notools",
        "--stereo",
        "--uninstall",
        "--usedefaults",
        "--version",
    ]
    if sys.platform.startswith("win"):
        arguments += ["--console", "--noconsole"]
    usage = '[' + "] [".join(arguments) + ']'
    usage += " or -m module_name [args]"
    # add in default argument values
    arguments += [
        "--nodebug",
        "--gui",
        "--nolineprofile",
        "--nosilent",
        "--nousedefaults",
        "--status",
        "--tools",
    ]
    if len(sys.argv) > 2 and sys.argv[1] == '-m':
        # treat like Python's -m argument
        opts.gui = False
        opts.silent = True
        opts.module = sys.argv[2]
        return opts, sys.argv[2:]
    try:
        shortopts = ""
        longopts = []
        for a in arguments:
            if a.startswith("--"):
                i = a.find(' ')
                if i == -1:
                    longopts.append(a[2:])
                else:
                    longopts.append(a[2:i] + '=')
            elif a.startswith('-'):
                i = a.find(' ')
                if i == -1:
                    shortopts += a[1]
                else:
                    shortopts += a[1] + ':'
        options, args = getopt.getopt(argv[1:], shortopts, longopts)
    except getopt.error as message:
        print("%s: %s" % (argv[0], message), file=sys.stderr)
        print("usage: %s %s\n" % (argv[0], usage), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)

    help = False
    for opt, optarg in options:
        if opt in ("--debug", "--nodebug"):
            opts.debug = opt[2] == 'd'
        elif opt == "--help":
            help = True
        elif opt in ("--gui", "--nogui"):
            opts.gui = opt[2] == 'g'
        elif opt in ("--lineprofile", "--nolineprofile"):
            opts.line_profile = opt[2] == 'l'
        elif opt == "--listfiletypes":
            opts.list_file_types = True
        elif opt in ("--silent", "--nosilent"):
            opts.silent = opt[2] == 's'
        elif opt in ("--status", "--nostatus"):
            opts.status = opt[2] == 's'
        elif opt in "--stereo":
            opts.stereo = True
        elif opt in ("--tools", "--notools"):
            opts.load_tools = opt[2] == 't'
        elif opt == "--uninstall":
            opts.uninstall = True
        elif opt in ("--usedefaults", "--nousedefaults"):
            opts.load_tools = opt[2] == 'u'
        elif opt == "--version":
            opts.version += 1
    if help:
        print("usage: %s %s\n" % (argv[0], usage), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)
    if opts.version or opts.list_file_types:
        opts.gui = False
        opts.silent = True
    return opts, args


def init(argv, event_loop=True):
    if sys.platform.startswith('darwin'):
        paths = os.environ['PATH'].split(':')
        if '/usr/sbin' not in paths:
            # numpy, numexpr, and pytables need sysctl in path
            paths.append('/usr/sbin')
            os.environ['PATH'] = ':'.join(paths)
        del paths

    # use chimera.core's version
    import pip
    dists = pip.get_installed_distributions(local_only=True)
    for d in dists:
        if d.key == 'chimera.core':
            version = d.version
            break
    else:
        print("error: unable to figure out %s's version" % app_name)
        return os.EX_SOFTWARE

    opts, args = parse_arguments(argv)

    # install line_profile decorator
    import builtins
    if not opts.line_profile:
        builtins.__dict__['line_profile'] = lambda x: x
    else:
        # write profile results on exit
        import atexit
        import line_profiler
        prof = line_profiler.LineProfiler()
        builtins.__dict__['line_profile'] = prof
        atexit.register(prof.dump_stats, "%s.lprof" % app_name)

    if opts.use_defaults:
        from chimera.core import configinfo
        configinfo.only_use_defaults = True

    if not opts.gui:
        # Flag to configure off-screen rendering before PyOpenGL imported
        from chimera import core
        core.offscreen_rendering = True

    # figure out the user/system directories for application
    executable = os.path.abspath(sys.argv[0])
    bindir = os.path.dirname(executable)
    if sys.platform.startswith('linux'):
        if os.path.basename(bindir) == "bin":
            configdir = os.path.dirname(bindir)
        else:
            configdir = bindir
        os.environ['XDG_CONFIG_DIRS'] = configdir

    from distlib.version import NormalizedVersion as Version
    epoch, ver, *_ = Version(version).parse(version)
    if len(ver) == 1:
        ver += (0,)
    partial_version = '%s.%s' % (ver[0], ver[1])

    import appdirs
    ad = appdirs.AppDirs(app_name, appauthor=app_author,
                         version=partial_version)
    # make sure app_dirs.user_* directories exist
    for var, name in (
            ('user_data_dir', "user's data"),
            ('user_config_dir', "user's configuration"),
            ('user_cache_dir', "user's cache")):
        dir = getattr(ad, var)
        try:
            os.makedirs(dir, exist_ok=True)
        except OSError as e:
            print("Unable to make %s directory: %s: %s" % (
                name, e.strerror, e.filename), file=sys.stderr)
            raise SystemExit(1)

    # app_dirs_unversioned is primarily for caching data files that will
    # open in any version
    # app_dirs_unversioned.user_* directories are parents of those in app_dirs
    adu = appdirs.AppDirs(app_name, appauthor=app_author)

    # Find the location of "share" directory so that we can inform
    # the C++ layer.  Assume it's a sibling of the directory that
    # the executable is in.
    rootdir = os.path.dirname(bindir)
    import chimera
    chimera.app_data_dir = os.path.join(rootdir, "share")
    chimera.app_bin_dir = os.path.join(rootdir, "bin")
    chimera.app_lib_dir = os.path.join(rootdir, "lib")

    # inform the C++ layer of the appdirs paths
    from chimera.core import _appdirs
    _appdirs.init_paths(os.sep, ad.user_data_dir, ad.user_config_dir,
                        ad.user_cache_dir, ad.site_data_dir,
                        ad.site_config_dir, ad.user_log_dir,
                        chimera.app_data_dir, adu.user_cache_dir)

    from chimera.core import session
    sess = session.Session(app_name, debug=opts.debug)
    sess.app_dirs = ad
    sess.app_dirs_unversioned = adu

    from chimera.core import core_settings
    core_settings.init(sess)

    session.common_startup(sess)

    if opts.uninstall:
        return uninstall(sess)

    # initialize the user interface
    if opts.gui:
        from chimera.core.ui import gui
        ui_class = gui.UI
    else:
        from chimera.core.ui import nogui
        ui_class = nogui.UI
    # sets up logging, splash screen if gui
    # calls "sess.save_in_session(self)"
    sess.ui = ui_class(sess)
    sess.ui.stereo = opts.stereo
    # splash step "0" will happen in the above initialization
    num_splash_steps = 2
    if opts.gui:
        num_splash_steps += 1
    if not opts.gui and opts.load_tools:
        num_splash_steps += 1
    import itertools
    splash_step = itertools.count()

    # common core initialization
    if not opts.silent:
        sess.ui.splash_info("Initializing core",
                            next(splash_step), num_splash_steps)
        if sess.ui.is_gui and opts.debug:
            print("Initializing core", flush=True)

    if not opts.silent:
        sess.ui.splash_info("Initializing tools",
                            next(splash_step), num_splash_steps)
        if sess.ui.is_gui and opts.debug:
            print("Initializing tools", flush=True)
    from chimera.core import toolshed
    # toolshed.init returns a singleton so it's safe to call multiple times
    sess.toolshed = toolshed.init(sess.logger, sess.app_dirs, debug=sess.debug)
    from chimera.core import tools
    sess.add_state_manager('tools', tools.Tools(sess, first=True))  # access with sess.tools
    from chimera.core import tasks
    sess.add_state_manager('tasks', tasks.Tasks(sess, first=True))  # access with sess.tasks

    if opts.version:
        print("%s: %s" % (app_name, version))
        if opts.version == 1:
            return os.EX_OK
        import pip
        dists = pip.get_installed_distributions(local_only=True)
        if not dists:
            sess.logger.error("no version information available")
            return os.EX_SOFTWARE
        dists = list(dists)
        dists.sort(key=lambda d: d.key)
        if opts.version == 2:
            print("Installed tools:")
        else:
            print("Installed packages:")
        for d in dists:
            key = d.key
            if opts.version == 2:
                if not key.startswith('chimera.'):
                    continue
                key = key[len('chimera.'):]
            if d.has_version():
                print("    %s: %s" % (key, d.version))
            else:
                print("    %s: unknown" % key)
        return os.EX_OK

    if opts.list_file_types:
        from chimera.core import io
        io.print_file_types()
        raise SystemExit(0)

    if sys.platform.startswith('linux'):
        from chimera.core import _xdg
        _xdg.install_if_needed(sess, localized_app_name)

    if opts.gui:
        # build out the UI, populate menus, create graphics, etc.
        if not opts.silent:
            sess.ui.splash_info("Starting main interface",
                                next(splash_step), num_splash_steps)
            if sess.ui.is_gui and opts.debug:
                print("Starting main interface", flush=True)
        sess.ui.build()

    if opts.load_tools:
        if not opts.silent:
            sess.ui.splash_info("Loading autostart tools",
                                next(splash_step), num_splash_steps)
            if sess.ui.is_gui and opts.debug:
                print("Loading autostart tools", flush=True)
        sess.tools.autostart()

    if not opts.silent:
        sess.ui.splash_info("Finished initialization",
                            next(splash_step), num_splash_steps)
        if sess.ui.is_gui and opts.debug:
            print("Finished initialization", flush=True)

    if opts.gui:
        sess.ui.close_splash()
        sess.logger.info('OpenGL ' + sess.main_view.opengl_version())

    if opts.module:
        import runpy
        import warnings
        sys.argv[:] = args  # runpy will insert appropriate argv[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=BytesWarning)
            global_dict = {
                'session': sess
            }
            runpy.run_module(opts.module, init_globals=global_dict,
                             run_name='__main__', alter_sys=True)
        return os.EX_OK

    # the rest of the arguments are data files
    from chimera.core import errors
    for arg in args:
        try:
            sess.models.open(arg)
        except (IOError, errors.UserError) as e:
            sess.logger.error(str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return os.EX_SOFTWARE

    # Allow the event_loop to be disabled, so we can be embedded in
    # another application
    if event_loop:
        try:
            sess.ui.event_loop()
        except SystemExit as e:
            return e.code
        except Exception as e:
            import traceback
            traceback.print_exc()
            return os.EX_SOFTWARE
    return os.EX_OK


def rm_rf_path(path, sess):
    # analogous to "rm -rf path"
    import shutil
    had_error = [False]

    def found_error(function, path, excinfo):
        had_error[0] = True

    shutil.rmtree(path, onerror=found_error)
    if had_error[0]:
        sess.logger.warning("unable to completely remove '%s'" % path)


def uninstall(sess):
    # for uninstall option
    import tempfile
    # change directory so we're guaranteed not to be in the Chimera2 app
    os.chdir(tempfile.gettempdir())

    # find location of Chimera2
    if sys.executable is None:
        sess.logger.error('unable to locate Chimera2 executable')
        return os.EX_SOFTWARE
    exe = os.path.realpath(sys.executable)
    exe_dir = os.path.dirname(exe)

    if sys.platform.startswith('linux'):
        if os.path.basename(exe_dir) != 'bin':
            sys.logger.error('non-standard Chimera2 installation')
            return os.EX_SOFTWARE
        from chimera.core import _xdg
        _xdg.uninstall(sess)
        #parent = os.path.dirname(exe_dir)
        #rm_rf_path(parent, sess)
        return os.EX_OK

    if sys.platform.startswith('darwin'):
        if os.path.basename(exe_dir) != 'MacOS':
            sess.logger.error('non-standard Chimera2 installation')
            return os.EX_SOFTWARE
        parent = os.path.dirname(exe_dir)
        if os.path.basename(parent) != 'Contents':
            sess.logger.error('non-standard Chimera2 installation')
            return os.EX_SOFTWARE
        parent = os.path.dirname(parent)
        if not os.path.basename(parent).endswith('.app'):
            sess.logger.error('non-standard Chimera2 installation')
            return os.EX_SOFTWARE
        rm_rf_path(parent, sess)
        return os.EX_OK

    sess.logger.error('can not yet uninstall on %s' % sys.platform)
    return os.EX_UNAVAILABLE

if __name__ == '__main__':
    raise SystemExit(init(sys.argv))
