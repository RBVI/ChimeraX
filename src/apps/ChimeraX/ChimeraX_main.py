# vim: set expandtab shiftwidth=4 softtabstop=4:
# Copyright © 2015-2016 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
import sys
import os

__version__ = "0.1.0a0"     # version of this file -- PEP 440 compatible

app_name = "ChimeraX"
app_author = "UCSF"
# remember locale codes are frequently different than country codes
localized_app_name = {
    'af': u'ChimeraX',          # Afrikaans
    'cs': u'PřízrakX',          # Czech
    'da': u'ChiemraX',          # Danish
    'de': u'ChimäreX',          # German
    'el': u'ΧίμαιραX',          # Greek
    'en': u'ChimeraX',          # English
    'es': u'QuimeraX',          # Spanish
    'fi': u'KauhukuvaX',        # Finish
    'fr': u'ChimèreX',          # French
    'hr': u'HimeraX',           # Croatian
    'in': u'Angan-anganX',      # Indonesian
    'it': u'ChimeraX',          # Italian
    'ja': u'キメラX',           # Japanese
    'ko': u'키메라X',           # Korean
    'nl': u'ChimeraX',          # Dutch
    'no': u'ChimeraX',          # Norwegian
    'pl': u'ChimeraX',          # Polish
    'pt': u'QuimeraX',          # Portuguese
    'ro': u'HimerăX',           # Romainian
    'ru': u'ХимераX',           # Russian
    'sr': u'ХимераX',           # Serbian
    'sk': u'PrízrakX',          # Slovak
    'sv': u'ChimeraX',          # Swedish
    'th': u'ความเพ้อฝันX',        # Thai
    'tr': u'KuruntuX',          # Turkish
    'uk': u'ХимераX',           # Ukrainian
    'zh': u'嵌合體X',           # Chinese
}


if sys.platform.startswith('win'):
    # Python on Windows is missing the <sysexits.h> exit codes
    os.EX_OK = 0                # successful termination
    os.EX_USAGE = 64            # command line usage error
    os.EX_DATAERR = 65          # data format error
    os.EX_NOINPUT = 66          # cannot open input
    os.EX_NOUSER = 67           # addressee unknown
    os.EX_NOHOST = 68           # host name unknown
    os.EX_UNAVAILABLE = 69      # service unavailable
    os.EX_SOFTWARE = 70         # internal software error
    os.EX_OSERR = 71            # system error (e.g., can't fork)
    os.EX_OSFILE = 72           # critical OS file missing
    os.EX_CANTCREAT = 73        # can't create (user) output file
    os.EX_IOERR = 74            # input/output error
    os.EX_TEMPFAIL = 75         # temp failure; user is invited to retry
    os.EX_PROTOCOL = 76         # remote error in protocol
    os.EX_NOPERM = 77           # permission denied
    os.EX_CONFIG = 78           # configuration error

    if 'LANG' in os.environ:
        # Double check that stdout matches what LANG asks for.
        # This is a problem when running in nogui mode from inside a cygwin
        # shell -- the console is supposed to use UTF-8 encoding in Python
        # 3.6 but sys.stdout.encoding is cpXXX (the default for text file
        # I/O) since cygwin shells are not true terminals.
        import io
        encoding = os.environ['LANG'].split('.')[-1].casefold()
        if encoding != sys.stdout.encoding.casefold():
            try:
                sys.__stdout__ = sys.stdout = io.TextIOWrapper(
                    sys.stdout.detach(), encoding, 'backslashreplace',
                    line_buffering=sys.stdout.line_buffering)
                sys.__stderr__ = sys.stderr = io.TextIOWrapper(
                    sys.stderr.detach(), encoding, 'backslashreplace',
                    line_buffering=sys.stderr.line_buffering)
            except LookupError:
                # If encoding is unknown, just leave things as is
                pass


def parse_arguments(argv):
    """Initialize ChimeraX application."""
    import getopt

    if sys.platform.startswith('darwin'):
        # skip extra -psn_ argument on Mac OS X 10.8 and earlier and Mac OS X 10.12 on first launch
        for i, arg in enumerate(argv):
            if i > 0 and arg.startswith('-psn_'):
                del argv[i]
                break

    class Opts:
        pass
    opts = Opts()
    opts.commands = []
    opts.cmd = None   # Python's -c option
    opts.debug = False
    opts.gui = True
    opts.module = None  # Python's -m option
    opts.line_profile = False
    opts.list_io_formats = False
    opts.load_tools = True
    opts.offscreen = False
    opts.scripts = []
    opts.silent = False
    opts.start_tools = []
    opts.status = True
    opts.stereo = False
    opts.uninstall = False
    opts.use_defaults = False
    opts.version = -1
    opts.get_available_bundles = True

    # Will build usage string from list of arguments
    arguments = [
        "--debug",
        "--nogui",
        "--help",
        "--lineprofile",
        "--listioformats",
        "--offscreen",
        "--silent",
        "--nostatus",
        "--start <tool name>",
        "--cmd <command>",
        "--script <python script and arguments>",
        "--notools",
        "--stereo",
        "--uninstall",
        "--usedefaults",
        "--version",
        "--qtscalefactor <factor>",
    ]
    if sys.platform.startswith("win"):
        arguments += ["--console", "--noconsole"]
    usage = '[' + "] [".join(arguments) + ']'
    usage += " or -m module_name [args]"
    usage += " or -c command [args]"
    usage += " or -u -c command [args]"
    # add in default argument values
    arguments += [
        "--nodebug",
        "--gui",
        "--nolineprofile",
        "--nosilent",
        "--nousedefaults",
        "--nooffscreen",
        "--status",
        "--tools",
        "--nousedefaults",
    ]
    if len(sys.argv) > 2 and sys.argv[1] == '-m':
        # treat like Python's -m argument
        opts.gui = False
        opts.silent = True
        opts.module = sys.argv[2]
        return opts, sys.argv[2:]
    if len(sys.argv) > 2 and sys.argv[1] == '-c':
        # treat like Python's -c argument
        opts.gui = False
        opts.silent = True
        opts.cmd = sys.argv[2]
        return opts, sys.argv[2:]
    if len(sys.argv) > 2 and sys.argv[1:3] == ['-u', '-c']:
        # treat like Python's -c argument
        import io
        sys.stdout = io.TextIOWrapper(os.fdopen(sys.stdout.fileno(), 'wb'),
                                      write_through=True)
        sys.stderr = io.TextIOWrapper(os.fdopen(sys.stderr.fileno(), 'wb'),
                                      write_through=True)
        opts.gui = False
        opts.silent = True
        opts.cmd = sys.argv[3]
        return opts, sys.argv[3:]
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
        elif opt == "--listioformats":
            opts.list_io_formats = True
        elif opt in ("--offscreen", "--nooffscreen"):
            opts.offscreen = opt[2] == 'o'
        elif opt in ("--silent", "--nosilent"):
            opts.silent = opt[2] == 's'
        elif opt in ("--status", "--nostatus"):
            opts.status = opt[2] == 's'
        elif opt in "--stereo":
            opts.stereo = True
        elif opt == "--start":
            opts.start_tools.append(optarg)
        elif opt == "--cmd":
            opts.commands.append(optarg)
        elif opt == "--script":
            opts.scripts.append(optarg)
        elif opt in ("--tools", "--notools"):
            opts.load_tools = opt[2] == 't'
        elif opt == "--uninstall":
            opts.uninstall = True
        elif opt in ("--usedefaults", "--nousedefaults"):
            opts.load_tools = opt[2] == 'u'
        elif opt == "--version":
            opts.version += 1
        elif opt == "--qtscalefactor":
            os.environ["QT_SCALE_FACTOR"] = optarg
        else:
            print("Unknown option: ", opt)
            help = True
            break
    if help:
        print("usage: %s %s\n" % (argv[0], usage), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)
    if opts.version >= 0 or opts.list_io_formats:
        opts.gui = False
        opts.silent = True
        opts.get_available_bundles = False
    return opts, args


def init(argv, event_loop=True):
    import site, sys
    if sys.platform.startswith('darwin'):
        paths = os.environ['PATH'].split(':')
        if '/usr/sbin' not in paths:
            # numpy, numexpr, and pytables need sysctl in path
            paths.append('/usr/sbin')
            os.environ['PATH'] = ':'.join(paths)
        del paths

    from chimerax.core.utils import initialize_ssl_cert_dir
    initialize_ssl_cert_dir()

    # find chimerax.core's version
    # we cannot use pip for this because we want to update
    # site.USER_SITE before importing pip, and site.USER_SITE
    # depends on the version
    try:
        from chimerax.core import version
    except ImportError:
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
        from chimerax.core import configinfo
        configinfo.only_use_defaults = True

    from chimerax import core
    if not opts.gui and opts.offscreen:
        # Flag to configure off-screen rendering before PyOpenGL imported
        core.offscreen_rendering = True

    if not opts.gui and opts.load_tools:
        # only load tools if we have a GUI
        opts.load_tools = False

    # figure out the user/system directories for application
    # invoked with -m ChimeraX_main, so argv[0] is full path to ChimeraX_main
    # Windows:
    # 'C:\\...\\ChimeraX.app\\bin\\lib\\site-packages\\ChimeraX_main.py'
    # Linux:
    # '/.../ChimeraX.app/lib/python3.5/site-packages/ChimeraX_main.py'
    # Mac OS X:
    # '/.../ChimeraX.app/Contents/lib/python3.5/site-packages/ChimeraX_main.py'
    # '/.../ChimeraX.app/Contents/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/ChimeraX_main.py'
    # TODO: more robust way
    dn = os.path.dirname
    rootdir = dn(dn(dn(dn(sys.argv[0]))))
    if sys.platform.startswith('darwin'):
        rootdir = dn(dn(dn(dn(dn(rootdir)))))
    if sys.platform.startswith('linux'):
        os.environ['XDG_CONFIG_DIRS'] = rootdir

    if sys.platform.startswith('win'):
        if 'HOME' in os.environ:
            # Windows uses HOMEPATH and HOMEDRIVE, so HOME's presence indicates
            # a non-standard startup environment.  So remove HOME to make
            # sure the the correct application paths are figured out.
            del os.environ['HOME']
        import ctypes
        # getpn = ctypes.pythonapi.Py_GetProgramName
        # getpn.restype = ctypes.c_wchar_p
        # pn = getpn()
        # assert(os.path.dirname(pn) == rootdir)
        # Python uses LoadLibraryEx with LOAD_WITH_ALTERED_SEARCH_PATH to
        # search in directory of library first instead of the directory of
        # application binary.  So add back the "bin" directory, which is
        # the Windows equivalent of the Linux/Mac OS X rpath directory.
        setdlldir = ctypes.windll.kernel32.SetDllDirectoryW
        setdlldir.argtypes = [ctypes.c_wchar_p]
        setdlldir.restype = ctypes.c_bool
        setdlldir(os.path.join(rootdir, 'bin'))

    from distlib.version import NormalizedVersion as Version
    epoch, ver, *_ = Version(version).parse(version)
    if len(ver) == 1:
        ver += (0,)
    partial_version = '%s.%s' % (ver[0], ver[1])

    import chimerax
    import appdirs
    chimerax.app_dirs = ad = appdirs.AppDirs(app_name, appauthor=app_author,
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
    chimerax.app_dirs_unversioned = adu = appdirs.AppDirs(app_name, appauthor=app_author)

    # update "site" user variables to use ChimeraX instead of Python paths
    # NB: USER_SITE is used by both pip and the toolshed, so
    # this must happen before pip is imported so that "--user" installs
    # will go in the right place.
    import site
    site.USER_BASE = adu.user_data_dir
    site.USER_SITE = os.path.join(ad.user_data_dir, "site-packages")

    # Find the location of "share" directory so that we can inform
    # the C++ layer.  Assume it's a sibling of the directory that
    # the executable is in.
    chimerax.app_bin_dir = os.path.join(rootdir, "bin")
    if sys.platform.startswith('win'):
        chimerax.app_data_dir = os.path.join(chimerax.app_bin_dir, "share")
    else:
        chimerax.app_data_dir = os.path.join(rootdir, "share")
    chimerax.app_lib_dir = os.path.join(rootdir, "lib")

    # inform the C++ layer of the appdirs paths
    from chimerax.core import _appdirs
    _appdirs.init_paths(os.sep, ad.user_data_dir, ad.user_config_dir,
                        ad.user_cache_dir, ad.site_data_dir,
                        ad.site_config_dir, ad.user_log_dir,
                        chimerax.app_data_dir, adu.user_cache_dir)

    from chimerax.core import session
    sess = session.Session(app_name, debug=opts.debug, silent=opts.silent)

    from chimerax.core import core_settings
    core_settings.init(sess)

    session.common_startup(sess)

    if opts.uninstall:
        return uninstall(sess)

    # initialize the user interface
    if opts.gui:
        from chimerax.core.ui import gui
        ui_class = gui.UI
    else:
        from chimerax.core.ui import nogui
        ui_class = nogui.UI
    # sets up logging, splash screen if gui
    # calls "sess.save_in_session(self)"
    sess.ui = ui_class(sess)
    sess.ui.stereo = opts.stereo
    sess.ui.autostart_tools = opts.load_tools

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

    from chimerax.core import toolshed
    # toolshed.init returns a singleton so it's safe to call multiple times
    sess.toolshed = toolshed.init(sess.logger, debug=sess.debug,
                                  check_available=opts.get_available_bundles)
    if opts.module != 'pip':
        # keep bugs in ChimeraX from preventing pip from working
        if not opts.silent:
            sess.ui.splash_info("Initializing bundles",
                                next(splash_step), num_splash_steps)
            if sess.ui.is_gui and opts.debug:
                print("Initializing bundles", flush=True)
        sess.toolshed.bootstrap_bundles(sess)
        from chimerax.core import tools
        sess.tools = tools.Tools(sess, first=True)
        from chimerax.core import tasks
        sess.tasks = tasks.Tasks(sess, first=True)

    if opts.version >= 0:
        sess.silent = False
        format = [None, 'verbose', 'bundles', 'packages'][opts.version]
        from chimerax.core.commands import command_function
        version_cmd = command_function("version")
        version_cmd(sess, format)
        return os.EX_OK

    if opts.list_io_formats:
        sess.silent = False
        from chimerax.core import io
        io.print_file_suffixes()
        # TODO: show database formats
        # TODO: show mime types?
        # TODO: show compression suffixes?
        raise SystemExit(0)

    if opts.gui and sys.platform.startswith('linux'):
        from chimerax.core import _xdg
        _xdg.install_if_needed(sess, localized_app_name)

    if opts.gui:
        # build out the UI, populate menus, create graphics, etc.
        if not opts.silent:
            sess.ui.splash_info("Starting main interface",
                                next(splash_step), num_splash_steps)
            if sess.ui.is_gui and opts.debug:
                print("Starting main interface", flush=True)
        sess.ui.build()

    if opts.start_tools:
        if not opts.silent:
            msg = 'Starting tools %s' % ', '.join(opts.start_tools)
            sess.ui.splash_info(msg, next(splash_step), num_splash_steps)
            if sess.ui.is_gui and opts.debug:
                print(msg, flush=True)
        # canonicalize tool names
        start_tools = [sess.toolshed.find_bundle_for_tool(t)[1] for t in opts.start_tools]
        sess.tools.start_tools(start_tools)

    if opts.commands:
        if not opts.silent:
            msg = 'Running startup commands'
            # sess.ui.splash_info(msg, next(splash_step), num_splash_steps)
            if sess.ui.is_gui and opts.debug:
                print(msg, flush=True)
        from chimerax.core.commands import run
        for cmd in opts.commands:
            run(sess, cmd)

    if opts.scripts:
        if not opts.silent:
            msg = 'Running startup scripts'
            # sess.ui.splash_info(msg, next(splash_step), num_splash_steps)
            if sess.ui.is_gui and opts.debug:
                print(msg, flush=True)
        from chimerax.core.commands import runscript
        for script in opts.scripts:
            runscript(sess, script)

    if not opts.silent:
        sess.ui.splash_info("Finished initialization",
                            next(splash_step), num_splash_steps)
        if sess.ui.is_gui and opts.debug:
            print("Finished initialization", flush=True)

    if opts.gui:
        sess.ui.close_splash()

    if not opts.silent:
        import chimerax.core.commands.version as vercmd
        vercmd.version(sess)  # report version in log

    if opts.gui or hasattr(core, 'offscreen_rendering'):
        r = sess.main_view.render
        log = sess.logger
        from chimerax.core.graphics import OpenGLVersionError
        try:
            mc = r.make_current()
        except OpenGLVersionError as e:
            log.error(str(e))
            mc = False
        if mc:
            info = log.info
            e = r.check_for_opengl_errors()
            if e:
                msg = 'There was an OpenGL graphics error while starting up.  This is usually a problem with the system graphics driver, and the only way to remedy it is to update the graphics driver. ChimeraX will probably not function correctly with the current graphics driver.'
                msg += '\n\n\t"%s"' % e
                log.error(msg)
            info('OpenGL version: ' + r.opengl_version())
            info('OpenGL renderer: ' + r.opengl_renderer())
            info('OpenGL vendor: ' + r.opengl_vendor())
            sess.ui.main_window.graphics_window.start_redraw_timer()

    if opts.module:
        import runpy
        import warnings
        sys.argv[:] = args  # runpy will insert appropriate argv[0]
        has_install = 'install' in sys.argv
        has_uninstall = 'uninstall' in sys.argv
        per_user = '-user' in sys.argv
        exit = SystemExit(os.EX_OK)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=BytesWarning)
            global_dict = {
                'session': sess
            }
            try:
                runpy.run_module(opts.module, init_globals=global_dict,
                                 run_name='__main__', alter_sys=True)
            except SystemExit as e:
                exit = e
        if opts.module == 'pip' and exit.code == os.EX_OK:
            if has_install or has_uninstall:
                sess.toolshed.reload(sess.logger, rebuild_cache=True)
                sess.toolshed.set_install_timestamp(per_user)
            if has_install:
                remove_python_scripts(chimerax.app_bin_dir)
        return exit.code

    from chimerax.core import startup
    startup.run_user_startup_scripts(sess)

    if opts.cmd:
        # Emulated Python's -c option.
        # This is needed for -m pip to work in some cases.
        sys.argv = args
        sys.argv[0] = '-c'
        global_dict = {
            'session': sess
        }
        exec(opts.cmd, global_dict)
        return os.EX_OK

    # the rest of the arguments are data files
    from chimerax.core import errors, commands
    for arg in args:
        try:
            commands.run(sess, 'open %s' % arg)
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
    # change directory so we're guaranteed not to be in the ChimeraX app
    os.chdir(tempfile.gettempdir())

    # find location of ChimeraX
    if sys.executable is None:
        sess.logger.error('unable to locate ChimeraX executable')
        return os.EX_SOFTWARE
    exe = os.path.realpath(sys.executable)
    exe_dir = os.path.dirname(exe)

    if sys.platform.startswith('linux'):
        if os.path.basename(exe_dir) != 'bin':
            sys.logger.error('non-standard ChimeraX installation')
            return os.EX_SOFTWARE
        from chimerax.core import _xdg
        _xdg.uninstall(sess)
        # parent = os.path.dirname(exe_dir)
        # rm_rf_path(parent, sess)
        return os.EX_OK

    if sys.platform.startswith('darwin'):
        if os.path.basename(exe_dir) != 'MacOS':
            sess.logger.error('non-standard ChimeraX installation')
            return os.EX_SOFTWARE
        parent = os.path.dirname(exe_dir)
        if os.path.basename(parent) != 'Contents':
            sess.logger.error('non-standard ChimeraX installation')
            return os.EX_SOFTWARE
        parent = os.path.dirname(parent)
        if not os.path.basename(parent).endswith('.app'):
            sess.logger.error('non-standard ChimeraX installation')
            return os.EX_SOFTWARE
        rm_rf_path(parent, sess)
        return os.EX_OK

    sess.logger.error('can not yet uninstall on %s' % sys.platform)
    return os.EX_UNAVAILABLE


def remove_python_scripts(bin_dir):
    # remove pip installed scripts since they have hardcoded paths to
    # python and thus don't work when ChimeraX is installed elsewhere
    import os
    if sys.platform.startswith('win'):
        # Windows
        script_dir = os.path.join(bin_dir, 'Scripts')
        for dirpath, dirnames, filenames in os.walk(script_dir, topdown=False):
            for f in filenames:
                path = os.path.join(dirpath, f)
                os.remove(path)
            os.rmdir(dirpath)
    else:
        # Linux, Mac OS X
        for filename in os.listdir(bin_dir):
            path = os.path.join(bin_dir, filename)
            if not os.path.isfile(path):
                continue
            with open(path, 'br') as f:
                line = f.readline()
                if line[0:2] != b'#!' or b'/bin/python' not in line:
                    continue
            print('removing (pip installed)', path)
            os.remove(path)

if __name__ == '__main__':
    exit_code = init(sys.argv)
    raise SystemExit(exit_code)
