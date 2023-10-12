# vim: set expandtab shiftwidth=4 softtabstop=4:
# Copyright © 2022 Regents of the University of California.
# All Rights Reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
import logging
import os
import sys

__version__ = "0.2.0a0"     # version of this file -- PEP 440 compatible

app_name = "ChimeraX"
app_author = "UCSF"
# remember locale codes are frequently different than country codes
localized_app_name = {
    'af': u'ChimeraX',          # Afrikaans
    'cs': u'PřízrakX',          # Czech
    'da': u'ChimeraX',          # Danish
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

    if 'LANG' in os.environ and sys.stdout is not None:
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


class Opts:

    def __init__(self):
        self.help = False
        self.commands = []
        self.cmd = None   # Python's -c option
        self.debug = False
        self.devel = False
        self.event_loop = True
        self.gui = True
        self.color = None
        self.module = None  # Python's -m option
        self.run_path = None  # Need to act like "python path args"
        self.line_profile = False
        self.list_io_formats = False
        self.load_tools = True
        self.offscreen = False
        self.scripts = []
        self.silent = False
        self.start_tools = []
        self.status = True
        self.stereo = False
        self.uninstall = False
        self.use_defaults = False
        self.version = -1
        self.get_available_bundles = True
        self.safe_mode = False
        self.toolshed = None
        self.disable_qt = False


def _parse_python_args(argv, usage):
    # ChimeraX can be invoked by pip thinking that it is Python.
    # Treat all single letter arguments as Python arguments
    # and make sure to cover all of the arguments that the
    # subprocess package generates as well as -c, -m, and -u.
    # Can't use getopt because -m short circuits argument parsing.

    opts = Opts()
    opts.gui = False
    opts.event_loop = False
    opts.get_available_bundles = False
    opts.load_tools = False
    opts.silent = True
    opts.safe_mode = True

    def next_arg(argv):
        no_arg = "bBdEhiIOqsSuvVx"  # python option w/o argument
        has_arg = "cmWX"            # python option w/argument
        cur_index = 1
        while len(argv) > cur_index and argv[cur_index][0] == '-':
            cur_opts = argv[cur_index]
            cur_index += 1
            if cur_opts == '--':
                yield None, argv[cur_index:]
                return
            for opt in cur_opts[1:]:
                if opt in no_arg:
                    yield f"-{opt}", None
                elif opt in has_arg:
                    if len(argv) <= cur_index:
                        raise RuntimeError(f"Missing argument for '-{opt}'")
                    if opt == 'm' or opt == 'c':
                        # special case, eats rest of arguments
                        yield f'-{opt}', argv[cur_index]
                        yield None, argv[cur_index + 1:]
                        return
                    arg = argv[cur_index]
                    cur_index += 1
                    yield f"-{opt}", arg
                else:
                    raise RuntimeError(f"Unknown argument '-{opt}'")
        yield None, argv[cur_index:]

    args = []
    version = 0
    try:
        for opt, optarg in next_arg(argv):
            if opt is None:
                args = optarg
                break  # last one anyway
            # silently ignore options we don't use
            if opt == "-c":
                opts.cmd = optarg
                opts.safe_mode = True
            elif opt == "-m":
                opts.module = optarg
                opts.safe_mode = True
            elif opt == "-u":
                import io
                sys.stdout = io.TextIOWrapper(os.fdopen(sys.stdout.fileno(), 'wb'),
                                              write_through=True)
                sys.stderr = io.TextIOWrapper(os.fdopen(sys.stderr.fileno(), 'wb'),
                                              write_through=True)
            elif opt == "-h":
                opts.help = True
            elif opt == "-d":
                opts.debug = True
                opts.devel = True
                opts.silent = False
            elif opt == "-V":
                version += 1
    except RuntimeError as message:
        print("%s: %s" % (argv[0], message), file=sys.stderr)
        print("usage: %s %s\n" % (argv[0], usage), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)
    if version:
        if version > 1:
            print('Python', sys.version)
        else:
            print('Python', sys.version.split(' ', 1)[0])
        raise SystemExit(0)
    return opts, args


def _parse_chimerax_args(argv, arguments, usage):
    import getopt

    try:
        longopts = []
        for a in arguments:
            i = a.find(' ')
            if i == -1:
                longopts.append(a[2:])
            else:
                longopts.append(a[2:i] + '=')
        options, args = getopt.getopt(argv[1:], "", longopts)
    except getopt.error as message:
        print("%s: %s" % (argv[0], message), file=sys.stderr)
        print("usage: %s %s\n" % (argv[0], usage), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)

    opts = Opts()
    for opt, optarg in options:
        if opt in ("--debug", "--nodebug"):
            opts.debug = opt[2] == 'd'
            if opts.debug:
                opts.devel = True
        elif opt in ("--devel", "--nodevel"):
            opts.devel = opt[2] == 'd'
        elif opt in ("--exit", "--noexit"):
            opts.event_loop = opt[2] != 'e'
            opts.get_available_bundles = False
        elif opt == "--help":
            opts.help = True
        elif opt in ("--gui", "--nogui"):
            opts.gui = opt[2] == 'g'
        elif opt in ("--color", "--nocolor"):
            opts.color = opt[2] == 'c'
        elif opt in ("--lineprofile", "--nolineprofile"):
            opts.line_profile = opt[2] == 'l'
            if opts.line_profile:
                opts.get_available_bundles = False
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
            opts.get_available_bundles = False
        elif opt == "--script":
            opts.scripts.append(optarg)
            opts.get_available_bundles = False
        elif opt in ("--tools", "--notools"):
            opts.load_tools = opt[2] == 't'
            if not opts.load_tools:
                opts.get_available_bundles = False
        elif opt == "--uninstall":
            opts.uninstall = True
        elif opt == "--safemode":
            opts.safe_mode = True
            opts.load_tools = False
        elif opt in ("--usedefaults", "--nousedefaults"):
            print(f"warning: {opt} is not supported yet", file=sys.stderr)
        elif opt == "--version":
            opts.version += 1
        elif opt == "--toolshed":
            opts.toolshed = optarg
        elif opt == "--disable-qt":
            opts.disable_qt = True
        else:
            print("Unknown option: ", opt)
            opts.help = True
            break
    if opts.version >= 0 or opts.list_io_formats:
        opts.gui = False
        opts.silent = True
        opts.get_available_bundles = False
    return opts, args


def parse_arguments(argv):
    """Initialize ChimeraX application."""
    if sys.platform.startswith('darwin'):
        # skip extra -psn_ argument on Mac OS X 10.8 and earlier and Mac OS X 10.12 on first launch
        for i, arg in enumerate(argv):
            if i > 0 and arg.startswith('-psn_'):
                del argv[i]
                break

    if len(argv) <= 1:
        return Opts(), []

    # Will build usage string from list of arguments
    arguments = [
        "--debug",
        "--devel",
        "--exit",   # No event loop
        "--nogui",
        "--nocolor",
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
        "--safemode",
        "--stereo",
        "--uninstall",
        "--usedefaults",
        "--version",
        "--qtscalefactor <factor>",
        "--toolshed preview|<url>",
        "--disable-qt",
    ]
    if sys.platform.startswith("win"):
        arguments += ["--console", "--noconsole"]
    usage = '[' + "] [".join(arguments) + ']'
    usage += " or Python command line arguments"
    usage += " (e.g., -m module_name [args]"
    usage += " or -c command [args])"
    # add in default argument values
    arguments += [
        "--nodebug",
        "--nodevel",
        "--noexit",
        "--gui",
        "--color",
        "--nolineprofile",
        "--nosilent",
        "--nousedefaults",
        "--nooffscreen",
        "--status",
        "--tools",
        "--nousedefaults",
    ]

    # This used to simply import pip, but this breaks new versions of setuptools.
    # See Trac#7159
    import site
    import os
    recursive_pip = any(
        [argv[1].startswith(os.path.join(path, "pip")) for path in site.getsitepackages()]
    )
    if recursive_pip:
        # treat like recursive invokation of pip
        opts = Opts()
        opts.gui = False
        opts.silent = True
        opts.event_loop = False
        opts.get_available_bundles = False
        opts.run_path = argv[1]
        opts.load_tools = False
        opts.safe_mode = True
        args = argv[1:]
    elif argv[1][0:2] == '--':
        # ChimeraX style options
        opts, args = _parse_chimerax_args(argv, arguments, usage)
    elif argv[1][0] == '-':
        # Python style options
        opts, args = _parse_python_args(argv, usage)
    else:
        # no options
        opts = Opts()
        args = argv[1:]

    if opts.help:
        print("usage: %s %s\n" % (argv[0], usage))
        raise SystemExit(os.EX_USAGE)
    return opts, args


def init(argv, event_loop=True):
    import sys
    # MacOS 10.12+ generates drop event for command-line argument before main()
    # is even called; compensate
    bad_drop_events = False
    if sys.platform.startswith('darwin'):
        paths = os.environ['PATH'].split(':')
        if '/usr/sbin' not in paths:
            # numpy, numexpr, and pytables need sysctl in path
            paths.append('/usr/sbin')
            os.environ['PATH'] = ':'.join(paths)
        del paths
        # ChimeraX is only distributed for 10.13+, so don't need to check version
        bad_drop_events = True

    if sys.platform.startswith('linux'):
        # Workaround for #638:
        # "any number of threads more than one leads to 200% CPU usage"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Setup SSL CA certificates file
    # This used to be only necessary for darwin, but Windows
    # appears to need it as well.  So use it for all platforms.
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()

    if len(argv) > 1 and argv[1].startswith('--'):
        # MacOS doesn't generate these drop events for args after '--' flags
        bad_drop_events = False
    opts, args = parse_arguments(argv)
    if not opts.devel:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    if not opts.gui and opts.disable_qt:
        # Disable importing Qt in nogui mode to catch imports that
        # would affect that would break ChimeraX pypi library
        sys.modules['qtpy'] = None
        sys.modules['Qt'] = None
        sys.modules['PyQt5'] = None
        sys.modules['PyQt6'] = None
        sys.modules['PySide6'] = None
        sys.modules['PySide2'] = None

    # install line_profile decorator, and install it before
    # initialize_ssl_cert_dir() in case the line profiling is in the
    # core (which would cause initialize_ssl_cert_dir() to fail)
    import builtins
    if not opts.line_profile:
        builtins.__dict__['line_profile'] = lambda x: x
    else:
        try:
            import line_profiler
            prof = line_profiler.LineProfiler()
            builtins.__dict__['line_profile'] = prof
        except ImportError:
            print("warning: line_profiler is not available", file=sys.stderr)
            builtins.__dict__['line_profile'] = lambda x: x
        else:
            # write profile results on exit
            import atexit
            atexit.register(prof.dump_stats, "%s.lprof" % app_name)

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

    if opts.use_defaults:
        from chimerax.core import configinfo
        configinfo.only_use_defaults = True

    if opts.offscreen:
        opts.gui = False

    if not opts.gui and opts.load_tools:
        # only load tools if we have a GUI
        opts.load_tools = False

    # Windows:
    #     python: C:\\...\\ChimeraX.app\\bin\\python.exe
    #     ChimeraX: C:\\...\\ChimeraX.app\\bin\\ChimeraX
    # Linux:
    #     python: /../ChimeraX.app/bin/python3.x
    #     ChimeraX: /../ChimeraX.app/bin/ChimeraX
    # macOS:
    #     python: /../ChimeraX.app/Contents/bin/python3.x
    #     ChimeraX: /../ChimeraX.app/Contents/MacOS/ChimeraX
    dn = os.path.dirname
    rootdir = dn(dn(os.path.realpath(sys.executable)))
    # On Linux, don't create user directories if root (the installer uid)
    is_root = False
    if sys.platform.startswith('linux'):
        os.environ['XDG_CONFIG_DIRS'] = rootdir
        is_root = os.getuid() == 0
        if is_root:
            # ensure toolshed cache is not written
            os.environ['HOME'] = "/do/not/run/as/root"

    if sys.platform.startswith('win'):
        rootdir = os.path.join(rootdir, "bin")
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
        setdlldir(rootdir)

    from packaging.version import Version
    ver = Version(version)
    partial_version = f"{ver.major}.{ver.minor}"

    import chimerax
    import appdirs
    chimerax.app_dirs = ad = appdirs.AppDirs(app_name, appauthor=app_author,
                                             version=partial_version)
    if not is_root:
        # make sure app_dirs.user_* directories exist
        for var, name in (
                ('user_data_dir', "user's data"),
                ('user_config_dir', "user's configuration"),
                ('user_cache_dir', "user's cache")):
            directory = getattr(ad, var)
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                print("Unable to make %s directory: %s: %s" % (
                    name, e.strerror, e.filename), file=sys.stderr)
                return os.EX_CANTCREAT

    # app_dirs_unversioned is primarily for caching data files that will
    # open in any version
    # app_dirs_unversioned.user_* directories are parents of those in app_dirs
    chimerax.app_dirs_unversioned = appdirs.AppDirs(app_name, appauthor=app_author)

    # update "site" user variables to use ChimeraX instead of Python paths
    # NB: USER_SITE is used by both pip and the toolshed, so
    # this must happen before pip is imported so that "--user" installs
    # will go in the right place.
    import site
    if not is_root:
        site.USER_BASE = ad.user_data_dir
        lib = "lib"
        python = "python"
        if sys.platform == "win32":
            lib = ""
            python = f"Python{sys.version_info.major}{sys.version_info.minor}"
        if sys.platform == "linux":
            python = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site.USER_SITE = os.path.join(site.USER_BASE, lib, python, "site-packages")
    else:
        import sysconfig
        site.USER_SITE = sysconfig.get_paths()['platlib']
    site.ENABLE_USER_SITE = True

    # Find the location of "share" directory so that we can inform
    # the C++ layer.  Assume it's a sibling of the directory that
    # the executable is in.
    if sys.platform == "win32":
        chimerax.app_bin_dir = os.path.join(rootdir)
    else:
        chimerax.app_bin_dir = os.path.join(rootdir, "bin")
    chimerax.app_data_dir = os.path.join(rootdir, "share")
    chimerax.app_lib_dir = os.path.join(rootdir, "lib")

    from chimerax.core import session
    try:
        sess = session.Session(app_name,
                               debug=opts.debug,
                               silent=opts.silent,
                               minimal=opts.safe_mode,
                               offscreen_rendering=opts.offscreen)
    except ImportError as err:
        if opts.offscreen and 'OpenGL' in err.args[0]:
            if sys.platform.startswith("linux"):
                why = "failed"
            else:
                why = "not supported on this platform"
            print("Offscreen rendering is", why, file=sys.stderr)
            return os.EX_UNAVAILABLE
        raise

    from chimerax.core import core_settings
    core_settings.init(sess)

    from chimerax.core.session import register_misc_commands
    register_misc_commands(sess)

    from chimerax.core import attributes
    attributes.RegAttrManager(sess)

    if opts.uninstall:
        return uninstall(sess)

    # initialize qt
    if opts.gui:
        if os.environ.get("WAYLAND_DISPLAY", ""):
            # ChimeraX needs to use XWayland for now
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            os.environ["PYOPENGL_PLATFORM"] = "x11"
        from chimerax.ui import initialize_qt
        initialize_qt()

    # initialize the user interface
    # sets up logging
    if opts.gui:
        from chimerax.ui import gui
        sess.ui = gui.UI(sess)
    else:
        from chimerax.core.nogui import NoGuiLog
        sess.logger.add_log(NoGuiLog())

    # Set ui options
    sess.ui.stereo = opts.stereo
    sess.ui.autostart_tools = opts.load_tools
    if not opts.gui:
        sess.ui.initialize_color_output(opts.color)    # Colored text

    # Set current working directory to Desktop when launched from icon.
    if ((sys.platform.startswith('darwin') and
         os.getcwd() == '/') or
        (sys.platform.startswith('win') and
         (os.getcwd().endswith('\\Users\\Public\\Desktop') or
          os.getcwd().endswith('\\ProgramData\\ChimeraX')))):
        try:
            os.chdir(os.path.expanduser('~/Desktop'))
        except Exception:
            pass

    # common core initialization
    if not opts.silent:
        if sess.ui.is_gui and opts.debug:
            print("Initializing core", flush=True)

    # Install any bundles before toolshed is initialized so
    # the new ones get picked up in this session
    from chimerax.core import toolshed
    inst_dir, restart_file = toolshed.restart_action_info()
    restart_action_msgs = []
    if os.path.exists(restart_file):
        # Move file out of the way so next restart of ChimeraX
        # (when we try to install the bundle) will not go into
        # an infinite loop reopening the restart file
        tmp_file = restart_file + ".tmp"
        try:
            # Remove in case old file lying around.
            # Windows does not allow renaming to an existing file.
            os.remove(tmp_file)
        except Exception:
            pass
        os.rename(restart_file, tmp_file)
        with open(tmp_file) as f:
            for line in f:
                restart_action(line, inst_dir, restart_action_msgs)
        os.remove(tmp_file)

    if opts.toolshed is None:
        # Default to whatever the restart actions needed
        toolshed_url = _restart_toolshed_url
    elif opts.toolshed == "preview":
        toolshed_url = toolshed.preview_toolshed_url()
    else:
        toolshed_url = opts.toolshed
    toolshed.init(sess.logger, debug=sess.debug,
                  check_available=opts.get_available_bundles,
                  remote_url=toolshed_url, session=sess)
    sess.toolshed = toolshed.get_toolshed()
    if opts.module != 'pip' and opts.run_path is None:
        # keep bugs in ChimeraX from preventing pip from working
        if not opts.silent:
            if sess.ui.is_gui and opts.debug:
                print("Initializing bundles", flush=True)
        sess.toolshed.bootstrap_bundles(sess, opts.safe_mode)
        from chimerax.core import tools
        sess.tools = tools.Tools(sess, first=True)
        from chimerax.core import undo
        sess.undo = undo.Undo(sess, first=True)

    if opts.version >= 0:
        sess.silent = False
        if opts.version > 3:
            opts.version = 3
        format = [None, 'verbose', 'bundles', 'packages'][opts.version]
        from chimerax.core.commands import command_function
        version_cmd = command_function("version")
        version_cmd(sess, format)
        return os.EX_OK

    if opts.list_io_formats:
        sess.silent = False
        collate = {}
        for fmt in sess.data_formats.formats:
            collate.setdefault(fmt.category, []).append(fmt)
        categories = list(collate.keys())
        categories.sort(key=str.casefold)
        print("Supported file suffixes:")
        print("  o = open, s = save")
        openers = set(sess.open_command.open_data_formats)
        savers = set(sess.save_command.save_data_formats)
        for cat in categories:
            print("\n%s:" % cat)
            fmts = collate[cat]
            fmts.sort(key=lambda fmt: fmt.name.casefold())
            for fmt in fmts:
                o = 'o' if fmt in openers else ' '
                s = 's' if fmt in savers else ' '
                if fmt.suffixes:
                    exts = ': ' + ', '.join(fmt.suffixes)
                else:
                    exts = ''
                print("%c%c  %s%s" % (o, s, fmt.name, exts))
        # TODO: show database formats
        # TODO: show mime types?
        # TODO: show compression suffixes?
        return os.EX_OK

    if opts.gui:
        # build out the UI, populate menus, create graphics, etc.
        if not opts.silent:
            if sess.ui.is_gui and opts.debug:
                print("Starting main interface", flush=True)
        sess.ui.build()

    if opts.start_tools:
        if not opts.silent:
            msg = 'Starting tools %s' % ', '.join(opts.start_tools)
            if sess.ui.is_gui and opts.debug:
                print(msg, flush=True)
        # canonicalize tool names
        start_tools = []
        for t in opts.start_tools:
            tools = sess.toolshed.find_bundle_for_tool(t)
            if not tools:
                sess.logger.warning("Unable to find tool %s" % repr(t))
                continue
            start_tools.append(tools[0][1])
        sess.tools.start_tools(start_tools)

    if opts.commands:
        if not opts.silent:
            msg = 'Running startup commands'
            if sess.ui.is_gui and opts.debug:
                print(msg, flush=True)
        from chimerax.core.commands import run
        with sess.in_script:
            for cmd in opts.commands:
                try:
                    run(sess, cmd)
                except Exception:
                    if not sess.ui.is_gui:
                        import traceback
                        traceback.print_exc()
                        return os.EX_SOFTWARE
                    # Allow GUI to start up despite errors;
                    if sess.debug:
                        import traceback
                        traceback.print_exc(file=sys.__stderr__)
                    else:
                        sess.ui.thread_safe(sess.logger.report_exception, exc_info=sys.exc_info())

    if opts.scripts:
        if not opts.silent:
            msg = 'Running startup scripts'
            if sess.ui.is_gui and opts.debug:
                print(msg, flush=True)
        from chimerax.core.commands import run
        for script in opts.scripts:
            try:
                run(sess, 'runscript %s' % script)
            except Exception:
                if not sess.ui.is_gui:
                    import traceback
                    traceback.print_exc()
                    return os.EX_SOFTWARE
                # Allow GUI to start up despite errors;
                if sess.debug:
                    import traceback
                    traceback.print_exc(file=sys.__stderr__)
                else:
                    sess.ui.thread_safe(sess.logger.report_exception, exc_info=sys.exc_info())
            except SystemExit as e:
                return e.code

    if not opts.silent:
        if sess.ui.is_gui and opts.debug:
            print("Finished initialization", flush=True)

    if not opts.silent:
        from chimerax.core.logger import log_version
        log_version(sess.logger)  # report version in log

    if opts.gui or opts.offscreen:
        sess.update_loop.start_redraw_timer()
        sess.logger.info('<a href="cxcmd:help help:credits.html">How to cite UCSF ChimeraX</a>',
                         is_html=True)

    # Show any messages from installing bundles on restart
    if restart_action_msgs:
        for where, msg in restart_action_msgs:
            if where == "stdout":
                sess.logger.info(msg)
            else:
                sess.logger.warning(msg)

    if opts.module or opts.run_path:
        import runpy
        import warnings
        exit = SystemExit(os.EX_OK)
        from chimerax.core.python_utils import chimerax_user_base
        with warnings.catch_warnings(), chimerax_user_base():
            warnings.filterwarnings("ignore", category=BytesWarning)
            global_dict = {
                'session': sess
            }
            try:
                if opts.module:
                    sys.argv = [opts.module] + args
                    runpy.run_module(opts.module, init_globals=global_dict,
                                     run_name='__main__', alter_sys=True)
                else:
                    sys.argv = args
                    runpy.run_path(opts.run_path, run_name='__main__')
            except SystemExit as e:
                exit = e
        if opts.module == 'pip' and exit.code == os.EX_OK:
            has_install = 'install' in sys.argv
            has_uninstall = 'uninstall' in sys.argv
            if has_install or has_uninstall:
                # TODO: --user is not given for uninstalls, so see
                # where the packages were installed to determine if
                # per_user should be true
                per_user = has_uninstall or '--user' in sys.argv
                sess.toolshed.reload(sess.logger, rebuild_cache=True)
                sess.toolshed.set_install_timestamp(per_user)
            # Do not remove scripts anymore since we may be installing
            # using ChimeraX which would put the right paths in
            # generated files.
            # if has_install:
            #     remove_python_scripts(chimerax.app_bin_dir)
        return exit.code

    from chimerax.core import startup
    startup.run_user_startup_scripts(sess)

    if opts.cmd:
        # Emulate Python's -c option.
        # This is needed for -m pip to work in some cases.
        # Also for pip, when it recursively calls sys.executable, it doesn't
        # always propagate the -I argument.  But the ChimeraX executable always
        # acts as if -I were passed in.  So add back '' to sys.path to compensate.
        sys.path.insert(0, '')  # get pip source installs to work
        sys.argv = ['-c'] + args
        global_dict = {
            'session': sess,
            '__name__': '__main__',
        }
        exec(opts.cmd, global_dict)
        return os.EX_OK

    # the rest of the arguments are data files
    from chimerax.core import commands
    for arg in args:
        if opts.safe_mode:
            # 'open' command unavailable; only open Python files
            if not arg.endswith('.py'):
                sess.logger.error("Can only open Python scripts in safe mode, not '%s'" % arg)
                return os.EX_SOFTWARE
            from chimerax.core.scripting import open_python_script
            try:
                open_python_script(sess, open(arg, 'rb'), arg)
            except Exception:
                if not sess.ui.is_gui:
                    import traceback
                    traceback.print_exc()
                    return os.EX_SOFTWARE
                # Allow GUI to start up despite errors;
                if sess.debug:
                    import traceback
                    traceback.print_exc(file=sys.__stderr__)
                else:
                    sess.ui.thread_safe(sess.logger.report_exception, exc_info=sys.exc_info())
        else:
            from chimerax.core.commands import StringArg
            try:
                commands.run(sess, 'open %s' % StringArg.unparse(arg))
            except Exception:
                if not sess.ui.is_gui:
                    import traceback
                    traceback.print_exc()
                    return os.EX_SOFTWARE
                # Allow GUI to start up despite errors;
                if sess.debug:
                    import traceback
                    traceback.print_exc(file=sys.__stderr__)
                else:
                    sess.ui.thread_safe(sess.logger.report_exception, exc_info=sys.exc_info())

    # Open files dropped on application
    if opts.gui:
        sess.ui.open_pending_files(ignore_files=(args if bad_drop_events else []))

    # By this point the GUI module will have redirected stdout if it's going to
    if bool(os.getenv("DEBUG")):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    # Allow the event_loop to be disabled, so we can be embedded in
    # another application
    if event_loop and opts.event_loop:
        try:
            sess.ui.event_loop()
        except SystemExit as e:
            return e.code
        except Exception:
            import traceback
            traceback.print_exc()
            return os.EX_SOFTWARE
    elif opts.gui:
        sess.ui.quit()  # Clean up gui to avoid errors at exit.
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
        from chimerax.linux import _xdg
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


_restart_toolshed_url = None


def restart_action(line, inst_dir, msgs):
    # Each line is expected to start with the bundle name/filename
    # followed by additional pip flags (e.g., --user)
    from chimerax.core import toolshed
    import sys
    import subprocess
    import os
    global _restart_toolshed_url
    parts = line.rstrip().split('\t')
    action = parts[0]
    # Options should match those in toolshed
    # Do not want to import toolshed yet, so we duplicate the code
    if action == "install":
        if _restart_toolshed_url is None:
            _restart_toolshed_url = toolshed.default_toolshed_url()
        bundles = parts[1]
        pip_args = parts[2:]
        command = [
            "install", "--extra-index-url", _restart_toolshed_url + "/pypi/",
            "--upgrade-strategy", "only-if-needed", "--no-warn-script-location",
            "--upgrade",
        ]
    elif action == "uninstall":
        bundles = parts[1]
        pip_args = parts[2:]
        command = ["uninstall", "--yes"]
    elif action == "toolshed_url":
        # Warn if already set?
        _restart_toolshed_url = parts[1]
        return
    else:
        msgs.append(("stderr", "unexpected restart action: %s" % line))
        return
    command.extend(pip_args)
    for bundle in bundles.split():
        if bundle.endswith(".whl"):
            command.append(os.path.join(inst_dir, bundle))
        else:
            command.append(bundle)
    cp = subprocess.run([sys.executable, "-m", "pip"] + command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
    if cp.returncode == 0:
        msgs.append(("stdout", "Successfully installed %r" % bundle))
    else:
        msgs.append(("stderr", "Error installing %r" % bundle))
    if cp.stdout:
        msgs.append(("stdout", cp.stdout.decode("utf-8", "backslashreplace")))
    if cp.stderr:
        msgs.append(("stderr", cp.stderr.decode("utf-8", "backslashreplace")))
    if bundle.endswith(".whl"):
        os.remove(os.path.join(inst_dir, bundle))


if __name__ == '__main__':
    exit_code = init(sys.argv)
    raise SystemExit(exit_code)
