# vi: set expandtab shiftwidth=4 softtabstop=4:
# Copyright © 2014 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.

__version__ = "0.1.0a0"
__app_name__ = "Chimera2"
__app_author__ = "UCSF"

import sys
import os


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
    opts.version = False

    # Will build usage string from list of arguments
    ARGS = [
        "--debug",
        "--nogui",
        "--help",
        "--lineprofile",
        "--listfiletypes",
        "--silent",
        "--nostatus",
        "--notools",
        "--version",
    ]
    if sys.platform.startswith("win"):
        ARGS.extend(["--console", "--noconsole"])
    USAGE = '[' + "] [".join(ARGS) + ']'
    USAGE += " or -m module_name [args]"
    # add in default argument values
    ARGS += [
        "--nodebug",
        "--gui",
        "--nolineprofile",
        "--nosilent",
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
        for a in ARGS:
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
        print("usage: %s %s\n" % (argv[0], USAGE), file=sys.stderr)
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
        elif opt in ("--tools", "--notools"):
            opts.load_tools = opt[2] == 't'
        elif opt == "--version":
            opts.version = True
    if help:
        print("usage: %s %s\n" % (argv[0], USAGE), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)
    if opts.list_file_types:
        opts.gui = False
        opts.silent = True
    return opts, args


def init(argv, app_name=None, app_author=None, version=None, event_loop=True):
    if app_name is None:
        app_name = __app_name__
    if app_author is None:
        app_author = __app_author__
    if version is None:
        version = __version__
    opts, args = parse_arguments(argv)
    if opts.version:
        print(version)
        return os.EX_OK

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

    from chimera.core import session
    sess = session.Session()
    sess.app_name = app_name
    sess.debug = opts.debug
    session.common_startup(sess)
    # or:
    #   sess.add_state_manager('scenes', session.Scenes(sess))
    #   from chimera.core import models
    #   sess.add_state_manager('models', models.Models(sess))
    # etc.

    # figure out the user/system directories for application
    if sys.platform.startswith('linux'):
        executable = os.path.abspath(sys.argv[0])
        bindir = os.path.dirname(executable)
        if os.path.basename(bindir) == "bin":
            configdir = os.path.dirname(bindir)
        else:
            configdir = bindir
        os.environ['XDG_CONFIG_DIRS'] = configdir
    import appdirs
    partial_version = '%s.%s' % tuple(version.split('.')[0:2])
    ad = sess.app_dirs = appdirs.AppDirs(app_name, appauthor=app_author,
                                         version=partial_version)
    # inform the C++ layer of the appdirs paths
    from chimera.core import _appdirs
    _appdirs.init_paths(os.sep, ad.user_data_dir, ad.user_config_dir,
                        ad.user_cache_dir, ad.site_data_dir,
                        ad.site_config_dir, ad.user_log_dir)

    # initialize the user interface
    if opts.gui:
        from chimera.core import gui
        ui_class = gui.UI
    else:
        from chimera.core import nogui
        ui_class = nogui.UI
    # sets up logging, splash screen if gui
    # calls "sess.save_in_session(self)"
    sess.ui = ui_class(sess)
    # splash step "0" will happen in the above initialization
    num_splash_steps = 4 if opts.gui else 3
    import itertools
    splash_step = itertools.count()

    if not opts.silent:
        sess.ui.splash_info("Getting preferences",
                            next(splash_step), num_splash_steps)
    from chimera.core import preferences
    # Only pass part of session needed in function call
    preferences.init(sess.app_dirs)

    # common core initialization
    if not opts.silent:
        sess.ui.splash_info("Initializing core",
                            next(splash_step), num_splash_steps)

    if not opts.silent:
        sess.ui.splash_info("Initializing tools",
                            next(splash_step), num_splash_steps)
    from chimera.core import toolshed
    # toolshed.init returns a singleton so it's safe to call multiple times
    sess.toolshed = toolshed.init(sess.logger, sess.app_dirs, debug=sess.debug)
    from chimera.core import tools
    sess.tools = tools.Tools(sess)
    sess.add_state_manager('tools', sess.tools)
    from chimera.core import tasks
    sess.tasks = tasks.Tasks(sess)
    sess.add_state_manager('tasks', sess.tasks)

    if opts.list_file_types:
        from chimera.core import io
        io.print_file_types()
        raise SystemExit(0)

    if opts.gui:
        # build out the UI, populate menus, create graphics, etc.
        if not opts.silent:
            sess.ui.splash_info("Starting main interface",
                                next(splash_step), num_splash_steps)
        sess.ui.build(opts.load_tools)

    if not opts.silent:
        sess.ui.splash_info("Finished initialization",
                            next(splash_step), num_splash_steps)

    if opts.module:
        import runpy
        import warnings
        sys.argv[:] = [opts.module] + args  # argv[0] is now module name
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=BytesWarning)
            runpy._run_module_as_main(opts.module, True)
        return os.EX_OK

    # the rest of the arguments are data files
    for arg in args:
        sess.models.open(arg)

    # Allow the event_loop to be disabled, so we can be embedded in
    # another application
    if event_loop:
        try:
            sess.ui.event_loop()
        except SystemExit as e:
            return e.code
    return os.EX_OK

if __name__ == '__main__':
    raise SystemExit(init(sys.argv))
