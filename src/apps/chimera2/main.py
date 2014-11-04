# vim: set expandtab shiftwidth=4 softtabstop=4:
# Copyright © 2014 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.

__version__ = "0.1.0a0"
__app_name__ = "Chimera"
__app_author__ = "UCSF"

import sys
import os


def parse_arguments(argv):
    """Initialize Chimera application."""
    import getopt

    class Opts:
        pass
    opts = Opts()
    opts.debug = False
    opts.gui = True
    opts.module = None
    opts.line_profile = False
    opts.load_tools = True
    opts.status = True
    opts.version = False

    # Will build usage string from list of arguments
    ARGS = [
        "--debug",
        "--nogui",
        "--help",
        "--lineprofile",
        "--nostatus",
        "--notools",
        "--version",
        "-m module",    # like Python, but doesn't end argument processing
    ]
    if sys.platform.startswith("win"):
        ARGS.extend(["--console", "--noconsole"])
    USAGE = '[' + "] [".join(ARGS) + ']'
    # add in default argument values
    ARGS += [
        "--gui",
        "--status",
    ]
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
        if opt == "--debug":
            opts.debug = True
        elif opt == "--help":
            help = True
        elif opt == "--gui":
            opts.gui = True
        elif opt == "--nogui":
            opts.gui = False
        elif opt == "--lineprofile":
            opts.line_profile = True
        elif opt == "--status":
            opts.status = True
        elif opt == "--nostatus":
            opts.status = False
        elif opt == "--notools":
            opts.load_tools = False
        elif opt == "--version":
            opts.version = True
        elif opt == "-m":
            opts.module = optarg
    if help:
        print("usage: %s %s\n" % (argv[0], USAGE), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)
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
        raise SystemExit(os.EX_OK)

    import envguard
    envguard.init(app_name)

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
    import cpp_appdirs
    cpp_appdirs.init_paths(os.sep, ad.user_data_dir, ad.user_config_dir,
        ad.user_cache_dir, ad.site_data_dir, ad.site_config_dir, ad.user_log_dir)

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
    num_splash_steps = 4
    import itertools
    splash_step = itertools.count()

    sess.ui.splash_info("Getting preferences",
        next(splash_step), num_splash_steps)
    from chimera.core import preferences
    # Only pass part of session needed in function call
    preferences.init(sess.app_dirs)

    # common core initialization
    sess.ui.splash_info("Initializing core",
        next(splash_step), num_splash_steps)
    session.common_startup(sess)
    # or:
    #   sess.scenes = session.Scenes(sess)
    #   from chimera.core import models
    #   sess.models = models.Models(sess)
    # etc.

    sess.ui.splash_info("Initializing tools",
        next(splash_step), num_splash_steps)
    from chimera.core import toolshed
    sess.tools = toolshed.init(sess.app_dirs)

    # build out the UI, populate menus, create graphics, etc.
    sess.ui.splash_info("Starting main interface",
        next(splash_step), num_splash_steps)
    sess.ui.build()

    # unless disabled, startup tools
    if opts.load_tools:
        # This needs sess argument because tool shed is session-independent
        for tool in sess.tools.startup_tools(sess):
            tool.start(sess)

    # the rest of the arguments are data files
    for arg in args:
        sess.models.open(arg)

    # Allow the event_loop to be disabled, so we can be embedded in
    # another application
    if event_loop:
        try:
            sess.ui.event_loop()
        except SystemExit:
            raise
        raise SystemExit(os.EX_OK)

if __name__ == '__main__':
    raise SystemExit(init(sys.argv))
