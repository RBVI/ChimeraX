# Copyright © 2014 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.

__version__ = "0.0.1"

import sys
import os

def parse_arguments(argv):
    """Initialize Chimera application."""
    import getopt
    class Args: pass
    args = Args()
    args.debug = False
    args.gui = True
    args.module = None
    args.line_profile = False
    args.load_tools = True
    args.status = True
    args.version = False

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
        opts, args = getopt.getopt(argv[1:], shortopts, longopts)
    except getopt.error as message:
        print("%s: %s" % (argv[0], message), file=sys.stderr)
        print("usage: %s %s\n" % (argv[0], USAGE), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)

    help = False
    for opt, optarg in opts:
        if opt == "--debug":
            args.debug = True
        elif opt == "--help":
            help = True
        elif opt == "--gui":
            args.gui = True
        elif opt == "--nogui":
            args.gui = False
        elif opt == "--lineprofile":
            args.line_profile = True
        elif opt == "--status":
            args.status = True
        elif opt == "--nostatus":
            args.status = False
        elif opt == "--notools":
            args.load_tools = False
        elif opt == "--version":
            args.version = True
        elif opt == "-m":
            args.module = optarg
    if help:
        print("usage: %s %s\n" % (argv[0], USAGE), file=sys.stderr)
        raise SystemExit(os.EX_USAGE)
    return args

def init(argv, app_name="Chimera"):
    args = parse_arguments(argv)
    if args.version:
        print(__version__)  # TODO
        raise SystemExit(os.EX_OK)

    import envguard
    envguard.init(app_name)

    # install line_profile decorator
    import builtins
    if not args.line_profile:
        builtins.__dict__['line_profile'] = lambda x: x
    else:
        # write profile results on exit
        import line_profiler, atexit
        prof = line_profiler.LineProfiler()
        builtins.__dict__['line_profile'] = prof
        atexit.register(prof.dump_stats, "%s.lprof" % app_name)

    from chimera.core import session
    sess = session.Session()
    sess.app_name = app_name
    sess.debug = args.debug

    if sys.platform.startswith('linux'):
        executable = os.path.abspath(sys.argv[0])
        bindir = os.path.dirname(executable)
        if os.path.basename(bindir) == "bin":
            configdir = os.path.dirname(bindir)
        else:
            configdir = bindir
        os.environ['XDG_CONFIG_DIRS'] = configdir
    import appdirs
    sess.app_dirs = appdirs.AppDirs(app_name, author="UCSF")    # leave out version

    if args.gui:
        from chimera.core import gui
        ui = gui.UI(sess)
    else:
        from chimera.core import nogui
        ui = nogui.UI(sess)
    sess.ui = ui
    ui.init(sess)   # sets up logging, splash screen if gui

    from chimera.core import preferences
    preferences.init(sess.app_dirs)

    # common core initialization
    session.common_startup(sess)
    # or:
    #   sess.scenes = session.Scenes(sess)
    #   from chimera.core import models
    #   sess.models = models.Models(sess)
    # etc.

    from chimera.core import toolshed
    sess.tools = toolshed.init(sess.app_dirs, load_tools=args.load_tools)

    ui.build(sess, sess.tools.tool_info())

    if args.tools:
        for tool in sess.tools.startup_tools():
            tool.start(sess)

    for arg in args:
        sess.models.open(sess, arg)

    try:
        ui.event_loop(sess)
    except SystemExit:
        raise
    raise SystemExit(os.EX_OK)

if __name__ == '__main__':
    raise SystemExit(init(sys.argv))
