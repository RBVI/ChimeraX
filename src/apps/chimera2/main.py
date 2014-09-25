# Copyright © 2014 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.

__version__ = "0.0.1"

import os

debug = False
gui = True
module = None
line_profile = False
tools = True
status = True
version = False

def parse_arguments():
    """Initialize Chimera application.

    Return exit status instead of raising SystemExit,
    so it can be embedded someday.
    """
    import getopt
    global debug, gui, module, line_profile, status, version

    # Will build usage string from list of arguments
    ARGS = [
        "--debug",
        "--nogui",
        "--help",
        "--lineprofile",
        "--nostatus",
        "--notools",
        "--version",
        "-m module",    # emulate Python
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
    except getopt.error, message:
        print("%s: %s" % (argv[0], message), file=sys.stderr)
        print("usage: %s %s\n" % (argv[0], USAGE), file=sys.stderr)
        return os.EX_USAGE

    help = False
    for opt, optarg in opts:
        if opt == "--debug":
            debug = True
        elif opt == "--help":
            help = True
        else opt == "--gui":
            gui = True
        else opt == "--nogui":
            gui = False
        else opt == "--lineprofile":
            line_profile = True
        else opt == "--status":
            status = True
        else opt == "--nostatus":
            status = False
        else opt == "--notools":
            tools = False
        else opt == "--version":
            version = True
        else opt == "-m":
            module = optarg
    if help:
        print("usage: %s %s\n" % (argv[0], USAGE), file=sys.stderr)
        return os.EX_USAGE

    return args

def init():
    args = parse_arguments()
    if version:
        print(__version__)  # TODO
        raise SystemExit(os.EX_OK)

    import envguard
    envguard.init("chimera")

    import builtins
    if not line_profile:
        builtins.__dict__['lineprofile'] = lambda x: x
    else:
        # install lineprofile decorator
        # and write results on exit
        import line_profiler, atexit
        prof = line_profiler.LineProfiler()
        builtins.__dict__['lineprofile'] = prof
        atexit.register(prof.dump_stats, "chimera.lprof")

    from chimera.core import session
    sess = session.Session()
    sess.debug = debug

    import sys
    if sys.platform.startswith('linux'):
        executable = os.path.abspath(sys.argv[0])
        bindir = os.path.dirname(executable)
        if os.path.basename(bindir) == "bin":
            configdir = os.path.dirname(bindir)
        else:
            configdir = bindir
        os.environ['XDG_CONFIG_DIRS'] = configdir
    import appdirs
    sess.app_dirs = appdirs.AppDirs("Chimera", author="UCSF")    # leave out version

    if gui:
        from chimera.core import gui
        ui = gui.UI()
    else:
        from chimera.core import nogui
        ui = nogui.UI()
    sess.ui = ui
    ui.init()   # sets up logging, splash screen if gui

    from chimera.core import preferences
    preferences.init(sess.app_dirs)

    from chimera.core import setup
    setup.init()

    from chimera.core import toolshed
    tool_info = toolshed.init(sess.app_dirs, not tools)

    ui.build(tool_info)

    # TODO: initialize startup tools

    from chimera.core import io
    for arg in args:
        io.open(arg)

    try:
        ui.event_loop()
    except SystemExit:
        raise
    raise SystemExit(os.EX_OK)

if __name__ == '__main__':
    raise SystemExit(init())
