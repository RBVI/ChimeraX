..  vim: set expandtab shiftwidth=4 softtabstop=4:

================
ChimeraX Startup
================

The ChimeraX application runs on Microsoft Windows, Apple Mac OS X,
and Linux.

By default, ChimeraX does an asynchronous network query
of the ChimeraX Toolshed for available bundles at startup once a week.
The interval can be changed in the Toolshed settings.
The Toolshed query is implicitly turned off when certain command line arguments are given
as noted below.

Command Line Arguments
======================

When running ChimeraX from a terminal, *a.k.a.*, a shell, it can be given
various options followed by data files.
The data files are specified with same syntax as the filename argument
of Models' :py:func:`~chimerax.core.models.Models.open`.

Command line options can start with either a single dash or a double dash
and may not be mixed.

Single Dash Commmand Line Options
---------------------------------

All of the Python single dash command line options are recognized.
The presence of any single dash argument turns off querying the Toolshed at startup.
Most are ignored except for the following:

``-c command``
    Only recognized if it is the first argument.
    Act like the Python interpreter and run the Python command
    with the rest of the arguments in :py:obj:`sys.argv`.
    Implies ``--nogui`` and ``--silent``.
    This is done after ChimeraX has started up, so a ChimeraX session
    is available in the global variable ``session``.

``-d``
    Turn on debugging.

``-h``
    Show command line help.

``-m module``
    Only recognized if it is the first argument.
    Act like the Python interpreter and run the module as the main module
    and the rest of the arguments are in :py:obj:`sys.argv`.
    Implies ``--nogui`` and ``--silent``.
    This is done after ChimeraX has started up, so a ChimeraX session
    is available in the global variable ``session``.
    The module name is ``__main__`` instead of a sandbox name that
    is used for normal :py:mod:`~chimerax.core.scripting`.

``-u``
    Force the stdout and stderr streams to be unbuffered.

``-V``
    Prints the internal Python version and exits.


Double Dash Command Line Options
--------------------------------

The follow command line arguments are recognized:

``--cmd command``

    Run the ChimeraX command at startup after starting tools.
    Turns off querying the Toolshed at startup.

``--color``
    Turn on colored text in nogui mode (default).

``--nocolor``
    Turn off colored text in nogui mode.

``--debug``
    Turn on debugging code.  Accessing within ChimeraX with ``session.debug``.

``--devel``
    Turn on development mode.  Currently, just enables Python deprecation warnings.

``--exit``
    Exit immediately after processing command line arguments.
    Turns off querying the Toolshed at startup.

``--noexit``
    Do not exit after processing command line arguments (default).
    Turns off querying the Toolshed at startup (not default).

``--lineprofile``
    Turn on line profiling.  See `Line Profiling`_ for details.
    Turns off querying the Toolshed at startup.

``--listioformats``
    Show all recognized file suffixes and if they can be opened or saved.
    Turns off querying the Toolshed at startup.
    
``--nogui``
    Turn off the gui.  Access with ChimeraX with ``session.ui.is_gui``.

``--nostatus``
    Don't output to status line.

``--notools``
    Do not autostart any tools at startup.

``--offscreen``
    Run without a gui but allow rendering images that can be saved to files,
    *i.e.*, implies ``--nogui``.
    This uses OSMesa for rendering which will not make use of
    a GPU, so rendering can be slow.
    But it can run on a server without a display.

``--safemode``
    Don't run bundle custom initialization code nor load any tools.

``--script python-script``
    Run Python script at startup.
    If they Python script has any specific arguments,
    they should be quoted along with the script name.
    Turns off querying the Toolshed at startup.
    
``--silent``
    Don't output startup splash text and otherwise refrain from being
    verbose.

``--start_tool tool_name``

    Start the named tool during ChimeraX startup after the autostart tools.

``--tools``
    Run ChimeraX tools at startup (default).

``--notools``
    Don't run ChimeraX tools at startup.

``--toolshed URL``
    Set the URL to use for the toolshed.
    The special name **preview** is recognized for using a preview of
    the next revision of the toolshed (currently only available internally).

``--uninstall``
    If needed, deregister any icons or mime types,
    then remove as much of the installation directory as possible.
    Intended for use by system App Store or package manager.

``--usedefaults``
    Ignore user settings and use default settings.
    Not implemented yet.

``--version``
    Print out current version.
    If given two times,
    then all of installed ChimeraX tools verions are listed.
    If given three times,
    then all of installed Python package versions are listed.

Run Custom Python Code at Start Up
==================================

To have ChimeraX run custom Python code each time you start it you can put Python files
in directory

	~/chimerax_start

Each Python file will be executed with the variable "session" added to the global namespace.
For example, the following line put in ~/chimerax_start/starttools.py automatically starts
the File History panel and Density Map toolbar.

        session.tools.start_tools(('File History', 'Density Map Toolbar'))

Python code can be used to register new commands, add mouse modes and file readers that you
develop.  In addition to executing Python files in the directory, the startup directory will
be appended to the Python sys.path search path so Python modules in the subdirectory can be
imported.  Subdirectories in the startup directory that contain an __init__.py file will be
imported and if they contain a function named "start" it will be called with session as
an argument.

To use a directory other than ~/chimerax_start as the startup directory set the environment
variable CHIMERAX_START to the desired directory in the shell where Chimera is started.

Initializing the Session
========================

A :py:class:`~chimerax.core.session.Session` instance is passed as an
argument to many functions.
It is the way to access per-session data.
Leaf functions frequently are only given one attribute (or none at all).

``session.debug``
    True if debugging.

``session.logger``
    A :py:class:`~chimerax.core.logger.Log` instance to log errors to.

``session.app_dirs``
    A versioned :py:class:`~appdirs.AppDirs` instance with directories
    to look for application and user files in.

``session.app_dirs_unversioned``
    An unversioned :py:class:`~appdirs.AppDirs` instance with directories
    to look for application and user files in.

``session.app_data_dir``
    The location of "share" directory.

``session.ui``
    A :py:class:`~chimerax.core.logger.Log` instance.

``session.toolshed``
    A :py:class:`~chimerax.core.toolshed.Toolshed` instance.

``session.tools``
    A :py:class:`~chimerax.core.tools.Tools` instance.

``session.tasks``
    A :py:class:`~chimerax.core.tasks.Tasks` instance.

Other initial sessiona attributes are initialized in :py:func:`chimerax.core.session.common_startup`.

Line Profiling
==============

    Line profiling is based on `Robert Kern's <https://github.com/rkern>`_
    `line_profiler <https://github.com/rkern/line_profiler>`_ package.
    Support is restricted to platforms that have binaries in pypi.org (just Linux for now).
    Differences from the conventional setup are given in parenthesizes.

    There are five parts to profiling:

    1. Decorate functions that you wish to profile with the
       ``@line_profile`` decorator and install them.
       This decorator is a no-op if ChimeraX is not profiled.
       (Instead of the conventional ``@profile``.)

    2. Run ``ChimeraX --lineprofile`` from the command line.
       (Instead of using ``kernprof``.)
       This generates a ``ChimeraX.lprof`` file in the current directory.

    3. Get your profiling results by running
       ``ChimeraX -m line_profiler ChimeraX.lprof``.

    4. Analyze your results, possibly edit your code, and return to step 1.

    5. Remove the function decorators before committing your changes.

To use the :py:mod:`timeit` module, see the :py:mod:`~chimerax.core.scripting`
documentation.
