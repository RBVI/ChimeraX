..  vim: set expandtab shiftwidth=4 softtabstop=4:

====================
Chimera2 Application
====================

The Chimera2 application should work on Microsoft Windows, Apple Mac OS X,
and Linux.

For the developer,
command line arguments are used to access functionality that is accessible
by the user.

Command Line Arguments
======================

When running Chimera2 from a terminal, *a.k.a.*, a shell, it can be given
various options followed by data files.
The data files are specified with same syntax as the filename argument
of Models' :py:func:`~chimerac.core.models.Models.open`.

Command Line Options
--------------------

In particular, the follow command line arguments are useful:

``--debug``
    Turn on debugging code.  Accessing within Chimera2 with ``session.debug``.
    
``--nogui``
    Turn off the gui.  Access with Chimera2 with ``session.ui.is_gui``.

``--lineprofile``
    Turn on line profiling.  See `Line Profiling`_ for details.

``--listfiletypes``
    Show all recognized file suffixes and if they can be opened or
    exported.

``--silent``
    Don't output startup splash text and otherwise refrain from being
    verbose.

``--nostatus``
    Don't output to status line.

``--notools``
    Do not autostart any tools at startup.

``--uninstall``
    If needed, deregister any icons or mime types,
    then remove as much of the installation directory as possible.

``--usedefaults``
    Ignore user settings and use default settings.

``--version``
    Print out current version.
    If given two times,
    then all of installed Chimera2 tools verions are listed.
    If given three times,
    then all of installed Python package versions are listed.

``-m module``
    Only recognized if it is the first argument.
    Act like the Python interpreter and run the module as the main module
    and the rest of the arguments are in :py:obj:`sys.argv`.
    Implies ``--nogui`` and ``--silent``.
    This is done after Chimera2 has started up, so a Chimera2 session
    is available in the global variable ``Chimera2_session``.


Initializing the Session
========================

A :py:class:`~chimera.core.session.Session` instance is passed as an
argument to many functions.
It is the way to access per-session data.
Leaf functions frequently are only given one attribute (or none at all).

``session.debug``
    True if debugging.

``session.logger``
    A :py:class:`~chimera.core.logger.Log` instance to log errors to.

``session.app_dirs``
    A versioned :py:class:`~appdirs.AppDirs` instance with directories
    to look for application and user files in.

``session.app_dirs_unversioned``
    An unversioned :py:class:`~appdirs.AppDirs` instance with directories
    to look for application and user files in.

``session.app_data_dir``
    The location of "share" directory.

``session.ui``
    A :py:class:`~chimera.core.logger.Log` instance.

``session.toolshed``
    A :py:class:`~chimera.core.toolshed.Toolshed` instance.

``session.tools``
    A :py:class:`~chimera.core.tools.Tools` instance.

``session.tasks``
    A :py:class:`~chimera.core.tasks.Tasks` instance.

Other initial sessiona attributes are initialized in :py:func:`chimera.core.session.common_startup`.

Line Profiling
==============

    Line profiling is based on `Robert Kern's <https://github.com/rkern>`_
    `line_profiler <https://github.com/rkern/line_profiler>`_ package.
    Differences from the conventional setup are given in parenthesizes.

    There are five parts to profiling:

    1. Decorate functions that you wish to profile with the
       ``@line_profile`` decorator and install them.
       This decorator is a no-op if Chimera2 is not profiled.
       (Instead of the conventional ``@profile``.)

    2. Run ``Chimera2 --lineprofile`` from the command line.
       (Instead of using ``kernprof``.)
       This generates a ``Chimera2.lprof`` file in the current directory.

    3. Get your profiling results by running
       ``Chimera2 -m line_profiler Chimera2.lprof``.

    4. Analyze your results, possibly edit your code, and return to step 1.

    5. Remove the function decorators before committing your changes.
