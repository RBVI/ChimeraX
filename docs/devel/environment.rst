..  vim: set expandtab shiftwidth=4 softtabstop=4:

..
    === UCSF ChimeraX Copyright ===
    Copyright 2017 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

#############################################
Setting Up a ChimeraX Development Environment
#############################################
What follows are instructions for setting up an environment to develop ChimeraX
or bundles for ChimeraX on your personal machine where e.g. compiler versions need
not be as strict. This is OK for development and personal use; however, to merge
core code or distribute bundles, please ensure that you are using the same compilers
as the reference machines used to build the ChimeraX distribution, either on your
build machine or in your CI scripts.

=======
Windows
=======
Microsoft Visual Studio Community 2017 or 2019 is used to compile C++ on Windows. For
personal builds you may use whatever version of Visual Studio, but to distribute bundles
you should use one of these compilers, as they were chosen to match the compiler used by
the standard Python 3 distribution.

Install Visual Studio. When choosing components, select at least the following: ::

    "Desktop Development with C++"
    "Universal Windows Platform development"

Then start Visual Studio and login, then quit, to complete setup.

MinGW
=====
Necessary Packages
------------------
MinGW comes with the ``pacman`` package manager. You will need at least:
::

    git
    make
    mingw-w64-x86_64-binutils
    mingw-w64-x86_64-imagemagick
    openssh
    patch
    rsync
    unzip

You will also need to ensure that ``/mingw64/bin`` comes before
``C:\Windows\System32`` in your ``PATH``; otherwise ``convert.exe``
will take precedence over imagemagick's ``convert`` and the build
will fail attempting to generate the ChimeraX desktop icon.

You may optionally install the Git for Windows version of ``git``,
which is noticeably faster than \*nix git. Instructions for doing so
can be found elsewhere online and involve adding Git for Windows's
repositories to ``pacman.conf``.

Cygwin
======
Necessary Packages
------------------
Ensure the following packages are chosen in the Cygwin installer.
Cygwin has no package manager, so missing packages are installed
by re-running the Cygwin installer. ::

    binutils
    git
    ImageMagick
    make
    openssh
    patch
    rsync
    unzip

WSL
===
It is not recommended to use WSL to develop ChimeraX on Windows.
Depending on the version of Linux chosen, ChimeraX may or may not
work on WSL. Regardless, the use of ChimeraX on WSL is unsupported
at this time. ChimeraX will launch on an Ubuntu 20.04 WSLg installation;
however, floating windows are unresponsive (October 2022).

Using the Windows Terminal
==========================
To add a new profile to the Windows Terminal, first open it. Then, click
the dropdown arrow after the last open tab and click 'Settings'. Next,
look at the bottom of the sidebar (scroll if necessary) and find 'Add
a new profile'. Click it, and continue with the instructions for your
chosen environment.

Adding MinGW
------------
Set the command line to ::

    C:\\msys64\\msys2_shell.cmd -defterm -here -no-start -mingw64

If you prefer, you can install ``zsh`` and append ``-shell zsh`` to the
command.

To start MinGW with ``HOME`` set to your Windows home directory and not
``C:\msys64\home\``, open a MinGW shell and edit ``/etc/nssswitch.conf``.
Change the line ::

    db_home: cygwin desc

to ::

    db_home: windows

Also ensure that the start directory in the Windows Terminal menu is
set to the directory you'd like to start in.

Adding Cygwin
-------------
Set the command line to ::

    "C:\cygwin64\bin\bash.exe" --login -i

If you prefer, you can install ``zsh``

To start Cygwin in your Windows home directory and not ``C:\cygwin64\home``, open
``/etc/nsswitch.conf`` (which is blank by default) and add ::

    db_home: windows

Also ensure that the start directory in the Windows Terminal menu is
set to the directory you'd like to start in.

=====
macOS
=====
You will need the Xcode Command Line Tools at a minimum. To install them,
open a terminal and type ::

    xcode-select --install

Note that this does not install ``homebrew``.

=====
Linux
=====
While it is theoretically possible to develop ChimeraX on any Linux flavor,
you may find that one of the many binary prerequisites refuses to compile
on your platform. The easiest distribution to set up is Ubuntu 20.04, while
the "reference" ChimeraX is built on Rocky 8 with ``gcc-toolset-10``

Regardless of the distribution you choose to use for development, if you
are a bundle developer ensure that any continuous integration of your
bundle or any builds for distribution are done on the same platform as
ChimeraX if your bundle includes compiled code, to ensure compatibility
with the ChimeraX distribution.

=============
All Platforms
=============
Git associates a user name and email address with all check-ins. The defaults
are based on your :envvar:`USERNAME` and your computer's hostname. The email
address is usually wrong. To explicitly set those values, issue the following
commands in your shell of choice: ::

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

Recommended ``.gitconfig`` Settings
===================================
::

    pull.rebase = true
