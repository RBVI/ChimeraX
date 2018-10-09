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

Building ChimeraX
=================

TODO: Fill out the details of requirements and procedures for building ChimeraX from scratch.

ChimeraX has been built on Windows 10, macOS 10.12 (Sierra), and Linux (version?).
ChimeraX is coded in Python (currently version 3.6) and C++.

A build is done using make by running

  make install

in the top-level directory of the ChimeraX source code.  This compiles third party-prequesite packages
such as Python 3, and installs many needed third party packages from PyPi using "pip install".  It also
installs various non-Python third-party libraries such as Qt 5 and HDF5 and several others.  It builds
wheels for about 100 ChimeraX-specific bundles written in Python and C++.  And it builds an application
with an appropriate directory structure the operating system.

The ChimeraX build uses a commercial license version of PyQt and gets this from plato.cgl.ucsf.edu and
requires a password or that the build machine can ssh to plato.

The prereqs subdirectory contains Makefiles to build the various third party packages.  Most of the ChimeraX
build time is making these prereq packages.  To save time on subsequent builds the installed packages are
archived in a file prereqs/prebuilt-<OS>.tar.bz2.

Nightly Builds
--------------

Nightly builds on Mac, Windows and Linux are build with scripts in the ChimeraX repository build_tools
subdirectory.

Windows Build
-------------

Microsoft Visual Studio Community 2015 is used to compile C++ on Windows.  This compiler was chosen
to match the compiler used by the standard Python 3 distribution.

Cygwin is use to provide a unix-like environment (bash shell, make, ...) to build ChimeraX on Windows.
Several cygwin packages are needed beyond the standard cygwin install.

  - make
  - unzip
  - patch
  - rsync
  - ImageMagick

ChimeraX also uses the standard Python 3.6 distribution so that should be installed from Python on the
build machine.  It requires an exact version (currently Python 3.6.4) specified in chimerax/prereqs/Python/Makefile.

macOS Build
-----------

XCode 9 compilers are currently used.

Linux Build
-----------

A virtual machine with a long list of required packages are used to compile ChimeraX on Linux.
A Debian package is created for distribution.  We are working on building RPM packages.
