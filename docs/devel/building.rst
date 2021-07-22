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

Compiling ChimeraX
==================

The source code is at GitHub `RBVI/ChimeraX repository <https://github.com/RBVI/ChimeraX>`_ with instructions for getting it :doc:`here <conventions>`.

ChimeraX depends on over 50
:doc:`third-party packages <dependencies>`
so it can be challenging to build.
We recommend using
`production releases <https://www.rbvi.ucsf.edu/chimerax/download.html#release>`_
or
`daily builds <https://www.rbvi.ucsf.edu/chimerax/download.html#daily>`_
from the
`ChimeraX download page <https://www.rbvi.ucsf.edu/chimerax/download.html>`_
instead of compiling it yourself.

ChimeraX has been built on Windows 10, macOS 10.12 through 10.15, and Linux (RedHat / CentOS 7 and 8, Ubuntu 20,18,16).
ChimeraX is coded in Python (currently version 3.7) and C++.

A build is done using make by running

  make install

in the top-level directory of the ChimeraX source code.  This compiles third party-prequesite packages
such as Python 3, and installs many needed third party packages from PyPi using "pip install".  It also
installs various non-Python third-party libraries.  It builds
wheels for about 100 ChimeraX-specific bundles written in Python and C++.  And it builds an application
with an appropriate directory structure the operating system.

The ChimeraX build as of December 2020 uses PySide2 for the Qt user interface.
Formerly it used a commercial license version of PyQt and this was fetched from plato.cgl.ucsf.edu and
required that the build machine can ssh to plato.

The prereqs subdirectory contains Makefiles to build the various third party packages.  Most of the ChimeraX
build time is making these prereq packages.  To save time on subsequent builds the installed packages are
archived in a file prereqs/prebuilt-<OS>.tar.bz2.

Nightly Builds
--------------

`Nightly builds <https://www.rbvi.ucsf.edu/chimerax/download.html#daily>`_
on Mac, Windows and Linux are build with scripts in the ChimeraX git repository build_tools on plato.cgl.ucsf.edu.


Windows Build
-------------

Microsoft Visual Studio Community 2017 or 2019 is used to compile C++ on Windows.
This compiler was chosen to match the compiler used by the standard Python 3 distribution.

Cygwin is use to provide a unix-like environment (bash shell, make, ...) to build ChimeraX on Windows.

Steps for compiling ChimeraX on Windows 10:

#. Install Microsoft Visual Studio 2017 or 2019 Community Edition::

    Select Programming Languages / Visual C++ (not done by default).
    Select "Windows and Web Development / Universal Windows App Development Tools".
    Make sure that Windows SDK 10.0.18362 is installed (matching vsvars.sh).
    Then start Visual Studio and login, then quit.
    
#. Install ​Cygwin, 64-bit version. In addition to the default packages, you'll need::

    binutils - xdr in md_crds needs ld.exe
    git - to be able to check in changes
    Imagemagick - to create icon files
    make - to run the build process
    openssh - to get network access to plato
    patch - to patch source distribution
    rsync - to install files and fetch them
    unzip - used to build ffmpeg
  
#. Clone the `ChimeraX repository <https://github.com/RBVI/ChimeraX>`_ from GitHub.

#. Run ". vsvars.sh" in chimerax root directory to set path to Visual Studio compiler.

#. "make install" in the repository root.

Linux Build
-----------
A long list of required packages are used to compile ChimeraX on Linux.
We currently use Singularity images to compile Linux ChimeraX versions on virtual machines
including Redhat / CentOS 7 and 8 and Ubuntu 16, 18, and 20.
Debian and RPM packages are created for distribution.

The definition files for the virtual machine images list the required packages and
are currently part of the private build_tools git repository on plato.cgl.ucsf.edu.
TODO: Definition files should be moved to ChimeraX repository at GitHub.

Here is an example of the requirements for building on Ubuntu 20.04::

    sudo apt-get install -y --no-install-recommends \
	language-selector-common ffmpeg vim-nox \
	build-essential g++ gfortran \
	chrpath patch openssh-client rsync ca-certificates \
	make autoconf automake libtool pkg-config \
	flex bison yasm unzip tar fakeroot curl wget \
	python2.7 \
	imagemagick \
	libasound2-dev \
	libatk1.0-dev \
	libbz2-dev \
	libcairo2-dev \
	libcups2-dev \
	libdbus-1-dev \
	libegl1-mesa-dev \
	libexpat1-dev \
	libffi-dev \
	libfftw3-dev \
	libfreetype6-dev \
	libgdk-pixbuf2.0-dev \
	libgl1-mesa-dev \
	libglu1-mesa-dev \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev \
	libgtk-3-dev \
	libhdf5-dev \
	libicu-dev \
	libjpeg-dev \
	liblzma-dev \
	libmysqlclient-dev \
	libnss3-dev \
	libosmesa6-dev \
	libpango1.0-dev \
	libpcre3-dev \
	libpng-dev \
	libpulse-dev \
	libssl-dev \
	libtiff-dev \
	libx11-dev \
	libxcomposite-dev \
	libxcursor-dev \
	libxft-dev \
	libxi-dev \
	libxmu-dev \
	libxpm-dev \
	libxrandr-dev \
	libxslt1-dev \
	libxss-dev \
	libxt-dev \
	libxtst-dev \
	llvm-dev \
	x11proto-scrnsaver-dev \
	xdg-utils \
	zlib1g-dev


macOS Build
-----------

XCode compilers are used.  Tested with XCode version 12.2 (Jan 2021) on macOS 10.15 (Catalina) and 11 (Big Sur).
  
#. Clone the `ChimeraX repository <https://github.com/RBVI/ChimeraX>`_ from GitHub::

     git clone git@github.com:RBVI/ChimeraX.git chimerax

#. In the repository chimerax directory run make to build the application::

    make build-from-scratch >& make.out

macOS with ARM CPUs
^^^^^^^^^^^^^^^^^^^

A native ARM CPU build of ChimeraX has not yet been made (July 2021).  We have made a
partly functional version and it was 1-2 times faster than Intel ChimeraX running under
Rosetta 2 emulation. A primary obstacle is PyQt5 is not distributed for Mac ARM CPUs.
Homebrew provides a native PyQt5 without QtWebEngine which we have tried.  The missing
QtWebEngine disables some ChimeraX tools like the Log panel.  Progress on a native
Mac ARM distribution is described in ChimeraX ticket
`#4663 <https://www.rbvi.ucsf.edu/trac/ChimeraX/ticket/4663>`_.
