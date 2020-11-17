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

Packages used by ChimeraX
=========================

ChimeraX depends on over 50 other packages.  Here is the list from September 2020.

Main Packages
-------------

* Python - most of ChimeraX is Python code
* PyQt5 - Qt window toolkit
* PyOpenGL - Python interface to OpenGL graphics used to render all graphics
* numpy - arrays, volume data, atomic coordinates, ... (from PyPi)

Packages from PyPi Python package manager
-----------------------------------------
* appdirs - cross-platform access to application directories, variable chimerax.app_dirs uses it.
* colorama - color terminal output for nogui mode
* comtypes - used for space navigator device input on Windows
* Cython - required by pytables which is the python interface to hdf5, for reading hdf5 density maps.  Also used to wrap C++ atomic data structures.
* distlib - used by toolshed for working with Python wheel files
* distro - Linux distribution info, used by bug reporter
* filelock - used by toolshed to lock files when installing bundles
* grako - grammar compiler for parsing atom specifiers in ChimeraX commands
* html2text - converts html to plain text, used to output log messages in no gui mode and for bug reports.
* ihm - library to read Integrative Hybrid Model files from PDBDev
* imagecodecs-lite - Reading Tiff image files light microscopy stacks with formats not handled by pillow.
* ipykernel - IPython kernel used by Jupyter in ChimeraX shell tool
* ipython - Jupyter kernel, used by ChimeraX shell which allows html rich output
* jupyter-client - front end for Jupyter kernel, used by ChimeraX shell
* lxml - used to convert html to plain text for nogui log messages, also for making bundle documentation index
* lz4 - very fast compression used when saving session files
* matplotlib - makes graphical plots for crosslinks and intefaces plots
* msgpack - format for saving ChimeraX sessions
* networkx - graph layout for interfaces plots
* openvr - OpenVR for using virtual reality headsets
* Pillow - Python image library for saving png, jpeg, tiff images
* pkginfo - read PKG-INFO files in wheels, used by toolshed
* psutil - process, OS and hardware info, used to set volume cache size based on physical memory available
* pycollada - Collada scene file format reader
* pydicom - reading DICOM files, pydicom uses gdcm for some image formats
* python-dateutil - extends Python's datetime module, used by numpy/pycollada/matplotlib and session save
* pywin32 - Windows operating system APIs, seems to only be used by WMI for bug reporting
* qtconsole - used to implement ChimeraX Python shell
* requests - simplifies handling HTTP requests; used in scripts for NIH3D printing service
* scipy - math/science library used to compute helix tubes
* six -	compatibility module for Python 2 and 3, used by numpy/matplotlib and more
* sortedcontainers - use sorted dictionary SortedDict for color and command managers
* suds-jurko - SOAP library, used for Opal web services
* tables - read HDF5 files, usually volume data
* tifffile - read TIFF image stacks for 3D light microscopy, high performance, 16-bit format support
* tinyarray - efficient small C arrays, used for atom coordinates, numpy is much less efficient
* WMI - Windows OS info used by bug reporter

Packages not from PyPi compiled for specific tools
--------------------------------------------------
* cxservices - used by remotecontrol command REST service
* expat - used to build the Python expat XML parser module, used by ChimeraX registration checking code
* ffmpeg - executable for recording videos, compiling ffmpeg uses these libraries

  * libogg - Ogg audio format
  * libtheora - Theora video format
  * libvpx - VP9 video format
  * x264 - H.264 video format
  * yasm - assembler for compiling optimized assembly code

* gdcm - used to read DICOM files, used by pydicom
* lineprofiler - used to profile python code, see command-line `--lineprofile <https://www.cgl.ucsf.edu/chimerax/docs/user/options.html>`_ option
* mesa - off-screen rendering in nogui mode using OSMesa (off-screen mesa), only used on Linux
  - llvm - compiler libraries used by offscreen-mesa to compile opengl shaders for image rendering in nogui mode.
  - pkgconfig - Required to build mesa, off-screen rendering.
* openmm - molecular dynamics used by tug mouse mode and ISOLDE plugin
* openssl - Transport Layer Security (TLS) support on Mac for fetching https urls
* zlib - used when compiling Python zlib module on mac and generic linux
  
PyPi packages used to generate ChimeraX documentation
-----------------------------------------------------
* blockdiag - generates block diagram image files
* docutils - html generation for Sphinx
* funcparserlib - used by blockdiag
* Jinja2 - used internally by Sphinx
* MarkupSafe - used by Jinja2
* numpydoc - doc string parsing of function arguments for Sphinx
* Pygments - syntax highlighting of code in Sphinx
* Sphinx - Generates ChimeraX programming documentation
* sphinxcontrib-blockdiag - extension for block diagrams in Sphinx, not sure if we ever use this
* webcolors - used by blockdiag

Packages used to do builds
--------------------------
Many packages in addition to these are required to be preinstalled on Linux, Windows and Mac build machines,
such as compilers, rsync, cygwin on Windows.  The needed build tools are listed on other pages (TODO: add link).

* chrpath - used for setting RPATH in linux ChimeraX executable
* imex - boilerplate for exporting symbols from C/C++ libraries
* libtool - used when building Qt from source (only done for debugging)
* p7zip - used to unpack *.xz source distributions Python, libtheora, libogg, libvpx, x264 on Mac

Packages from PyPi used to do builds
------------------------------------
* flake8 - Python syntax checker used by bundles/Makefile pylint target
  
  * mccabe - used by flake8
  * pep8 - used by flake8
  * pep8-naming - used by flake8
  * pycodestyle - used by flake8
  * pyflakes - used by flake8

* pytz - time zone calculations, could not find this used anywhere. TOOO: remove.
