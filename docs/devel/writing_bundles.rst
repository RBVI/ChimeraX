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

Building and Distributing Bundles
=================================

A *bundle* is a collection of code and data that can be added to
ChimeraX to provide support for new graphical tools, commands,
file formats, web databases and selection specifiers.
This document describes the details of how to create a bundle
and publish it in the ChimeraX toolshed.
There is also a step-by-step example of writing a bundle
available :ref:`here <A Few Steps>`.

Bundle Format
-------------

A ChimeraX bundle is packaged as a Python `wheel
<https://packaging.python.org/wheel_egg/>`_.

A wheel usually contains Python and/or compiled code
along with additional resources such as icons,
data files, and documentation.  While the
general Python wheel specification supports installing
files into arbitrary location, ChimeraX bundles
are limited to provide a single folder/directory,
which may be installed using the ``toolshed install``
command.  Bundle folders are typically placed in a
per-user location, which may be listed using the
``toolshed cache`` command.

It is possible but not recommended to use *pip* to
install a bundle.  ChimeraX maintains a bundle
metadata cache for fast initialization, which
*pip* will not update, and therefore the bundle
functionality may not be available even though
the wheel is installed.  In this event, try running
the ``toolshed refresh`` command to force an update.

Creating bundles follows the same basic
procedure as `creating Python wheel
<https://packaging.python.org/distributing/>`_,
with a few ChimeraX customizations.
The most straightforward way is to start
with some ChimeraX sample code and modify it appropriately.

Bundle Sample Code
------------------

To build a bundle from the `sample code
<https://www.cgl.ucsf.edu/chimerax/cgi-bin/bundle_sample.zip>`_,
you can either use the ``make`` program, or the
ChimeraX application if you do not have ``make``.
On Linux and macOS, ``make`` is available as part of the
developer package.  On Windows, ``make`` is
available as part of `Cygwin <https://cygwin.com>`_.

Because the sample code includes C++ source code that
need to be compiled, you will need a C++ compiler for
the build.  On Windows, we use Microsoft Visual
Studio, Community 2015.  On the Mac, we use ``Xcode``.
On Linux, ``gcc`` and ``g++`` are available in different
packages depending on the flavor of Linux.

The sample code is organized with "administrative" code
at the top level and actual bundle code in the ``src``
folder.  Administrative code, with the exception of
license text, is only used for building the bundle.
All other contents of the bundle should be in ``src``.


*Administrative Files*

    **Makefile** is the configuration file used by
    the ``make`` command.  This file is not used
    if you use the ``devel`` command to build and
    install your bundle.)

    **README** contains a pointer back to this document.

    **bundle_info.xml** is an XML file containing
    information about the bundle, including its name,
    version, dependencies, *etc*.  This file is
    used when you use the ``devel`` command to build and
    install your bundle.

    **license.txt.bsd** and **license.txt.mit** are
    two sample license text files.  The actual file
    used when building the bundle is **license.txt**
    which must exist.  For testing, simply renaming
    one of the sample license text file is sufficient.
    You may want to use a custom license for your
    actual bundle.

    **setup.py.in** contains Python code for building
    the bundle.  This file is a remnant from when
    bundles were built using the Python interpreter
    instead of ChimeraX It is here only as a potential
    starting point for developers who need greater
    control over the build process.

    **setup.cfg** is the configuration file used when
    **setup.py** is run.  This file should not be modified.


*Bundle Source Code Files*

    **__init__.py** contains the bundle initialization
    code.  Typically, it defines a subclass of the
    ``chimerax.core.toolshed.BundleAPI`` class and
    instantiates a single instance named ``bundle_api``.
    ChimeraX communicates with the bundle through this
    singleton, which must conform to the `bundle API`.

    **cmd.py** contains code called by ``bundle_api``
    from **__init__.py** for executing the ``sample``
    command.  Before deciding on the name and syntax
    of your own command, you should look at the
    :doc:`command style guide <command_style>`.

    **io.py** contains code called by ``bundle_api``
    from **__init__.py** for opening XYZ files.

    **tool.py** contains code called by ``bundle_api``
    from **__init__.py** for starting the graphical
    interface.

    **_sample_pyapi.cpp** and **_sample_pybind11.cpp**
    contain sample C++ code that demonstrate two
    possible ways of binding C++ to Python.
    Thye compile into Python modules that each define
    two module functions. Which binding gets used at
    runtime is determined by the ``api`` argument of
    the ``sample`` command.


*Bundle Help Files*

    This sample bundle does not provide any help files,
    but if it did they would be provided as HTML files 
    under a ``src/docs`` folder.  Inside that folder
    documentation intended for developers should be in
    a ``devel`` subfolder and documentation for users
    in a ``user`` subfolder.  Specifically, documentation
    for commands should be under ``user/commands`` as
    described :ref:`here <command help>`,
    and documentation for tools under ``user/tools`` as
    described :ref:`here <help>`.  The ``docs`` directory
    also needs to be added to the list of data files in
    **bundle_info.xml**.


*Building and testing the Sample Bundle using ``ChimeraX``*
    #. Create a **license.txt** file.  The easiest way is to copy
       **license.txt.bsd** to **license.txt**.
    #. Start ChimeraX.  In the command line, type ``devel install pathname``
       where *pathname* is the path to the folder containing your
       bundle.  This will build a wheel from your bundle and install
       it as a user bundle, *i.e.*, it will **not** be installed in
       the user-specific folder rather than the ChimeraX folder.
    #. Check that the bundle works by opening a molecule and executing
       the command ``sample count``.  It should report the number of atoms
       and bonds for each molecule in the log.


*Building the Sample Bundle using ``make``*
    #. Edit **Makefile** and change ``CHIMERAX_APP`` to match the location
       of **ChimeraX.app** on your system.
    #. Create a **license.txt** file.  The easiest way is to copy
       **license.txt.bsd** to **license.txt**.
    #. Execute ``make install`` (which simply executes
       ``devel install .`` in ChimeraX).
    #. Check directory **dist** to make sure the wheel was created.
    #. Check that the bundle works by opening a molecule and executing
       the command ``sample count``.  It should report the number of atoms
       and bonds for each molecule in the log.


Customizing the Sample Code
---------------------------

To convert the sample code into your own bundle, there are several
importants steps:

#. First, customize the source code in the **src** folder for
   your bundle.
#. Edit **bundle_info.xml** to update bundle information.
   The supported elements are listed below in `Bundle Information
   XML Tags`_.


Building and Testing Bundles
----------------------------

To build and test your bundle, execute the following command
(or run ``make install`` which invokes the same command):

``$(CHIMERAX_EXE) --nogui --cmd "devel install . ; exit"``
    Execute the ``devel install .`` command in ChimeraX.
    Python source code and other resource files are copied
    into the *build* folder.  C/C++ source files, if any,
    are compiled and also copied into the *build* folder.
    The files in *build* are then assembled into a wheel
    in the *dist* directory.  The assembled wheel is installed
    as a user bundle.

Note that on Windows ``$(CHIMERAX_EXE)`` uses the
``ChimeraX-console.exe`` executable rather than the normal
``ChimeraX.exe``.  This is because on Windows an executable
cannot be both a GUI and a console app.  So for running
ChimeraX with the ``--nogui`` flag, you need to use the
console executable.

If the command completes successfully, fire up ChimeraX
(``make test`` is a shortcut if ``make`` is available)
and try out your command.  Warning and error messages
should appear in the ``Log`` window.
If the bundle is not working as expected, *e.g.*, command is
not found, tool does not start, and no messages are being
displayed, try executing ``$(CHIMERAX_EXE) --debug``
(or ``make debug`` for short), which runs ChimeraX
in debugging mode, and see if more messages are shown in
the console.


Distributing Bundles
--------------------

With ChimeraX bundles being packages as standard Python
wheel-format files, they can be distributed as plain files
and installed using the ChimeraX ``toolshed install``
command.  Thus, electronic mail, web sites and file
sharing services can all be used to distribute ChimeraX
bundles.

Private distributions are most useful during bundle
development, when circulation may be limited to testers.
When bundles are ready for public release, they can be
published on the `ChimeraX Toolshed`_, which is designed
to help developers by eliminating the need for custom
distribution channels, and to aid users by providing
a central repository where bundles with a variety of
functionality may be found.

Customizable information for each bundle on the toolshed
includes its description, screen captures, authors,
citation instructions and license terms.
Automatically maintained information
includes release history and download statistics.

To submit a bundle for publication on the toolshed,
you must first sign in.  Currently, only Google
sign in is supported.  Once signed in, use the
``Submit a Bundle`` link at the top of the page
to initiate submission, and follow the instructions.
The first time a bundle is submitted to the toolshed,
approval from ChimeraX staff is needed before it is
published.  Subsequent submissions, using the same
sign in credentials, do not need approval and should
appear immediately on the site.

.. _`ChimeraX Toolshed`: https://cxtoolshed.rbvi.ucsf.edu


Cleaning Up ChimeraX Bundle Source Folders
------------------------------------------

Two ``make`` targets are provided for removing intermediate
files left over from building bundles:

``make clean``
    Remove generated files, *e.g.*, **setup.py** and **build** folder,
    as well as the **dist** folder containing the built wheels.


Bundle Information XML Tags
---------------------------

ChimeraX bundle information is stored in **bundle_info.xml**.
Details of supported XML tags are found in
:doc:`tutorials/bundle_info`.

Using ``pyproject.toml`` for Bundles
------------------------------------

ChimeraX bundle information can optionally be stored in **pyproject.toml**.
Details of that file are found in
:doc:`tutorials/pyproject`
