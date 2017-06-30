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
    command.

    **io.py** contains code called by ``bundle_api``
    from **__init__.py** for opening XYZ files.

    **tool.py** contains code called by ``bundle_api``
    from **__init__.py** for starting the graphical
    interface.

    **_sample.cpp** contains sample C++ code that
    compiles into a Python module that defines two
    module functions.


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
XML elements in the file typically have *attributes* and either
(a) *child elements*, or (b) *element text*.
An attributes is used for a value that may be represented
as a simple string, such as an identifiers or a version numbers.
The element text is used for a more complex value, such as a
file name which may contain spaces.
Child elements are used for multi-valued data, such as a
list of file names, one element per value.

The supported elements are listed below in alphabetical order.
The root document elements is **BundleInfo**, which contains
all the information needed to build the bundle.

- **Author**

  - Element text:

    - Name of bundle author

- **BundleInfo**:

  - Root element containing all information needed to build the bundle.
  - Attributes:

    - **name**: bundle name (must start with ``ChimeraX-`` for now)
    - **custom_init**: set to ``true`` if bundle has custom initialization
      function; omit otherwise
    - **minSessionVersion**: version number of oldest supported Chimera session
    - **maxSessionVersion**: version number of newest supported Chimera session
    - **package**: Python package name corresponding to bundle
    - **pure_python**: set to ``false`` if bundle should be treated as
      binary, *i.e.*, includes a compiled module; omit otherwise
    - **version**: bundle version

  - Child elements:

    - **Author** (one)
    - **Email** (one)
    - **Categories** (one)
    - **Classifiers** (one)
    - **DataFiles** (zero or more)
    - **CModule** (zero or more)
    - **Dependencies** (zero or more)
    - **Description** (one)
    - **Synopsis** (one)
    - **URL** (one)

- **Categories**

  - List of categories where bundle may appear, *e.g.*, in menus
  - Child elements:

    - **Category** (one or more)

- **Category**

  - Attribute:

    - **name**: name of category (see **Tools** menu in ChimeraX for
      a list of well-known category names)

- **DataFiles**

  - List of data files in package source tree that should be included
    in bundle
  - Attribute:

    - **package**: name of package that has the extra data files.
      If omitted, the current bundle package is used.

  - Child elements:

    - **DataFile** (one or more)

- **DataFile**

  - Element text

    - Data file name (or wildcard pattern) relative to package
      source.  For example, because current package source is expected
      to be in folder **src**, a data file **datafile** in the
      same folder is referenced as ``datafile``, not ``src/datafile``.

- **ChimeraXClassifier**

  - Lines similar to Python classifiers but containing
    ChimeraX-specific information about the bundle.
    See `ChimeraX Metadata Classifiers`_
    below for detailed description of ChimeraX
  - Element text:

    - Lines with ``::``-separated fields.

- **Classifiers**
  
  - Child elements:

    - **ChimeraXClassifier** (zero or more)
    - **PythonClassifier** (zero or more)

- **CModule**

  - List of compiled modules in the current bundle.
  - Attribute:

    - **major**: major version number for compiled module.
    - **minor**: minor version number for compiled module.
    - **name**: name of compiled module.  This should not include
      file suffixes, as they vary across platforms.  The compiled
      module will appear as a submodule of the Python package
      corresponding to the bundle.
    - **platform**: name of platform that builds this module.
      This may be used when a compiled module is only needed
      on a specific platform.  For example, supporting the
      Space Navigator device requires a compiled module on
      macOS, but may be accomplished using ``ctypes`` on other
      platforms.  Supported values for platform are: **mac**,
      **windows**, and **linux**.

  - Child elements:
    
    - **FrameworkDir** (zero or more)
    - **IncludeDir** (zero or more)
    - **Library** (zero or more)
    - **LibraryDir** (zero or more)
    - **Requires** (zero or more)
    - **SourceFile** (one or more)

- **Dependencies**

  - List of all ChimeraX bundles and Python packages that the current
    bundle depends on.
  - Child elements:

    - **Dependency** (one or more)

- **Dependency**

  - Attributes:

    - **name**: name of ChimeraX bundle or Python package that current
      bundle depends on.
    - **version**: version of bundle of package that current bundle
      depends on.

- **Description**

  - Element text:

    - Full description of bundle.  May be multiple lines.

- **Email**

  - Element text:

    - Contact address for bundle maintainer.

- **Framework**

  - Child element of **CModule**, applicable only for macOS.
  - Element text:

    - Name of a macOS framework required to compile the current module.

- **FrameworkDir**

  - Child element of **CModule**.
  - Element text:

    - Name of a directory (folder) containing frameworks required
      to compile the current module.

- **IncludeDir**

  - Child element of **CModule**.
  - Element text:

    - Name of a directory (folder) containing header files required
      to compile the current module.  Standard C/C++ and ChimeraX
      include directories are automatically supplied by the build
      process.

- **Library**

  - Child element of **CModule**.
  - Element text:

    - Name of a link library required to compile the current module.
      Standard C/C++ libraries are automatically supplied by the build
      process.  Additional libraries, such as those included in
      **ChimeraX.app**, must be listed if used in the compiled module.
      For example, to use atomic structure functionality, a **Library**
      directive for ``atomstruct`` should be included.

- **LibraryDir**

  - Child element of **CModule**.
  - Element text:

    - Name of a directory (folder) containing link libraries required
      to compile the current module.  Standard C/C++ and ChimeraX
      library directories are automatically supplied by the build
      process.

- **PythonClassifier**

  - Element text:

    - Standard `Python classifier
      <https://pypi.python.org/pypi?%3Aaction=list_classifiers>`_
      with ``::``-separated fields.

- **Requires**

  - Child element of **CModule**.
  - Element text:

    - Full path name of a system file that must be present in
      order to compile the current module.

- **SourceFile**

  - Child element of **CModule**.
  - Element text:

    - Name of source file in a compiled module.  The path should be
      relative to **bundle_info.xml**.

- **Synopsis**

  - Element text:

    - One line description of bundle (*e.g.*, as tool tip text)

- **URL**

  - Element text:

    - URL containing additional information about bundle


ChimeraX Metadata Classifiers
-----------------------------

ChimeraX gathers metadata from Python-wheel-style classifiers
listed in the bundle.  The only required classifier is
for overall bundle metadata; additional classifiers provide
information about tools (graphical interfaces), commands,
data formats, and selectors.

*Bundle Metadata*

    ``ChimeraX`` :: ``Bundle`` :: *categories* :: *session_versions* :: *api_module_name* :: *supercedes* :: *custom_session_init*

    - *categories* is a comma separated list of category names.
      (Category names are the names that appear under the ``Tools``
      menu.)
      This value is currently unused but are intended for constructing
      "toolboxes" in the future.
    - *session_versions* is a comma-separated two-tuple of
      integers, representing the minimum and maximum session
      versions that this tool can read.
    - *api_module_name* is a string with the name of the module that
      has the bundle_api in it.
    - *supercedes* is an optional comma separated list of names that
      under which the bundle was previously released.
    - *custom_session_init* is a string.  If not set to ``true``, the
      bundle is not imported until actually invoked.  If set to
      ``true``, the ``bundle_api.initialize`` method for the bundle
      is called after the main session has been created.

    For example::

      ChimeraX :: Bundle :: Volume data :: 1,1 ::


*Tool Metadata*

    ``ChimeraX`` :: ``Tool`` :: *tool_name* :: *categories* :: *synopsis*

    - *tool_name* is a string that uniquely identifies the tool.
    - *categories* is a comma separated list of category names under
      which the tool will appear.
    - *synopsis* is a short description of the tool.  It is here for
      uninstalled tools, so that users can get more than just a
      name for deciding whether they want the tool or not.

    For example::

      ChimeraX :: Tool :: Help Viewer :: General :: Show help

    Notes:

    - Tool instances are created via the ``bundle_api.start_tool`` method.
    - Bundles may provide more than one tool.

*Command Metadata*

    ``ChimeraX`` :: ``Command`` :: *name* :: *categories* :: *synopsis*

    - *name* is a string and may have spaces in it.
    - *categories* should be a subset of the bundle's categories. 
    - *synopsis* is a short description of the command.  It is here for
      uninstalled commands, so that users can get more than just a
      name for deciding whether they want the command or not.

    For example::

      ChimeraX :: Command :: exit :: General :: terminate ChimeraX

    Notes:

    - Commands are lazily registered, so the argument specification
      isn't needed until the command is first used.
    - Command registration is done via the
      ``bundle_api.register_command`` method.
    - Bundles may provide more than one command.


*Data Format Metadata*

    ``ChimeraX`` :: ``DataFormat`` :: *format_name* :: *nicknames* :: *category* :: *suffixes* :: *mime_types* :: *url* :: *dangerous* :: *icon* :: *synopsis* :: *encoding*

    - *format_name* is a string.
    - *nicknames* is an optional comma-separated list of strings.
      If no nickname is given, it defaults to the lowercased format_name.
    - *category* is a toolshed category.
    - *suffixes* is an optional comma-separated list of strings with
      leading periods, i.e., ``.pdb``.
    - *mime_types* is an optinal comma-separated list of strings, e.g.,
      chemical/x-pdb.
    - *url* is a string that has a URL that points to the data format's docmentation.
    - *dangerous* is an optional boolean and should be ``true`` if the data
      format is insecure -- defaults to true if a script.
    - *icon* is an optional string containing the filename of the icon --
      it defaults to the default icon for the category.
    - *synopsis* is a short description of the data format.  It is here
      because it needs to be part of the metadata available for
      uninstalled data format, so that users can get more than just a
      name for deciding whether they want the data format or not.
    - *encoding* should be given for text formats and is the file encoding.

    For example::

      ChimeraX :: DataFormat :: PDB :: :: Molecular Structure :: .pdb, .ent :: chemical/x-pdb :: http://www.pdb.org/ :: :: :: Protein DataBank file
      ChimeraX :: DataFormat :: mmCIF :: :: Molecular Structure :: .mmcif, .cif :: chemical/x-mmcif :: http://www.pdb.org/ :: :: :: MacroMolecular CIF

    In addition to describing the format, the bundle should say how if it
    can fetch, open or save data in that format.

        ``ChimeraX`` :: ``Open`` :: *format_name* :: *tag* :: *is_default* :: *extra_keywords*

        ``ChimeraX`` :: ``Save`` :: *format_name* :: *tag* :: *is_default* :: *extra_keywords*

        ``ChimeraX`` :: ``Fetch`` :: *database_name* :: *format_name* :: *prefixes* :: *example_id* :: *is_default*

    - *format_name* is a format previously given in a ChimeraX :: DataFormat
      line.
    - *prefixes* is a comma-separated list of strings associated with the
      (database_name, format_name).
    - *tag* is a string is disambiguate multiple readers or writers.
    - *is_default* is a string.  If set to ``true``, this format is
      the default format for the database.
    - *extra_keywords* is an optional comma-separated list of additional
      keyword arguments.  The keyword can be followed by a colon and a
      ChimeraX argument type without the Arg suffix.  If the argument type
      isn't found in the ``chimerax.commands`` module, the bundle API class is
      searched for it.
    - *database_name* is a string with the name of the databasea to fetch
      the data from.
    - *example_id* is a string with an example identifier.

    For example::
    
      ChimeraX :: Open :: PDB :: PDB ::
      ChimeraX :: Save :: PDB :: PDB ::
      ChimeraX :: Fetch :: PDB :: mmcif :: pdb :: 1a0m ::
      ChimeraX :: Fetch :: PDB :: PDB :: :: 1a0m ::

    Notes:

    - File operations are performed via the ``bundle_api.open_file``,
      ``bundle_api.save_file``, and
      ``bundle_api.fetch_from_database`` methods.
    - The data format metadata is used to generate the macOS
      application property list.
    - Bundles may provide more than one data format.


*Selector Metadata*

    ``ChimeraX`` :: ``Selector`` :: *name* :: *synopsis*

    - *name* is a string and may have spaces in it.
    - *synopsis* is a short description of the selector.  It is here for
      uninstalled selectors, so that users can get more than just a
      name for deciding whether they want the selector or not.

    For example::
    
      ChimeraX :: Selector :: helix :: Helical regions in proteins

    Notes:

    - Bundles may provide more than one selector.
    - Many commands take optional keywords before atom and object
      specifiers.  If a selector name is the same as the optional
      keyword, the command will interpret it as the keyword rather
      than the selector.  The bottom line is "choose your selector
      names carefully."
