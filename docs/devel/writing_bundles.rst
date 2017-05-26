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

Writing and Distributing Bundles
================================

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

Sample Bundle
-------------

To build a bundle from the `sample code
<https://www.cgl.ucsf.edu/chimerax/cgi-bin/bundle_sample.zip>`_,
you will need access the the ``make`` program.  On Linux
and macOS, ``make`` is available as part of the
developer package.  On Windows, ``make`` is
available as part of `Cygwin <https://cygwin.com>`_.

**Bundle Sample Code**

The sample code is organized with "administrative" code
at the top level and actual bundle code in the ``src``
folder.  Administrative code, with the exception of
license text, is only used for building the bundle.
All other contents of the bundle should be in ``src``.


    *Administrative Files*

    **README** contains terse instructions on how to
    build the sample code.  (Because the sample code
    includes source code that need to be compiled,
    you will need a C++ compiler for the build.
    If your bundle does not contain C++ code,
    the compiler is not needed, as discussed in
    the *Pure Python Bundles* section below.)

    **Makefile** is the configuration file used by
    the ``make`` command.  This file will need to
    be modified for your bundle.

    **license.txt.bsd** and **license.txt.mit** are
    two sample license text files.  The actual file
    used when building the bundle is **license.txt**
    which must exist.  For testing, simply renaming
    one of the sample license text file is sufficient.
    You may want to use a custom license for your
    actual bundle.

    **setup.py.in** contains Python code for building
    the bundle.  It is preprocessed during the ``make``
    command to create **setup.py**, which is then
    executed using the Python interpreter that comes
    as part of ChimeraX.  This file will need to be
    modified for your bundle.

    **setup.cfg** is the configuration file used when
    **setup.py** is run.  This file should not be modified.

    **wheel_tag.py** is a Python script used while
    building the bundle to determine part of the output
    file name.   This file should not be modified.


    *Bundle Source Code Files*

    **__init__.py** contains the bundle initialization
    code.  Typically, it defines a subclass of the
    ``chimerax.core.toolshed.BundleAPI`` class and
    instantiates a single instance named ``bundle_api``.
    ChimeraX communicates with the bundle through this
    singleton, which must conform to the `bundle API`.

    **cmd.py** contains code called by ``bundle_api``
    from **__init__.py**.

    **_sample.cpp** contains sample C++ code that
    compiles into a Python module that defines two
    module functions.

    .. _`Building the Sample Bundle`:

    *Building the Sample Bundle*

    #. Edit **Makefile** and change ``CHIMERAX_APP`` to match the location
       of **ChimeraX.app** on your system.
    #. Create a **license.txt** file.  The easiest way is to copy
       **license.txt.bsd** to **license.txt**.
    #. Execute ``make``.  See `Transcript for building sample code`_ below.
    #. Check directory **dist** to make sure the wheel was created.


    *Verifying Bundle Works*

    #. Execute ``make app-install`` to install the wheel into your copy
       of **ChimeraX.app** (assuming you have write permission).
       See `Transcript for installing sample code`_ below.
    #. Check that the bundle works by opening a molecule and executing
       the command ``sample count``.  It should report the number of atoms
       and bonds for each molecule in the log.


Customizing the Sample Code
---------------------------

To convert the sample code into your own bundle, there are several
importants steps:

#. Set the bundle name and version number by editing **Makefile**
   and changing the ``BUNDLE_NAME`` and ``BUNDLE_VERSION`` variables.
   ``BUNDLE_NAME`` **must** start with the string ``ChimeraX_``
   followed by the name of your bundle.  The "official" name for your
   bundle includes the ``ChimeraX_`` prefix but the displayed name for
   your bundle will not include the prefix.  Version numbers should be
   either two or three integers separated by dots.  The numbers are
   referred to as major, minor, and (if present) micro version numbers.
#. Set the Python package name from where your code will be imported
   by editing **Makefile** and changing the ``PKG_NAME`` variable.
   Similar to ``BUNDLE_name``, ``PKG_NAME`` **must** start with
   ``chimerax.``.
#. Select the type of bundle (pure Python or platform-specific) by
   by editing **Makefile** and changing the ``TAG`` variable.
   If your bundle does not require compiled code, *e.g.*, Python
   modules written in C/C++, include the ``-p`` flag on the ``TAG``
   line.  For example, a pure Python bundle should use::

     TAG = $(shell $(PYTHON_EXE) wheel_tag.py -p)

   while a platform-specific one should use::

     TAG = $(shell $(PYTHON_EXE) wheel_tag.py)

#. Define the code and resources that should be included in the
   bundle by editing **setup.py.in**.
   
  -  Configure whether there is a compiled extension module
     in your bundle.  The line::

      ext_sources = ["src/_sample.cpp"] 

     defines that an extension module will be built from a
     single C/C++ source file: **src/_sample.cpp**.
     If your bundle is pure Python, change the assignment to::

      ext_sources = []

     If your bundle is platform-specific, set ``ext_sources``
     to your list of C/C++ source files *to be compiled*, *i.e.*,
     no header or include files.  You should also set the
     name of the compiled extension module.  The statement
     that creates the extension module is::

      ext_mods = [Extension("PKG_NAME._sample", ...

     which names the extension module as ``_sample`` within
     your bundle.  By ChimeraX convention, a compiled
     Python module's name starts with an underscore.
     The remainder of the name is up to you.
  -  You do not need to list the Python files to be included
     in the bundle.  By default, all ``.py`` files in **src**
     will be part of the bundle.
  -  If you have other resource files that need to be part
     of the bundle, you need to review
     https://packaging.python.org/distributing/#data-files
     to see what additional arguments needs to be passed to
     ``setup()``.
  -  Various "standard" ``setup()`` argument values need to
     be updated to match your bundle, *e.g.*, ``description``,
     ``author``, ``author_email``, ``url``.
  -  If your bundle depends on another ChimeraX bundle (other
     than the core), you need to list the dependency in
     ``install_requires``.
  -  Finally, you need to update the ``classifiers`` list
     which contains metadata describing the bundle/wheel.
     Two general classifiers that should be checked for
     correctness are ``Development Status`` and ``License``.
     In addition, there are a number of ChimeraX-specific
     classifiers that must be correctly set in order for
     ChimeraX to make proper use of your bundle (see next
     section).


**ChimeraX Metadata and Python Wheel Classifiers**

ChimeraX gathers metadata from Python wheel classifiers
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
    - *database_name* is a string with the name of the databasea to fetch
      the data from.
    - *prefixes* is a comma-separated list of strings associated with the
      (database_name, format_name).
    - *example_id* is a string with an example identifier.
    - *tag* is a string is disambiguate multiple readers or writers.
    - *is_default* is a string.  If set to ``true``, this format is
      the default format for the database.
    - *extra_keywords* is an optional comma-separated list of additional
      keyword arguments.  The keyword can be followed by a colon and a
      ChimeraX argument type without the Arg suffix.  If the argument type
      isn't found in the ``chimerax.commands`` module, the bundle API class is
      searched for it.

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


Testing Bundles
---------------

To test your bundle, you need to first build it in a similar
manner as `Building the Sample Bundle`_.

Distributing Bundles
--------------------

**Toolshed Submission**


Output Samples
--------------

.. _`Transcript for building sample code`:

**Transcript for building sample code**

::

    sed -e 's,BUNDLE_NAME,ChimeraX_Sample,' \
            -e 's,BUNDLE_VERSION,0.1,' \
            -e 's,PKG_NAME,chimerax.sample,' \
            < setup.py.in > setup.py
    /e/chimerax/ChimeraX.app/bin/python.exe setup.py --no-user-cfg build
    running build
    running build_py
    creating build
    creating build\lib.win-amd64-3.6
    creating build\lib.win-amd64-3.6\chimerax
    creating build\lib.win-amd64-3.6\chimerax\sample
    copying src\cmd.py -> build\lib.win-amd64-3.6\chimerax\sample
    copying src\__init__.py -> build\lib.win-amd64-3.6\chimerax\sample
    running build_ext
    building 'chimerax.sample._sample' extension
    creating build\temp.win-amd64-3.6
    creating build\temp.win-amd64-3.6\Release
    creating build\temp.win-amd64-3.6\Release\src
    C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\BIN\x86_amd64\cl.exe /c /nologo /Ox /W3 /GL /DNDEBUG /MD -DMAJOR_VERSION=0 -DMINOR_VERSION=1 -IE:\chimerax\ChimeraX.app\include -IE:\chimerax\ChimeraX.app\bin\include -IE:\chimerax\ChimeraX.app\bin\include "-IC:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\INCLUDE" "-IC:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\INCLUDE" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\NETFXSDK\4.6.1\include\um" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\shared" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\um" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\winrt" "-Ic:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include" "-Ic:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\atlmfc\include" "-Ic:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\ucrt" "-Ic:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\shared" "-Ic:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\um" "-Ic:\Program Files (x86)\Windows Kits\10\include\10.0.10586.0\winrt" /EHsc /Tpsrc/_sample.cpp /Fobuild\temp.win-amd64-3.6\Release\src/_sample.obj
    _sample.cpp
    [... Compiler warning messages not shown ...]
    C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\BIN\x86_amd64\link.exe /nologo /INCREMENTAL:NO /LTCG /DLL /MANIFEST:EMBED,ID=2 /MANIFESTUAC:NO /LIBPATH:E:\chimerax\ChimeraX.app\lib /LIBPATH:E:\chimerax\ChimeraX.app\bin\libs /LIBPATH:E:\chimerax\ChimeraX.app\bin\PCbuild\amd64 "/LIBPATH:C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\LIB\amd64" "/LIBPATH:C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\ATLMFC\LIB\amd64" "/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.10586.0\ucrt\x64" "/LIBPATH:C:\Program Files (x86)\Windows Kits\NETFXSDK\4.6.1\lib\um\x64" "/LIBPATH:C:\Program Files (x86)\Windows Kits\10\lib\10.0.10586.0\um\x64" "/LIBPATH:c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\lib\amd64" "/LIBPATH:c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\atlmfc\lib\amd64" "/LIBPATH:c:\Program Files (x86)\Windows Kits\10\lib\10.0.10586.0\ucrt\x64" "/LIBPATH:c:\Program Files (x86)\Windows Kits\10\lib\10.0.10586.0\um\x64" libatomstruct.lib /EXPORT:PyInit__sample build\temp.win-amd64-3.6\Release\src/_sample.obj /OUT:build\lib.win-amd64-3.6\chimerax\sample\_sample.cp36-win_amd64.pyd /IMPLIB:build\temp.win-amd64-3.6\Release\src\_sample.cp36-win_amd64.lib
    [... Linker warning messages not shown ...]
       Creating library build\temp.win-amd64-3.6\Release\src\_sample.cp36-win_amd64.lib and object build\temp.win-amd64-3.6\Release\src\_sample.cp36-win_amd64.exp
    Generating code
    Finished generating code
    /e/chimerax/ChimeraX.app/bin/python.exe setup.py --no-user-cfg test
    running test
    running egg_info
    creating ChimeraX_Sample.egg-info
    writing ChimeraX_Sample.egg-info\PKG-INFO
    writing dependency_links to ChimeraX_Sample.egg-info\dependency_links.txt
    writing requirements to ChimeraX_Sample.egg-info\requires.txt
    writing top-level names to ChimeraX_Sample.egg-info\top_level.txt
    writing manifest file 'ChimeraX_Sample.egg-info\SOURCES.txt'
    reading manifest file 'ChimeraX_Sample.egg-info\SOURCES.txt'
    writing manifest file 'ChimeraX_Sample.egg-info\SOURCES.txt'
    running build_ext
    copying build\lib.win-amd64-3.6\chimerax\sample\_sample.cp36-win_amd64.pyd -> src
    
    ----------------------------------------------------------------------
    Ran 0 tests in 0.000s
    
    OK
    /e/chimerax/ChimeraX.app/bin/python.exe setup.py --no-user-cfg bdist_wheel
    running bdist_wheel
    running build
    running build_py
    running build_ext
    installing to build\bdist.win-amd64\wheel
    running install
    running install_lib
    creating build\bdist.win-amd64
    creating build\bdist.win-amd64\wheel
    creating build\bdist.win-amd64\wheel\chimerax
    creating build\bdist.win-amd64\wheel\chimerax\sample
    copying build\lib.win-amd64-3.6\chimerax\sample\cmd.py -> build\bdist.win-amd64\wheel\.\chimerax\sample
    copying build\lib.win-amd64-3.6\chimerax\sample\_sample.cp36-win_amd64.pyd -> build\bdist.win-amd64\wheel\.\chimerax\sample
    copying build\lib.win-amd64-3.6\chimerax\sample\__init__.py -> build\bdist.win-amd64\wheel\.\chimerax\sample
    running install_egg_info
    running egg_info
    writing ChimeraX_Sample.egg-info\PKG-INFO
    writing dependency_links to ChimeraX_Sample.egg-info\dependency_links.txt
    writing requirements to ChimeraX_Sample.egg-info\requires.txt
    writing top-level names to ChimeraX_Sample.egg-info\top_level.txt
    reading manifest file 'ChimeraX_Sample.egg-info\SOURCES.txt'
    writing manifest file 'ChimeraX_Sample.egg-info\SOURCES.txt'
    Copying ChimeraX_Sample.egg-info to build\bdist.win-amd64\wheel\.\ChimeraX_Sample-0.1-py3.6.egg-info
    running install_scripts
    creating build\bdist.win-amd64\wheel\ChimeraX_Sample-0.1.dist-info\WHEEL
    E:\chimerax\ChimeraX.app\bin\lib\site-packages\wheel\pep425tags.py:77: RuntimeWarning: Config variable 'Py_DEBUG' is unset, Python ABI tag may be incorrect
      warn=(impl == 'cp')):
    E:\chimerax\ChimeraX.app\bin\lib\site-packages\wheel\pep425tags.py:81: RuntimeWarning: Config variable 'WITH_PYMALLOC' is unset, Python ABI tag may be incorrect
      warn=(impl == 'cp')):
    rm -rf ChimeraX_Sample.egg-info
    echo Distribution is in dist/ChimeraX_Sample-0.1-cp36-cp36m-win_amd64.whl
    Distribution is in dist/ChimeraX_Sample-0.1-cp36-cp36m-win_amd64.whl


.. _`Transcript for installing sample code`:

**Transcript for installing sample code**

::

    [... Output from building the bundle ...]
    /e/chimerax/ChimeraX.app/bin/ChimeraX.exe --nogui --cmd "toolshed install dist/ChimeraX_Sample-0.1-cp36-cp36m-win_amd64.whl reinstall true ; exit"
    0.00% done: Initializing core
    50.00% done: Initializing bundles
    INFO:
    Executing: toolshed install dist/ChimeraX_Sample-0.1-cp36-cp36m-win_amd64.whl reinstall true 
    INFO:
    Installed ChimeraX-Sample (0.1)
    INFO:
    Executing: exit
    STATUS:
    Exiting ...
