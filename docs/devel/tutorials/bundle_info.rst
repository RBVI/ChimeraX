.. vim: set expandtab shiftwidth=4 softtabstop=4:

:orphan:

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

.. _Bundle Information XML Tags:

Bundle Information XML Tags
===========================

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

NB: All elements except **BundleInfo** may have a **platform**
attribute.  If the **platform** attribute *is* present and its
value does *not* matches the build platform, then the element and
all its children are ignored.  Supported values for **platform**
are: ``mac``, ``windows``, and ``linux``.  An example use for the
**platform** attribute is in supporting the Space Navigator device.
On macOS, ChimeraX relies on a compiled C module, while on Windows
and Linux, it uses pure Python with the ``ctypes`` module;
in this case, the **CModule** element has a **platform** attribute
of ``mac``.

- **AdditionalPackages**

  - List of additional packages to include in bundle
    in bundle

  - Child elements:

    - **Package** (one or more)

- **Author**

  - Element text:

    - Name of bundle author

- **BundleInfo**:

  - Root element containing all information needed to build the bundle.
  - Attributes:

    - **name**: bundle name (must start with ``ChimeraX-`` for now)
    - **customInit**: set to ``true`` if bundle has custom initialization
      function; omit otherwise
    - **installedDataDir**: name of directory containing data files,
      relative to the bundle Python package directory
    - **installedIncludeDir**: name of directory containing C/C++ header files,
      relative to the bundle Python package directory
    - **installedLibraryDir**: name of directory containing DLLs (Windows),
      shared objects (Mac OS X), or shared libraries (Linux),
      relative to the bundle Python package directory
    - **installedDataDir**: name of directory containing data files, relative
      to the bundle Python package directory
    - **minSessionVersion**: version number of oldest supported Chimera session
    - **maxSessionVersion**: version number of newest supported Chimera session
    - **package**: Python package name corresponding to bundle
    - **purePython**: set to ``false`` if bundle should be treated as
      binary, *i.e.*, includes a compiled module; omit otherwise
    - **version**: bundle version

  - Child elements:

    - **Author** (one)
    - **Categories** (one)
    - **Classifiers** (one)
    - **CModule** (zero or more)
    - **DataFiles** (zero or more)
    - **Dependencies** (zero or more)
    - **Description** (one)
    - **ExtraFiles** (zero or more)
    - **Email** (one)
    - **ExtraFiles** (zero or more)
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

- **CExecutable**

  - Compiled executable in the current bundle
  - Attribute:

    - **name**: name of executable file

  - Child elements:

    - **FrameworkDir** (zero or more)
    - **IncludeDir** (zero or more)
    - **Library** (zero or more)
    - **LibraryDir** (zero or more)
    - **Requires** (zero or more)
    - **SourceFile** (one or more)

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

- **CLibrary**

  - Compile library or shared object in the current bundle
  - Attribute:

    - **name**: name of library or shared object
    - **usesNumpy**: whether library requires ``numpy`` headers.
      If set to ``true``, ``numpy`` header directories (folders)
      are included on the compilation command.
    - **static**: whether to build a static (``true``) or
      dynamic (``false``) library

  - Child elements:

    - **FrameworkDir** (zero or more)
    - **IncludeDir** (zero or more)
    - **Library** (zero or more)
    - **LibraryDir** (zero or more)
    - **Requires** (zero or more)
    - **SourceFile** (one or more)

- **CModule**

  - List of compiled modules in the current bundle.
  - Attribute:

    - **major**: major version number for compiled module.
    - **minor**: minor version number for compiled module.
    - **name**: name of compiled module.  This should not include
      file suffixes, as they vary across platforms.  The compiled
      module will appear as a submodule of the Python package
      corresponding to the bundle.
    - **usesNumpy**: whether module requires ``numpy`` headers.
      If set to ``true``, ``numpy`` header directories (folders)
      are included on the compilation command.

  - Child elements:
    
    - **FrameworkDir** (zero or more)
    - **IncludeDir** (zero or more)
    - **Library** (zero or more)
    - **LibraryDir** (zero or more)
    - **Requires** (zero or more)
    - **SourceFile** (one or more)

- **DataDir**

  - Element text

    - Data directory name (no wildcard characters) relative to package
      source.  For example, because current package source is expected
      to be in folder **src**, a data directory **datadir** in the
      same folder is referenced as ``datadir``, not ``src/datafile``.
      All files and subdirectories in the specified directory are
      included in the bundle.

- **DataFile**

  - Element text

    - Data file name (or wildcard pattern) relative to package
      source.  For example, because current package source is expected
      to be in folder **src**, a data file **datafile** in the
      same folder is referenced as ``datafile``, not ``src/datafile``.

- **DataFiles**

  - List of data files in package source tree that should be included
    in bundle
  - Attribute:

    - **package**: name of package that has the extra data files.
      If omitted, the current bundle package is used.

  - Child elements:

    - **DataDir** (zero or more)
    - **DataFile** (zero or more)

- **Dependencies**

  - List of all ChimeraX bundles and Python packages that the current
    bundle depends on.  For building bundles containing C/C++ source code,
    *include* and *library* directories of bundles in the dependency lists
    are automatically incorporated in compilation options.  (This implies
    that bundles on the dependency list must alreay be installed.)
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

- **ExtraDir**

  - Extra directory in the bundle that is copied from elsewhere in
    the source tree.
  - Element text

    - Directory name (no wildcard characters) relative to package
      source.  For example, because current package source is expected
      to be in folder **src**, a directory **extradir** in the
      same folder is referenced as ``extradir``, not ``src/extrafile``.
      All files and subdirectories in the specified directory are
      included in the bundle.

  - Attributes:

    - **source**: Directory name relative to bundle source directory.
      The source directory will be copied into the ``src`` directory
      with the directory name given in the element text.

- **ExtraFile**

  - Element text

    - Extra file name (or wildcard pattern) relative to package
      source.  For example, because current package source is expected
      to be in folder **src**, a data file **datafile** in the
      same folder is referenced as ``datafile``, not ``src/datafile``.

  - Attributes:

    - **source**: File name relative to bundle source directory.
      The source file will be copied into the ``src`` directory
      with the file name given in the element text.

- **ExtraFiles**

  - List of extra files in package source tree that should be included
    in bundle.  The extra files, *e.g.*, C++ header files, are copied
    from elsewhere in the source tree into the ``src`` directory for
    inclusion in the bundle.  Files listed under **ExtraFiles** do not
    need to be listed under **DataFiles**.
  - Attribute:

    - **package**: name of package that has the extra data files.
      If omitted, the current bundle package is used.

  - Child elements:

    - **ExtraDir** (zero or more)
    - **ExtraFile** (zero or more)

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
      header directories are automatically supplied by the build
      process.

- **Initializations**

  - List of bundles that must be initialized before this one.
  - Currently, the supported types of initializations are:
    **manager** and **custom**.  Managers across all bundles
    are initialized first; then custom initialization across
    all bundles.
  - Child elements:

    - **InitAfter** (one or more)

- **InitAfter**

  - Attribute:

    - **type**: type of initialization.  Currently supported
      values are **manager** and **custom**.
    - **bundle**: name of bundle that must be initialized before
      this one.
    - There should be one **InitAfter** tag for each bundle that
      must be initialized first.  There is no way to specify
      the exact initialization order for these bundles; the
      relative dependencies will be computed from the initialization
      information of the bundles.

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

- **Managers**

  - List of managers that bundle provides
  - Child elements:

    - **Manager** (one or more)

- **Manager**

  - Attribute:

    - **name**: name of manager.  The bundle must implement the
      ``init_manager`` method.  The two positional arguments to
      ``init_manager`` are the session instance and the manager name.
    - **uiOnly**: set to ``true`` if manager should only be created
      when the graphical user interface is being used; omit otherwise
    - Other attributes listed in the **Manager** tag are passed
      as keyword arguments to ``init_manager``.
    - ``init_manager`` should create and return an instance of a
      subclass of :py:class:`chimerax.core.toolshed.ProviderManager`.
      The subclass must implement at least one method:
        ``add_provider(bundle_info, provider_name, **kw)``
      which is called once for each **Provider** tag whose manager
      name matches this manager (whether the bundle with the provider
      is installed or not).  A second method:
        ``end_providers()``
      is optional.  ``end_providers`` is called after all calls
      to ``add_provider`` have been made and is useful for finishing
      manager initialization.

- **Package**

  - Attributes:

    - **name**: name of Python package to be added.
    - **folder**: folder containing source files in package.

- **Providers**

  - List of providers that bundle provides
  - Attribute:

    - **manager**: optional default manager for nested **Provider** elements

  - Child elements:

    - **Provider** (one or more)

- **Provider**

  - Attribute:

    - **manager**: name of the manager with which this provider
      will be registered.  Optional if **manager** is given in
      parent **Providers** element.
    - **name**: name of provider.
    - Other attributes listed in the **Provider** tag are passed
      as keyword arguments to the manager's ``add_provider`` method.
    - Bundles that supply providers should implement the method:
        ``run_provider(session, provider_name, manager, **kw)``
      which may be used by the manager to invoke provider functionality.

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

  - Child element of **CExecutable**, **CLibrary**, or **CModule**.
  - Element text:

    - Name of source file in a compiled module.  The path should be
      relative to **bundle_info.xml**.

- **Synopsis**

  - Element text:

    - One line description of bundle (*e.g.*, as tool tip text)

- **URL**

  - Element text:

    - URL containing additional information about bundle

.. _ChimeraX Metadata Classifiers:

ChimeraX Metadata Classifiers
-----------------------------

ChimeraX gathers metadata from Python-wheel-style classifiers
listed in the bundle.  The only required classifier is
for overall bundle metadata; additional classifiers provide
information about tools (graphical interfaces), commands,
data formats, and selectors.

*Bundle Metadata*

    ``Bundle`` :: *categories* :: *session_versions* :: *api_module_name* :: *supercedes* :: *custom_init*

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
    - *custom_init* is a string.  If not set to ``true``, the
      bundle is not imported until actually invoked.  If set to
      ``true``, the ``bundle_api.initialize`` method for the bundle
      is called after the main session has been created.

    For example::

      Bundle :: Volume data :: 1,1 ::

    This classifier is automatically generated when using the ``devel``
    command and **bundle_info.xml**.


*Tool Metadata*

    ``Tool`` :: *tool_name* :: *categories* :: *synopsis*

    - *tool_name* is a string that uniquely identifies the tool.
    - *categories* is a comma separated list of category names under
      which the tool will appear.
    - *synopsis* is a short description of the tool.  It is here for
      uninstalled tools, so that users can get more than just a
      name for deciding whether they want the tool or not.

    For example::

      Tool :: Help Viewer :: General :: Show help

    Notes:

    - Tool instances are created via the ``bundle_api.start_tool`` method.
    - Bundles may provide more than one tool.

*Command Metadata*

    ``Command`` :: *name* :: *categories* :: *synopsis*

    - *name* is a string and may have spaces in it.
    - *categories* should be a subset of the bundle's categories. 
    - *synopsis* is a short description of the command.  It is here for
      uninstalled commands, so that users can get more than just a
      name for deciding whether they want the command or not.

    For example::

      Command :: exit :: General :: terminate ChimeraX

    Notes:

    - Commands are lazily registered, so the argument specification
      isn't needed until the command is first used.
    - Command registration is done via the
      ``bundle_api.register_command`` method.
    - Bundles may provide more than one command.


*Data Format Metadata*

    ``DataFormat`` :: *format_name* :: *nicknames* :: *category* :: *suffixes* :: *mime_types* :: *url* :: *dangerous* :: *icon* :: *synopsis* :: *encoding*

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

      DataFormat :: PDB :: :: Molecular Structure :: .pdb, .ent :: chemical/x-pdb :: http://www.pdb.org/ :: :: :: Protein DataBank file
      DataFormat :: mmCIF :: :: Molecular Structure :: .mmcif, .cif :: chemical/x-mmcif :: http://www.pdb.org/ :: :: :: MacroMolecular CIF

    In addition to describing the format, the bundle should say how if it
    can fetch, open or save data in that format.

        ``Open`` :: *format_name* :: *tag* :: *is_default* :: *extra_keywords*

        ``Save`` :: *format_name* :: *tag* :: *is_default* :: *extra_keywords*

        ``Fetch`` :: *database_name* :: *format_name* :: *prefixes* :: *example_id* :: *is_default*

    - *format_name* is a format previously given in a DataFormat line.
    - *prefixes* is a comma-separated list of strings associated with the
      (database_name, format_name).
    - *tag* is a string is disambiguate multiple readers or writers.
    - *is_default* is a string.  If set to ``true``, this format is
      the default format for the database.
    - *extra_keywords* is an optional comma-separated list of additional
      keyword arguments.  The keyword can be followed by a colon and a
      ChimeraX argument type without the Arg suffix.  If the argument type
      isn't found in the :py:class:`chimerax.core.commands` module, the bundle API class is
      searched for it.
    - *database_name* is a string with the name of the databasea to fetch
      the data from.
    - *example_id* is a string with an example identifier.

    For example::
    
      Open :: PDB :: PDB ::
      Save :: PDB :: PDB ::
      Fetch :: PDB :: mmcif :: pdb :: 1a0m ::
      Fetch :: PDB :: PDB :: :: 1a0m ::

    Notes:

    - File operations are performed via the ``bundle_api.open_file``,
      ``bundle_api.save_file``, and
      ``bundle_api.fetch_from_database`` methods.
    - The data format metadata is used to generate the macOS
      application property list.
    - Bundles may provide more than one data format.


*Selector Metadata*

    ``Selector`` :: *name* :: *synopsis*

    - *name* is a string and may have spaces in it.
    - *synopsis* is a short description of the selector.  It is here for
      uninstalled selectors, so that users can get more than just a
      name for deciding whether they want the selector or not.

    For example::
    
      Selector :: helix :: Helical regions in proteins

    Notes:

    - Bundles may provide more than one selector.
    - Many commands take optional keywords before atom and object
      specifiers.  If a selector name is the same as the optional
      keyword, the command will interpret it as the keyword rather
      than the selector.  The bottom line is "choose your selector
      names carefully."


*Manager Metadata*

    ``Manager`` :: *name* [:: *keyword:value*]*

    - *name* is a string and may have spaces in it.
    - *keyword:value* pairs are zero-or more manager-specific options,
      separated by ``::``.

    For example::
    
      Manager :: http_scheme :: guiOnly:true

    Notes:

    - Bundles may provide more than one manager.
    - The toolshed reserved the following keywords:
      - **guiOnly**, if present and set to ``true``, means the manager
        should only be created if the graphical user interface is in use.
