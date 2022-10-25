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

.. role:: raw-html(raw)
    :format: html

.. _Bundle Information XML Tags:

Bundle Information XML Tags
===========================

ChimeraX bundle information is stored in **bundle_info.xml**.
XML elements in the file typically have *attributes* and either
(a) *child elements*, or (b) *element text*.
An attribute is used for a value that may be represented
as a simple string, such as an identifiers or a version numbers.
The element text is used for a more complex value, such as a
file name which may contain spaces.
Child elements are used for multi-valued data, such as a
list of file names, one element per value.

The supported elements are listed below in alphabetical order.
The root document element is **BundleInfo**, which contains
all the information needed to build the bundle.

NB: All elements except **BundleInfo** may have a **platform**
attribute.  If the **platform** attribute *is* present and its
value does *not* matches the build platform, then the element and
all its children are ignored.  Supported values for **platform**
are: ``mac``, ``windows``, and ``linux``.  An example use for the
**platform** attribute is to support the Space Navigator device.
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

.. _BundleInfo:

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
    - **limitedAPI**: set to Python stable ABI version; omit otherwise.
      Typicaly used by binary bundles to declare that they can work with older versions of Python
      via Python's `Stable Application Binary Interface <https://docs.python.org/3/c-api/stable.html>`_
      but can also be used by pure-Python bundles that are using very new language features
      to declare that they *can't* work with older Python versions.
      In either case, the value is the oldest version that the bundle works with,
      and is of the form "3.x" (e.g. 3.7).
    - **minSessionVersion**: for session data saved from this bundle, the oldest version that the
      bundle currently supports (an integer).  
    - **maxSessionVersion**: the newest version of this bundle's session data.  Presumably the bundle
      currently writes this version.  The version number should only be increased if the change is not
      backwards compatible with old readers, because the session-restore code checks these version numbers
      in order to decide if a session will be able to be restored by the currently installed bundles.
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

    - **CompileArgument** (zero or more)
    - **Define** (zero or more)
    - **FrameworkDir** (zero or more)
    - **IncludeDir** (zero or more)
    - **Library** (zero or more)
    - **LibraryDir** (zero or more)
    - **LinkArgument** (zero or more)
    - **Requires** (zero or more)
    - **SourceFile** (one or more)
    - **Undefine** (zero or more)

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

    - **CompileArgument** (zero or more)
    - **Define** (zero or more)
    - **FrameworkDir** (zero or more)
    - **IncludeDir** (zero or more)
    - **Library** (zero or more)
    - **LibraryDir** (zero or more)
    - **LinkArgument** (zero or more)
    - **Requires** (zero or more)
    - **SourceFile** (one or more)
    - **Undefine** (zero or more)

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
    
    - **CompileArgument** (zero or more)
    - **Define** (zero or more)
    - **FrameworkDir** (zero or more)
    - **IncludeDir** (zero or more)
    - **Library** (zero or more)
    - **LibraryDir** (zero or more)
    - **LinkArgument** (zero or more)
    - **Requires** (zero or more)
    - **SourceFile** (one or more)
    - **Undefine** (zero or more)

- **CompileArgument**

  - Element text

    - Additional argument to provide to the compiler when compiling.

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

- **Define**

  - Element text

    - Symbolic name to be defined during compilation.  Can just be
      the symbolic name itself, or the symbolic name plus *=value*, as
      needed.

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

- **LinkArgument**

  - Element text

    - Additional argument to provide to the linker when linking.

- **Managers**

  - List of managers that bundle provides
  - Child elements:

    - **Manager** (one or more)

.. _Manager:

- **Manager**

  - Attribute:

    - **name**: name of manager.  If **autostart** is true (see below), the bundle
      must implement the ``init_manager`` method.  The two positional arguments to
      ``init_manager`` are the session instance and the manager name.
    - **guiOnly**: set to ``true`` if manager should only be created
      when the graphical user interface is being used; omit otherwise
    - **autostart**: If true, the manager is started during Chimera startup.
      Defaults to true.
    - Other attributes listed in the **Manager** tag are passed
      as keyword arguments to ``init_manager``.
    - ``init_manager`` should create an instance of a
      subclass of :py:class:`chimerax.core.toolshed.ProviderManager`.
      The ProviderManager constructor must be passed the **name** of the manager.
      The subclass must implement at least one method:
      ``add_provider(bundle_info, provider_name, **kw)``
      which is called once for each **Provider** tag whose manager
      name matches this manager (whether the bundle with the provider
      is installed or not).  To distinguish between installed and uninstalled
      providers check ``bundle_info.installed``.
      A second method: ``end_providers()`` is optional.
      ``end_providers`` is called after all calls to ``add_provider`` have been made
      and is useful for finishing manager initialization.

- **Package**

  - Attributes:

    - **name**: name of Python package to be added.
    - **folder**: folder containing source files in package.

.. _Providers:

- **Providers**

  - List of providers that bundle provides
  - Attribute:

    - **manager**: optional default manager for nested **Provider** elements

  - Child elements:

    - **Provider** (one or more)

.. _Provider:

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

- **Undefine**

  - Element text

    - Symbolic name to be explictly undefined during compilation.

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
    - Before deciding on your command name and syntax, you should peruse the
      :doc:`command style guide <../command_style>`.


*Data Format Metadata*
    The old ``DataFormat``, ``Open``, and ``Save`` tags have been replaced with
    a manager/provider mechanism, as described in the `Opening/Saving/Fetching Files`_
    section below.


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


.. _Opening/Saving/Fetching Files:

Opening/Saving/Fetching Files
-----------------------------

For a bundle to hook into the ``open`` or ``save`` commands
it must have a `Providers`_ section in its **bundle_info.xml**
to provide the relevant information to the "open command" or
"save command" manager via `Provider`_ tags.
The bundle also typically defines the file/data format via a
`Provider`_ tag for the "data formats" manager, though in
some cases the data format is defined in another bundle.

As per normal XML, `Provider`_ attributes are strings
(*e.g.* ``name="Chimera BILD object"``)
and for attributes that can accept multiple values, those
values are comma separated
(*e.g.* ``suffixes=".bld,.bild"``).

.. _data format:

Defining a File/Data Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To define a data(/file) format, you supply a `Provider`_ tag in the
`Providers`_ section of your **bundle_info.xml** file.  The value of
the ``manager`` of the tag or section should be "data formats".  The
information supplied by the `Provider`_ tag will be all that is
required for the format definition -- *i.e.* the data-formats manager
will never call the :py:class:`~chimerax.core.toolshed.BundleAPI`'s
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider`
method, so that method does not need to be customized
for this manager.

These are the possible `Provider`_ attributes:

- **Mandatory** Attributes

    .. _name:

    *name*
        The full official name of the format, typically omitting the word "format"
        though, since all such names are formats.  The *name* attribute must be
        unique across all format definitions.

- **Frequently-Used** Attributes

    *category*
        The general kind of information that the format provides, used to organize
        formats in some interfaces.  Commonly used categories are: Generic 3D objects,
        Molecular structure, Molecular trajectory, Volume data, Image, Higher-order
        structure, Sequence, and Command script.  The default is the catchall category
        "General".

    *encoding*
        If the format is textual, the encoding for that text.  Binary formats should
        omit this attribute.  The most common encoding for text formats is "utf-8".

    .. _nicknames:

    *nicknames*
        A short, easy-to-type name for the format, typically used in conjunction with
        the ``format`` keyword of the ``open``/``save`` commands.  Still needs to be verbose
        enough to not easily conflict with nicknames of other formats.  Also typically
        all lower case.  Default is an all-lower-case version of *name*.

    *reference_url*
        If there is a web page describing the format, the URL to that page.

    *suffixes*
        The file-name suffixes (starting with a '.') that are used by files in this
        format.  If no suffixes are specified, then files in this format will only be
        able to be opened/saved by supplying the ``format`` keyword to the ``open``/``save``
        commands.  Also, formats that can only be fetched from the web frequently don't
        specify suffixes.

    *synopsis*
        The description of the format used by user-interface widgets that list formats
        (*e.g.* the Open-File dialog), so typically shorter than *name* but more verbose 
        than the *nicknames*.  The first word should be capitalized unless that word is
        mixed case (*e.g.* mmCIF).  Like *name*, *synopsis* should typically omit the
        word "format".  Defaults to *name*.

- **Infrequently-Used** Attributes

    *allow_directory*
        If this is specified as "true", then the data for this format can be organized as
        a folder rather than a single file.  Regardless of the value of *suffixes*, such a
        folder can only be opened/saved by providing the ``format`` keyword to the corresponding
        command.  Specifying *allow_directory* as "true" does not preclude also possibly
        opening this format from individual files (in which case *suffixes* would matter).
        The default is "false".

    *insecure*
        If opening this format's data could cause arbitrary code to execute, then *insecure*
        should be specified as "true".  Formats in the "Command script" *category* default
        to "true" and others to "false".

    *mime_types*
        If the data for this format may be obtained by the user providing an URL to the
        ``open`` command, and the URL might not end in one of the *suffixes* (*e.g.* it's
        a CGI script), but the web server does provide a format-specific Content-Type header
        for the data, then mime_types lists Content-Type header values that the server
        or servers could possibly provide.  Only relevant to the user providing an URL, not
        to the "fetching" of database identifiers outlined in the `Fetching Files`_ section.
        If the data format has a `Wikipedia <https://en.wikipedia.org>`_ page, the "mime type"
        will frequently be specified there (as "Internet media type").

For example::

    <Providers manager="data formats">
        <Provider name="Sybyl Mol2" suffixes=".mol2" nicknames="mol2"
            category="Molecular structure" synopsis="Mol2" encoding="utf-8" />
    </Providers>
  
A detailed example of defining a data format can be found in :ref:`Bundle Example: Read a New File Format`.

.. _open command:

Opening Files
^^^^^^^^^^^^^

For your bundle to open a file, it needs to provide information to the "open command" manager
about what data format it can open, what arguments it needs, what function to call, *etc.*.
Some of that info is provided as attributes in the `Provider`_ tag, but the lion's share is
provided when the open-command manager calls your bundle's
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider` method.
That call will only occur when ChimeraX tries to open the kind of data that your `Provider`_
tag says you can open.

To specify that your bundle can open a data format, you supply a `Provider`_ tag in the
`Providers`_ section of your **bundle_info.xml** file.  The value of
the ``manager`` attribute in the tag or section should be "open command".
The other possible `Provider`_ attributes are:

- **Mandatory** Attributes

    *name*
        The `name`_ of the `data format`_ you can open.  Can also be one of the format's
        `nicknames`_ instead.

- **Infrequently-Used** Attributes

    *batch*
        If your provider can open multiple files of its format as one combined model, then
        it should specify *batch* as "true" and it will be called with a list of path names
        instead of an open file stream.

    *check_path*
        If the user can type something other than an existing file name, and your provider
        will expand that into a real file name or names (*e.g.* there is some kind of substitution
        the provider does with the text), then specify *check_path* as "false" (which implies
        *want_path*\="true", you don't have to explicitly specify that).

    *is_default*
        **Deprecated.**  Will be replaced by the scheme described `here <https://www.rbvi.ucsf.edu/trac/ChimeraX/ticket/7813#comment:2>`_.
        :raw-html:`<font color="gray">` If your data format has suffixes that are the same as another format's suffixes, *is_default*
        will determine which format will be used when the open command's ``format`` keyword is omitted.
        *is_default* defaults to "true", so therefore typically lesser known/used formats supply this
        attribute with a value of "false". :raw-html:`</font>`

    *pregrouped_structures*
        If a provider returns multiple models, the open command will automatically group them
        so that the entire set of models can be referenced with one model number (the individual
        models can be referenced with submodel numbers).  The provider *could* pre-group them in
        order to give the group a name other the default (which is based on the file name; the user can
        still override that with the ``name`` keyword of the open command).  In the specific case
        where the provider is pre-grouping atomic structures, it should specify *pregrouped_structures*
        as "true" so the the open command's return value can be the actual list of structures rather
        than a grouping model.  This greatly simplifies scripts trying to handle return values
        from various kinds of structure-opening commands.

    *type*
        If you are providing information about opening a file rather than fetching from a
        database, *type* should be "open", and otherwise "fetch".  Since the default value
        for *type* is "open", providers that open files typically skip specifying *type*.

    *want_path*
        The provider is normally called with an open file stream rather than a file name,
        which allows ChimeraX to handle compressed files automatically for you.  If your
        file reader must be able to open/read the file itself instead, then specify *want_path*
        as "true" and you will receive a file path instead of a stream, and attempting
        to open a compressed version of your file type will result in an error before your
        provider is even called.
  
For example::

  <Providers manager="open command">
    <Provider name="AutoDock PDBQT" want_path="true" />
    <Provider name="Sybyl Mol2" want_path="true" />
  </Providers>

The remainder of the information the bundle provides about how to open a file comes from the
return value of the bundle's
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider` method, which must return
an instance of the
:py:class:`chimerax.open_command.OpenerInfo` class.
The doc strings of that class discuss its methods in detail, but briefly:

* You must override the :py:meth:`~chimerax.open_command.OpenerInfo.open` method to take
  the input provided and return a (models, status message) tuple.

* If your format has format-specific keywords that the ``open`` command should accept,
  you must override the :py:meth:`~chimerax.open_command.OpenerInfo.open_args` property
  to return a dictionary that maps **Python** keywords of your opener-function to corresponding
  :ref:`Annotation <Type Annotations>` subclasses (such classes convert user-typed text into
  corresponding Python values).
  
A detailed example for opening a file type can be found in :ref:`Bundle Example: Read a New File Format`.

.. _save command:

Saving Files
^^^^^^^^^^^^

For your bundle to save a file, it needs to provide information to the "save command" manager
about what data format it can save, what arguments it needs, what function to call, *etc.*.
Some of that info is provided as attributes in the `Provider`_ tag, but the lion's share is
provided when the save-command manager calls your bundle's
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider` method.
That call will only occur when ChimeraX tries to save the kind of data that your `Provider`_
tag says you can save.

To specify that your bundle can save a data format, you supply a `Provider`_ tag in the
`Providers`_ section of your **bundle_info.xml** file.  The value of
the ``manager`` attribute in the tag or section should be "save command".
The other possible `Provider`_ attributes are:

- **Mandatory** Attributes

    *name*
        The `name`_ of the `data format`_ you can save.  Can also be one of the format's
        `nicknames`_ instead.

- **Infrequently-Used** Attributes

    *compression_okay*
        *compression_okay* controls whether your format will be able to save directly as a compressed
        file as implied by the user adding an additional compression suffix (*e.g.* ".gz") to
        your file name.  There are two main reasons that you would change *compression_okay*
        from its default value of "true" to "false":

            1. For whatever reason your bundle cannot use
            :py:meth:`~chimerax.io.io.open_output` to open the file, which
            is the routine that handles the automatic compression.  This frequently happens for bundles
            where compiled code opens the file and cannot handle being passed a Python stream.

            2. If the data you are writing out is *already* compressed and therefore it would probably
            be bad to compress it again (likely slower with no space savings).

    *is_default*
        If your data format has suffixes that are the same as another format's suffixes, *is_default*
        will determine which format will be used when the save command's ``format`` keyword is omitted.
        *is_default* defaults to "true", so therefore typically lesser known/used formats supply this
        attribute with a value of "false".  For example, ChimeraX can save both image TIFF files and
        `ImageJ TIFF stacks <https://imagej.net/TIFF>`_, which both use the suffixes .tif and .tiff.
        The ImageJ TIFF stack uses ``is_default="false"`` so that the command ``save image.tif``
        produces the more commonly desired image file.  To get an ImageJ stack, the user would have
        to add ``format imagej`` to the save command.

For example::

  <Providers manager="save command">
    <Provider name="Sybyl Mol2" />
  </Providers>

The remainder of the information the bundle provides about how to save a file comes from the
return value of the bundle's
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider` method, which must return
an instance of the
:py:class:`chimerax.save_command.SaverInfo` class.
The doc strings of that class discuss its methods in detail, but briefly:

* You must override the :py:meth:`~chimerax.save_command.SaverInfo.save` method to take
  the input provided and save the file.

* If your format has format-specific keywords that the ``save`` command should accept,
  you must override the :py:meth:`~chimerax.save_command.SaverInfo.save_args` property
  to return a dictionary that maps **Python** keywords of your saver-function to corresponding
  :ref:`Annotation <Type Annotations>` subclasses (such classes convert user-typed text into
  corresponding Python values).

* If you have format-specific options and wish to show a user interface to some or all of those
  options in the ChimeraX Save dialog, you must override the
  :py:meth:`~chimerax.save_command.SaverInfo.save_args_widget` method and return a widget
  containing your interface (typically a subclass of
  `QFrame <https://doc.qt.io/qt-5/qframe.html>`_).
  Conversely, you must also override
  :py:meth:`~chimerax.save_command.SaverInfo.save_args_string_from_widget`
  that takes your widget and returns a string containing the corresponding options and
  values that could be added to a ``save`` command.
  
A detailed example for saving a file type can be found in :ref:`Bundle Example: Save a New File Format`.

.. _fetch command:

Fetching Files
^^^^^^^^^^^^^^

For your bundle to fetch a file from a web database, it needs to provide information to the
"open command" manager about what data format it can open, what arguments it needs,
what function to call, *etc.*.
Some of that info is provided as attributes in the `Provider`_ tag, but the lion's share is
provided when the open-command manager calls your bundle's
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider` method.
That call will only occur when ChimeraX tries to fetch the kind of data that your `Provider`_
tag says you can fetch.

To specify that your bundle can fetch from a database, you supply a `Provider`_ tag in the
`Providers`_ section of your **bundle_info.xml** file.  The value of
the ``manager`` attribute in the tag or section should be "open command".
The other possible `Provider`_ attributes are:

- **Mandatory** Attributes

    *format_name*
        The `name`_ of the `data format`_ for the data that is fetched.  Can also be one of
        the format's `nicknames`_ instead.

    *name*
        The name of the database that the data is fetched from, typically an easily typed
        lowercase string, since this name will be used directly in the ``open`` command
        as either the value for the ``fromDatabase`` keyword or as the prefix in the
        *from_database:identifier* form of fetch arguments.  So "pdb" is better then
        "Protein Databank".  Note that single-character database names are disallowed to
        avoid confusion with Windows single-character drive names.
        
    *type*
        *type* should be "fetch" to indicate that your bundle fetches data
        from the web (as opposed to opening local files).  The default is "open".

- **Frequently-Used** Attributes

    *example_ids*
        A list of one or more valid example identifiers for your database.  For use in
        graphical user interfaces.

    *synopsis*
        The description of the fetcher used by user-interface widgets that list fetchers
        (like the Fetch By ID dialog in Chimera), so typically somewhat more verbose than *name*.
        The first word should be capitalized unless that word is mixed case (*e.g.* mmCIF).
        Defaults to a capitalized *name* followed by the *format_name* in parentheses.

- **Infrequently-Used** Attributes

    *is_default*
        If a database can be fetched from using different `data format`_\s, the one that
        should be used when the user omits the ``format`` keyword should have *is_default*
        as "true", and the others should have it as "false".  *is_default* defaults to "true",
        so since most databases only have one format this attribute is in most cases omitted.

    *pregrouped_structures*
        If a provider returns multiple models, the open command will automatically group them
        so that the entire set of models can be referenced with one model number (the individual
        models can be referenced with submodel numbers).  The provider *could* pre-group them in
        order to give the group a name other the default (which is based on the database entry ID;
        the user can still override that with the ``name`` keyword of the open command).
        In the specific case where the provider is pre-grouping atomic structures, it should specify
        *pregrouped_structures* as "true" so the the open command's return value can be the actual list
        of structures rather than a grouping model.  This greatly simplifies scripts trying to handle
        return values from various kinds of structure-opening commands.

For example::

  <Providers manager="open command">
    <Provider name="pubchem" type="fetch" format_name="sdf" synopsis="PubChem" example_ids="12123" />
  </Providers>

The remainder of the information the bundle provides about how to fetch from a database comes
from the return value of the bundle's
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider` method, which must return
an instance of the
:py:class:`chimerax.open_command.FetcherInfo` class.
The doc strings of that class discuss its methods in detail, but briefly:

* You must override the :py:meth:`~chimerax.open_command.FetcherInfo.fetch` method to take
  the input provided and return a (models, status message) tuple.

* If your format has database-specific keywords that the ``open`` command should accept,
  you must override the :py:meth:`~chimerax.open_command.FetcherInfo.fetch_args` property
  to return a dictionary that maps **Python** keywords of your fetcher-function to corresponding
  :ref:`Annotation <Type Annotations>` subclasses (such classes convert user-typed text into
  corresponding Python values).  

  If the `data format`_ being fetched can also be opened directly from a file (*i.e.* there's
  an "open command" `Provider`_ with *type*\="open"), then 
  :py:meth:`~chimerax.open_command.FetcherInfo.fetch_args` should only return keywords applicable
  just to fetching.  The "opening" keywords will be automatically combined with those.

A detailed example for saving a file type can be found in :ref:`Bundle Example: Fetch from Network Database`.


.. _Defining Presets:

Defining Presets
----------------

For a bundle to define new presets,
it must have a `Providers`_ section in its **bundle_info.xml**
to provide the relevant information to the "presets" manager via one or more `Provider`_ tags.
The `Provider`_ tags are nested within the `Providers`_ section.
If your bundle only offers `Provider`_ tags for the "presets" manager, then you can put
the ``manager="presets"`` attribute in your `Providers`_ tag and that will apply to all the `Provider`_ tags
within the `Providers`_ section.  If your bundle offers `Provider`_ tags for multiple managers,
then you can either specify the manager within each `Provider`_ tag, or you can have
multiple `Providers`_ sections, each with their own ``manager`` attribute.

As per normal XML, `Provider`_ and `Providers`_ attributes are strings
(*e.g.* ``name="sticks"``).  Aside from "manager", the other possible `Provider`_ tags are:

- **Mandatory** Attributes

    *name*
        The name of the preset as shown in the Presets menu and as used by the ``preset`` command.
        Case does not matter.

- **Frequently-Used** Attributes

    *category*
        The category that the preset should be grouped into, as shown in the Presets menu
        and as used in the ``preset`` command.  Case does not matter.  Default is "General".

    *order*
        Controls the placement of the preset within its category in the Presets menu.
        Must be an integer (*e.g.* ``order="1"``).
        Default is to arrange presets in alphabetical order.

For example::

  <Providers manager="presets">
    <Provider category="fun looks" name="shiny balls" />
    <Provider category="fun looks" name="thin sticks" />
  </Providers>

When the execution of a preset from your bundle is requested, the preset manager will run the
:py:meth:`~chimerax.core.toolshed.BundleAPI.run_provider` method (with ``name`` and ``mgr`` arguments),
which should in turn execute the named preset.
So that the appropriate information about the preset gets logged,
your code implementing the preset should call ``mgr.execute(info)`` where ``info`` is
either a function that takes no arguments (if your preset is implemented in Python) or a list of commands.
