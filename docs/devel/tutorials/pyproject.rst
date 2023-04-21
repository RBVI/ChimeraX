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

.. _Bundle Information in pyproject.toml files:

Using ``pyproject.toml`` for Bundles
======================================
The python ecosystem is converging on using the pyproject.toml
file as the standard for configuring project metadata and Python
programming tools, such as setuptools (among others). ChimeraX is
no different. 

Bundle Builder is a wrapper around the ``setuptools.setup()`` function
that mostly serves to ensure that certain metadata crucial to bundles
is written correctly in the bundle's dist-info/METADATA file. 

Important Notes
---------------
There are a handful of small caveats when using Bundle Builder to build
a package:

- ``classifiers`` MUST be set as dynamic data in the ``[project]`` table.
- We RECOMMEND that ``requires-python`` be set as dynamic data in 
  the ``[project]`` table.
- Any classifier that would have gone into a ``<PythonClassifier>`` tag
  in **bundle_info.xml** files MUST be specified in the ``[chimerax]`` 
  table AND NOT the ``[project]`` table.
- Python requires underscores, but TOML convention is to use dashes. 
  Either is acceptable; however, any dashes in any key will be converted
  to underscores as the data is written to the bundle metadata.

Finally, the syntax of bundle builder has evolved but the semantics have 
not. This documentation is intentionally left sparse to minimize duplication
of documentation across pages. For detailed explanations of the metadata
listed here, see the page on :doc:`bundle_info`.

Dynamic Metadata
----------------
Because bundle builder piggybacks off of setuptools, it supports many of 
the same dynamic metadata fields that setuptools does. Supported tags are
listed below:

- version
- readme
- description

We have not evaluated whether other dynamic metadata that setuptools supports
works with ChimeraX, because they do not affect the function of the bundle as 
a part of ChimeraX, but all static metadata is supported.

We do not support user-specified dynamic classifiers because we use the 
``[chimerax]`` table to construct a list of classifiers that gives a bundle
functionality within ChimeraX.

Unlike other metadata, dynamic metadata should go under a ``[tool.setuptools.dynamic]`` table. 

For more information, see :doc:`setuptools's documentation. <setuptools:userguide/pyproject_config>`

Package Data and Extra Files
----------------------------
``[chimerax.package-data]`` and ``[chimerax.extra-files]`` can be used to specify
files to include in the distribution from within the source tree and from outside
the source tree, respectively. ::

    [chimerax.package-data]
    "src/" = ["colorbrewer.json"]

    [chimerax.extra-files]
    "src/include/chutil" = ["core_cpp/chutil/*.h"]
    "src/include/ioutil" = ["core_cpp/ioutil/*.h"]
    "src/include/logger" = ["core_cpp/logger/*.h"]
    "src/include/pysupport" = ["core_cpp/pysupport/*.h"]
    "src/include/mac_util" = ["core_cpp/mac_util/*.h"]

The general syntax is ``"target/directory" = ["list", "of", "source", "files"]``.

Platform-Specific Extra Files
----------------------------
You may choose to optionally include some extra files on certain platforms only. In this
case put them in a table such as: ::

    [chimerax.extra-files.platform.mac]

Configuring Basic Metadata
--------------------------
The basic bundle metadata, such as the minimum and maximum supported session versions, bundle 
categories, python classifiers, and whether your bundle needs a custom init should go in the
top of the ``[chimerax]`` table. For example: ::

    [chimerax]
    pure = true
    min-session-ver = 1
    max-session-ver = 1
    custom-init = false
    categories = ["General"]
    classifiers = []

Notes:

- ``pure`` is optional and defaults to True. In cases where pure is True and extensions,
  libraries, or executables are present, it will be overridden as False.
- ``category`` MUST be either a string or list of strings
- ``category`` MAY instead be labelled ``categories``
- ``custom-init`` is optional and assumed to be false if not present. 
- Python classifiers are not required but are encouraged

Declaring Commands
------------------
Use separate headings for each command, e.g. ::

    [chimerax.command.blastprotein]
    category = "Sequence"
    description = "Search PDB/NR/AlphaFold using BLAST"

Notes:

- The command name is part of the table label. This is a pattern which repeats
  for all other metadata.
- ``category`` MUST be either a string or list of strings
- ``description`` MAY instead be labelled ``synopsis``

Declaring Tools
---------------
Use separate headings for each tool or inline tables, e.g. ::

    [chimerax.tool."Blast Protein"]
    category = "Sequence"
    description = "..."

Notes:

- ``category`` MUST be either a string or list of strings
- ``description`` MAY instead be labelled ``synopsis``

Declaring Selectors
-------------------
You may either use separate headings for each tool or inline tables. ::

    [chimerax.selector]
    helix = { description = "Helical regions in proteins"}

or ::
    
    [chimerax.selector.helix]
    description = "Helical regions in proteins"

Notes:

- ``description`` MAY instead be labelled ``synopsis``

Declaring Providers
-------------------
Use separate tables for each provider. ::

    [chimerax.provider."Sybyl Mol2"]
    manager = "data formats"
    ...

Besides the manager and name, other attributes are passed as keyword arguments
to the manager's ``add_provider`` method. 


Declaring Data Formats
----------------------
Data formats are a special case of providers. When declaring them, putting them
under the providers table is not necessary. ::

    [chimerax.data-format."ChimeraX session"]
    category = "Session"
    encoding = "utf-8"
    nicknames = ["session"]
    mime-types = ["application/x-chimerax-code"]
    reference-url = "help:user/commands/usageconventions.html"
    suffixes = [".cxc"]
    description
    open = { want-path = true }
    save = {}
    insecure = false
    allow-directory = false

The above table is equivalent to one titled ``[chimerax.provider."ChimeraX session"]``
that includes ``manager = "data formats"`` in the table.

Field types:

- ``category`` MUST be either a string or list of strings
- ``nicknames`` MUST be either a string or list of strings
- ``suffixes`` MUST be either a string or list of strings 
- ``reference-url`` MUST be a string
- ``description`` MUST be a string
- ``insecure`` MUST be boolean
- ``allow-directory`` MUST be boolean
- ``mime-types`` MUST be a list of strings
- ``open`` MUST be an object, boolean, or TOML table
- ``save`` MUST be an object, boolean, or TOML table

Optional fields:

- ``category`` MAY be omitted and will default to ``"General"``
- ``encoding`` MAY be omitted and will default to ``"utf-8"``
- ``nicknames`` MAY be omitted and will default to the lower case version of the format name
- ``reference-url`` MAY be omitted and will default to ``None``
- ``suffixes`` MAY be omitted, but omitting them will require specifying the format when opening or saving with a ChimeraX command
- ``description`` MAY be omitted and will default to the format name
- ``save`` MAY be omitted and will default to ``false``
- ``open`` MAY be omitted and will default to ``false``

Alternative field labels:

- ``description`` MAY instead be labelled ``synopsis``
- ``category`` MAY instead be labelled ``categories``
- ``suffixes`` entries MAY omit the leading period

Declaring File Openers
----------------------
At the most basic level, file opening can be enabled by setting 
``open`` to ``true`` in the format's root table. When using a 
boolean value, all the options below are set to their defaults. ::

    [chimerax.data-format."ChimeraX session"]
    ...
    [chimerax.data-format."ChimeraX session".open]
    want-path = true 
    batch = false
    check_path = false
    is_default = true
    pregrouped_structures = true
    type = "open"


To simply customize one value, the table can be inlined: ::

    [chimerax.data-format."ChimeraX session"]
    ...
    open = { want-path = true }
 
Or you can specify the one value you want like so: ::

    [chimerax.data-format."ChimeraX session"]
    ...
    open.want-path = true 
 
Finally, data formats with many openers can use TOML tables to declare many
openers at the same time: ::

    [chimerax.data-format."web fetch"]
    category = "I/O"
    
    [[chimerax.data-format."web fetch".open]]
    name = "http"
    type = "fetch"
    
    [[chimerax.data-format."web fetch".open]]
    name = "https"
    type = "fetch"
    
    [[chimerax.data-format."web fetch".open]]
    name = "ftp"
    type = "fetch"

Each type will be correctly associated with the web fetch provider.

Declaring File Savers
---------------------
At the most basic level, file saving can be enabled by setting 
``save`` to ``true`` in the format's root table. When using a 
boolean value, all the options below are set to their defaults. ::

    [chimerax.data-format."ChimeraX session".save]
    compression-okay = false
    is-default = true

Declaring Presets
-----------------
Presets are another special case of providers, and so they are not required
to be placed under the provider table either. ::

    [chimerax.preset."thin sticks"]
    category = "fun looks"

Declaring Managers
------------------
Managers may either be declared as separate tables or a list of inline tables. ::

    [chimerax.manager.foo]
    gui-only = true
    autostart = true

or ::

    [chimerax.manager]
    bar = { gui-only = true, autostart = true }

Notes:
- ``gui-only`` MAY be omitted and will default to ``false``.
- ``autostart`` MAY be omitted and will default to ``true``.

Declaring Initializations
-------------------------
Initializations are a list of bundles that must be initialized before your bundle.
The supported types of initializations are **manager** and **custom**. Managers
across all bundles are initialized first, then custom initializations across all
bundles. Initializations can be declared as follows: ::

    [chimerax.initializations.manager]
    bundles = []

or ::

    [chimerax.initializations.custom]
    bundles = []

Notes:

- ``bundles`` MUST be a string or list of strings.

Source Extensions
=================
The other main function of bundle builder is ensuring compatibility between
compiled extensions meant to be used within ChimeraX. We want to make the 
process of building bundles as simple as possible for bundle developers, so 
we've made bundle builder able to build extensions, libraries, and executables.

All of the fields that are normally available to the ``setuptools.Extension``
initializer are available in Bundle Builder with two additions: 

- ``include-modules``

  and

- ``lib-modules``

Each is expected to be a list of python modules on which your extension, library, 
or executable depends. At build time, Bundle Builder will attempt to import the 
modules and call either ``module.get_include()`` or ``module.get_lib()`` depending
on which list the module appears in. 

Inspired by ``numpy.get_include()``, these 
functions MUST take no arguments and MUST return a path to the the package's 
include or library directory, so that the compiler and linker respectively can find 
the headers or libraries required.

Declaring Modules, Libraries, and Executables
---------------------------------------------

All of the fields that are normally passed to ``setuptools.Extension`` objects are
exposed in bundle builder:::

    [chimerax.extension.ioutil]
    sources = []
    include-dirs = []
    define-macros = []
    undef-macros = []
    library-dirs = []
    libraries = []
    runtime-library-dirs = []
    extra-objects = []
    extra-compile-args = []
    extra-link-args = []
    export-symbols = []
    depends = []
    language = ""
    optional = false
    include-modules = []
    lib-modules = []

Notes:

- The name argument is taken from the table label.

- The difference between declaring a module or library is that modules should use
  ``[chimerax.extension]`` and libraries should use ``[chimerax.library]``. 

- Using ``[chimerax.executable]`` will place an executable binary in the bundle's ``src/bin`` directory.

- Libraries may specify a ``static`` parameter, assumed false if absent, to build a 
  static library.

Platform-specific arguments can be specified in subtables just like 
the above example for file openers. Accepted platform keys are 
``mac, macos, darwin, linux, windows, win, win32``.

Platform-specific arguments will be concatenated to the general arguments.

Take care when compiling libraries or executables. Because setuptools does not
handle these especially well, we build them in-place in the source tree, so 
be sure to add them to your makefile's ``clean`` target.

Platform-Specific Extensions
----------------------------
The only required arguments to ``setuptools.Extension()`` are a name and a list of 
source files. The list of source files can even be an empty list! In that case, no
extension is compiled.

In the above example we wrote the metadata for the ``ioutil`` extension in ChimeraX.
If instead we had written ::

    [chimerax.extension.ioutil.platform.mac]

The net effect would be an extension that is only compiled on macOS.

.. TODO: A documentation tag
