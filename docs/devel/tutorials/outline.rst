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


ChimeraX Bundles
================

- Bundles provide an avenue for adding functionality to ChimeraX
  without modifying core code, much as apps do for Cytoscape or
  add-ons for Firefox.

- Bundles can add support for new commands, graphical interfaces
  (tools), file formats, and selectors.

- Bundles are actually Python *wheels* and may be installed using
  standard Python tools such as *pip*.

- ChimeraX also supports a *toolshed*, serving the same purpose
  as the *Cytoscape App Store*.  Users can make their bundles
  available to the ChimeraX community by contributing it to the
  toolshed, which supports custom licenses, download tracking, etc.


Bundle Organization
-------------------

- Bundles are split into two parts: bundle information and Python
  package.

  - ChimeraX is designed to start quickly, which means it does
    execute code from every bundle for initialization.
  - Instead the *bundle information*, which is processed once
    when a bundle is installed, is used at initialization time,
    e.g., to register commands and populate menus.
  - Code in the bundle Python package is only executed when
    commands or menu items are invoked.
  - If absolutely necessary, there is a mechanism for bundle code
    to be executed at ChimeraX startup, but the practice is
    discouraged.

- ChimeraX bundles are built using the ChimeraX ``dev`` commands.

  - Normally, Python *wheels* are built using **setup.py** files
    which describe all the files that are needed for installing
    the Python package.
  - To simplify writing bundles, ChimeraX combines the specification
    of required files with supplying bundle information into a
    single **bundle_info.xml** file.
  - When a ``dev install`` or ``dev build`` command is issued,
    ChimeraX automatcally generates a **setup.py** file and
    builds the bundle (using standard Python tools).

- The ``dev`` commands depends on the bundle source being organized
  in a specific way.

  - The folder for the bundle must contain a **bundle_info.xml**
    file and a **src** sub-folder.
  - The **src** folder must contain the Python code, in the form
    of a Python package, that go into the bundle.  For example,
    there must be a **src/__init__.py** file.
  - Additional support files, such as icons, data files, etc.,
    may be placed in either **src** or other sub-folders and
    must be explicitly listed in **bundle_info.xml** for them
    to be included in the generated bundle.


Bundle Information
------------------

- Bundle information includes items such as the name and version
  of the bundle, the name of the corresponding Python package,
  the commands, tools and file formats supported by the bundle, etc.

  - Bundle information is kept in XML format in **bundle_info.xml**.

  - Descriptions of what ChimeraX functionality are supported are
    listed in *Classifier* XML tags, while information about what
    files to include, source file dependencies, contact addresses,
    etc. are listed in "normal" *Bundle* XML tags listed below.

  - Classifier tags are split into Python and ChimeraX classifiers.
    Python classifiers are Python-wheel-style classifiers such as 
    *Development Status* and *License*.  ChimeraX classifiers are
    described below.


  TODO: FOR SUBTOPICS BELOW, COPY FROM "Building and Distributing Bundles"

  Bundle XML tags
  ---------------

  Python Classifier XML tags
  --------------------------

  ChimeraX Classifier XML tags
  ----------------------------


Bundle Python Code
------------------

- Bundle source is typically arranged as a Python package in the
  **src** folder.  ChimeraX interfaces with bundles by using the
  ``bundle_api`` object defined in the bundle package.  All other
  files in the bundle are referenced through ``bundle_api``.
  That is, ChimeraX does *not* introspect other bundle package
  files for additional information.

  - The ``bundle_api`` object must be an instance of a class
    derived from ``chimerax.core.toolshed.BundleAPI``, which
    defines methods ChimeraX uses to get bundles to:
   
      - start graphical interfaces (``start_tool``),
      - register commands (``register_command``),
      - register selectors (``register_selector``),
      - open files (``open_file``),
      - save files (``save_file``),
      - execute initialization and finalization code
        (``initialize``, ``finish``), and
      - when restoring a ChimeraX session, return class objects
        corresponding to class names save in session files
        (``get_class``).

    A bundle must define a subclass of ``BundleAPI`` that overrides
    the appropriate methods for the functionality it provides.  For
    example, a bundle should override ``start_tool`` and
    ``register_command`` if it provides both new graphical interfaces
    and commands.

    - TODO: MORE HERE FOR EACH METHOD

  - ChimeraX defined APIs for managing data, e.g., getting a
    list of open models, changing the state of atomic structures,
    creating new graphical objects, etc.  These APIs are described
    in the Programmer Reference Documentation.

  - The best way to learn to writing a ChimeraX bundle is to
    start with a sample bundle.

    - TODO: LINKS TO SAMPLES (BASIC, C++, etc.)


Building a Bundle
-----------------

- The recommended way to build bundles is from ChimeraX via
  the ``dev`` command.

  - ``dev install *source_folder*``

  - ``dev build *source_folder*``

  - ``dev clean *source_folder*``

  - ``dev`` commands may be invoked from the command line (e.g.,
    when the bundle is built is part of a batch process).  For
    example,
      ``chimerax --nogui --exit --cmd 'dev install BUNDLE_FOLDER exit true'``
    should build a bundle without displaying any graphics.


Distributing a Bundle
---------------------

- Once built, ChimeraX bundles are simply Python wheels and may
  be installed using the ``dev install`` command.  It is important
  to use the command rather than ``pip`` because ChimeraX needs to
  include the bundle information in the start-up cache so that the
  supported commands and tools appear in the user interface.

- Bundles may also be made available to the general community by
  submitting them to the ChimeraX toolshed.

  - Login using a Google account and use the ``Submit a Bundle``
    link at the top of the page to start the process.

  - Some basic checks are made when a bundle is submitted, like
    whether the bundle name is conformant, etc.

  - The first time a bundle is submitted, it is held for inspection
    by the ChimeraX team, which may contact the authors for more
    information.  Once the team approves the submission, the bundle
    appears on the toolshed.  Subsequence submissions of newer
    versions of the bundle are posted without having to wait
    for approval.
