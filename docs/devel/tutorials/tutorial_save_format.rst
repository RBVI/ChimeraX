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

.. include:: references.rst

.. _Bundle Example\: Save a New File Format:


=======================================
Bundle Example: Save a New File Format
=======================================

This example describes how to create a ChimeraX bundle
that allows ChimeraX to open and save data files in `XYZ format`_,
which is a simple format containing only information
about atomic types and coordinates.  The example files
are almost identical to those from :doc:`tutorial_read_format`,
with a few additions for saving XYZ files.

Code for both reading and writing a new format is typically
supplied in the same bundle.  However, an alternative is to
have separate bundles for reading and writing, by making
one bundle dependent on the other.  For example, the base
bundle can define the data format and handle open requests;
the dependent bundle can then handle only save requests
(using the same data format definition from the base bundle).

The steps in implementing the bundle are:

#. Create a ``bundle_info.xml`` containing information
   about the bundle,
#. Create a Python package that interfaces with ChimeraX
   and implements the file-reading functionality, and
#. Install and test the bundle in ChimeraX.

The final step builds a Python wheel that ChimeraX uses
to install the bundle.  So if the bundle passes testing,
it is immediately available for sharing with other users.


Source Code Organization
========================

The source code for this example may be downloaded
as a `zip-format file
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_save>`_
containing a folder named `tut_save`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_save`` folder are:

- ``tut_save`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``io.py`` - source code to read and save XYZ format files

The file contents are shown below.


``bundle_info.xml``
-------------------

``bundle_info.xml`` is an `eXtensible Markup Language`_
format file whose tags are listed in :doc:`bundle_info`.
While there are many tags defined, only a few are needed
for bundles written completely in Python.  The
``bundle_info.xml`` in this example is similar to the one
from the :doc:`tutorial_tool` example with changes highlighted.
For explanations of the unhighlighted sections, please
see :doc:`tutorial_hello`, :doc:`tutorial_command` and
:doc:`tutorial_tool`.

.. literalinclude:: ../../../src/examples/tutorials/tut_save/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-23,41-43

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-23).

The ``ChimeraXClassifier`` tags on lines 41-43 informs ChimeraX that
this bundle supports reading a data format named **XYZ**.
The **DataFormat** classifier consists of several fields after
the format name:

- an optional comma-separated list of alternative names for the format
  (none in this example).
- the category of data stored in this format (**Molecular structure**).
- a comma-separated list of suffixes that files in this format may use
  (**.xyz**).
- the MIME types associated with the format (none in this example).
- the URL to the format specifications
  (**https://en.wikipedia.org/wiki/XYZ_file_format**).
- whether the format potentially contains dangerous data, *e.g.*,
  an executable script (not in this example).  If the format
  supports script, this field should be set to **true** and users
  would be asked whether to try to open a file of this format.
- the path to an icon for files in this format (none in this example).
- the description for the format to show to users (**XYZ format**).
- the encoding for the file contents (**utf-8**).

The **Open** classifier fields are:

- the name of the data format (in this example, **XYZ**),
- a (currently unused) tag name (**XYZ**), and
- a boolean value for whether this bundle should be the default handler
  for the named data format (none, defaulting to **false**).  Bundles
  that provide the canonical format reader for a format should set this
  value to **true**.

The **Save** classifier fields are:

- the name of the data format (in this example, **XYZ**),
- a (currently unused) tag name (**XYZ**),
- a boolean value for whether this bundle should be the default handler
  for the named data format (none, defaulting to **false**).  Bundles
  that provide the canonical format writer for a format should set this
  value to **true**, and
- descriptions for **save** command keywords accepted for this data format
  (**models:Models**).  The descriptions are a comma-separated list of
  colon-separated keyword-*datatype* pairs.  *datatype*
  must match one of the type names from :py:mod:`chimerax.core.commands`
  less a **Arg** suffix.  In this example, the **save** command will
  accept a **models** keyword when an XYZ file is saved; the syntax
  for the **models** argument matches
  :py:class:`~chimerax.core.commands.cli.ModelsArg`
  (*i.e.*, **Models** + **Arg**).

``src``
-------

.. include:: src.rst


``src/__init__.py``
-------------------

As described in :doc:`tutorial_hello`, ``__init__.py`` contains
the initialization code that defines the ``bundle_api`` object
that ChimeraX needs in order to invoke bundle functionality.
ChimeraX expects ``bundle_api`` class to be derived from
:py:class:`chimerax.core.toolshed.BundleAPI` with methods
overridden for registering commands, tools, etc.

.. literalinclude:: ../../../src/examples/tutorials/tut_save/src/__init__.py
    :language: python
    :linenos:
    :emphasize-lines: 35-48

The ``open_file`` method is overridden to handle opening XYZ data
files, and is described in detail in :doc:`tutorial_read_format`.

The ``save_file`` method is called by ChimeraX to save a file.
The first two arguments are **session**, a
:py:class:`chimerax.core.session.Session` instance; and
**path**, the path to the output file as a string.
An optional **format_name** argument may be listed if
the bundle supports multiple data formats.
Additional arguments should match the keyword arguments listed
by the **Save** classifier in **bundle_info.xml**.

For this example, the **format_name** argument is omitted because
the bundle only supports XYZ format.  The **models** argument is
listed on line 37, matching the **bundle_info.xml** specification.
All three arguments are passed through to ``io.save_xyz`` to
actually save the models to the output file in XYZ format.


``src/io.py``
--------------

.. literalinclude:: ../../../src/examples/tutorials/tut_save/src/io.py
    :language: python
    :linenos:
    :emphasize-lines: 109-150

The ``open_xyz`` and ``_read_block`` functions are described in
detail in :doc:`tutorial_read_format`.

The ``save_xyz`` function performs the following steps:

- open the output file for writing using the ChimeraX function
  :py:func:`chimerax.core.io.open_filename` (lines 112-114),
- if the **models** keyword was not given, include all atomic structures
  for saving (lines 116-119),
- initialize some statistics counters (lines 120-121),
- loop through the structures to save (line 124) and:

  - try to get the lists of atoms and coordinates for the structure.
    If that fails, assume that the model is not an atomic structure
    and skip it (lines 125-134),
  - print the first two lines (number of atoms and comment) for the
    structure to the file (lines 136-140),
  - print one line per atom using the atom and coordinates lists, and
  - update statistics (lines 134 and 145).

- finally, log a status message to let the user know what was written
  (lines 147-149).


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello`
- :doc:`tutorial_command`
- :doc:`tutorial_tool`
- :doc:`tutorial_read_format` (previous topic)
- :doc:`tutorial_save_format` (current topic)
- :doc:`tutorial_fetch` (next topic)
- :doc:`tutorial_selector`
