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

.. _Bundle Example\: Read a New File Format:


=======================================
Bundle Example: Read a New File Format
=======================================

This example describes how to create a ChimeraX bundle
that allows ChimeraX to open data files in `XYZ format`_,
which is a simple format containing only information
about atomic types and coordinates.

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
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_read>`_
containing a folder named `tut_read`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_read`` folder are:

``tut_read`` - bundle folder
    ``bundle_info.xml`` - bundle information read by ChimeraX
    ``src`` - source code to Python package for bundle
        ``__init__.py`` - package initializer and interface to ChimeraX
        ``io.py`` - source code to read XYZ format files

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
`tutorial_tool`.

.. literalinclude:: ../../../src/examples/tutorials/tut_read/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-24,41-43

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-24).

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

The **Open** classifier states that the bundle supports opening **XYZ** files;
the second **XYZ** field is a currently unused (but required) tag.


``src``
-------

.. include:: src.rst


``__init__.py``
---------------

As described in :doc:`tutorial_hello`, ``__init__.py`` contains
the initialization code that defines the ``bundle_api`` object
that ChimeraX needs in order to invoke bundle functionality.
ChimeraX expects ``bundle_api`` class to be derived from
:py:class:`chimerax.core.toolshed.BundleAPI` with methods
overridden for registering commands, tools, etc.

.. literalinclude:: ../../../src/examples/tutorials/tut_read/src/__init__.py
    :language: python
    :linenos:

The ``open_file`` method is called by ChimeraX to read a file,
and return a list of models and a status message.  Unlike standard
Python methods, the parameter names are significant because ChimeraX
introspects the method definition to construct the arguments that
are passed to the method.

1. The first argument must be named **session**,
   and corresponds to a :py:class:`chimerax.core.session.Session` instance.
2. The second argument must be named either **path** or **stream**.
   If **path**, the argument corresponds to the path to the input file;
   if **stream**, it corresponds to a file-like object.
3. The third argument must be named **format_name**, and corresponds
   to a string with the full name of the format that is either specified
   by the user, *e.g.*, via an argument to the **open** command, or
   deduced by ChimeraX from the file suffix.
4. An optional fourth argument, **file_name** may be supplied when
   both the file path and file-like object are needed.

If the ``open_file`` method expects a file-like stream, ChimeraX
takes care of decompressing files with **.gz** suffix, as well as
opening the file in either text or binary mode, matching the format
specification in ``bundle_info.xml``.

In this example, the file path is not required, so the three arguments
given are **session**, **stream**, and **format_name**.
If the bundle supported multiple formats, **format_name** may
be used to select the appropriate reader function.
In this example, only one format is supported, so
``io.open_xyz`` is called directly.


``io.py``
----------

.. literalinclude:: ../../../src/examples/tutorials/tut_read/src/io.py
    :language: python
    :linenos:

The ``open_xyz`` function is called from the ``__init__.bundle_api.open_file``
method to open an input file in `XYZ format`_.  The contents of such
a file is a series of blocks, each representing a single molecular
structure.  Each block in an XYZ format file consists of

- a line with the number atoms in the structure,
- a comment line, and
- one line per atom, containing four space-separated fields: element type
  and x, y, and z coordinates.

The return value that ChimeraX expects from ``open_xyz`` is a 2-tuple
of a list of structures and a status message.  The ``open_xyz`` code
simply initializes an empty list of structures (line 10) and repeatedly
calls ``_read_block`` until the entire file is read (lines 14-20).
When ``read_block`` successfully reads a block, it returns an instance
of ``chimerax.core.atomic.AtomicStructure``, which is added to the
structure list (line 18); otherwise, it returns **None** which
terminates the block-reading loop (lines 16-17).
A status message is constructed from the total number of structures,
atoms, and bonds (lines 21-22).
The structure list and the status message are then returned to
ChimeraX for display (line 23).

``_read_block`` reads and constructs an atomic structure in several steps:

1. read the number of atoms in the block (lines 32-43).
2. build an empy atomic structure to which atoms will be added
   (lines 45-51).  The ``chimerax.core.atomic.AtomicStructure`` instance
   is created on line 50, and a ``chimerax.core.atomic.Residue`` instance
   is created on line 51.  The latter is required because ChimeraX
   expects every atom in a structure to be part of exactly one residue
   in the same structure.  Even though XYZ format does not support the
   concept of residues, a *dummy* one is created anyway.
3. skip the comment line (lines 61-63).
4. loop over the expected number of atoms and add them to the structure
   (lines 66-94).  The construction of a ``chimerax.core.atomic.Atom``
   instance is somewhat elaborate (lines 80-94).  First, the atom
   parameters are prepared: the atomic coordinates are extracted from
   the input (line 84), and the atom name is constructed from the
   element type and an element-specific running index (lines 85-88).
   The **Atom** instance is created on line 92; the newly created atom
   is part of the structure being built through the use of the ``new_atom``
   method of the structure.  The atomic coordinates are set on line 93.
   Finally, the atom is added to the dummy residue on line 94.
5. XYZ format files do not have connectivity information, so no bonds
   are created while processing input lines.  Instead, the
   ``~chimerax.core.atomic.AtomicStructure.connect_structure``
   method of the structure is called to deduce connectivity from
   interatomic distances (line 97).
6. The structure is *finalized* with the call to ``new_atoms`` (line 103).
   Some atomic structure data, such as atom and bond types, change as
   atoms and bonds are added.  Rather than recomputing on every change,
   ChimeraX waits until a call to ``new_atoms`` before updating
   structure data.
7. Return success or failure to read a structure to ``open_xyz`` (line 106).


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello`
- :doc:`tutorial_command`
- :doc:`tutorial_tool` (previous topic)
- :doc:`tutorial_read_format` (current topic)
- :doc:`tutorial_save_format` (next topic)
- :doc:`tutorial_fetch`
- :doc:`tutorial_selector`
