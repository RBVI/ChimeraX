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

- ``tut_read`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``io.py`` - source code to read XYZ format files

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

.. literalinclude:: ../../../src/examples/tutorials/tut_read/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-24,36-45

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-24).

The ``Providers`` sections on lines 36 through 45 use the
:ref:`Manager <Manager>`/:ref:`Provider <Provider>` protocol to inform
the "data formats" manager about the XYZ format, and the "open command"
manager that this bundle can open XYZ files.

The attributes usable with the "data formats" manager are described in
detail in :ref:`data format`.  Note that most formats have a longer
official name than "XYZ" and therefore most formats will also specify
``nicknames`` and ``synopsis`` attributes, whereas they are unneeded
in this example.

Similarly, the "open command" attributes are described in detail in
:ref:`open command`.  It *is* typical that the only attribute
specified is ``name``.

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

.. literalinclude:: ../../../src/examples/tutorials/tut_read/src/__init__.py
    :language: python
    :linenos:

The :py:meth:`run_provider` method is called by a ChimeraX manager
when it needs additional information from a provider or it needs a
provider to execute a task.
The *session* argument is a :py:class:`~chimerax.core.session.Session` instance,
the *name* argument is the same as the ``name`` attribute in your Provider
tag, and the *mgr* argument is the manager instance.
These arguments can be used to decide what to do when your bundle offers
several Provider tags (to possibly several managers), but since the "data
formats" manager never calls :py:meth:`run_provider`, we can customize the
routine specifically for the "open command" manager and don't need to check
the :py:meth:`run_provider` arguments.

When called by the "open command" manager, :py:meth:`run_provider` must return
an instance of a subclass of :py:class:`chimerax.open_command.OpenerInfo`.
The methods of the class are thoroughly documented if you click the preceding
link, but briefly:

1. The :py:meth:`open` method is called to actually open/read the file and
   should return a (models, status message) tuple.  The method's *data*
   argument is normally an opened stream encoded as per the format's ``encoding``
   attribute (binary if omitted), but it can be a file path if certain
   Provider attributes were specified (most often, ``want_path="true"``).
2. If there are format-specific keyword arguments that the ``open`` command should
   handle, then an :py:meth:`open_args` property should be implemented, which
   returns a dictionary mapping **Python** keyword names to :ref:`Annotation <Type Annotations>`
   subclasses.  Such keywords will be passed to your :py:meth:`open` method.
3. So long as your :py:meth:`open` method accepts a stream, opening compressed
   files of your format (e.g. with additional suffixes such as .gz, .bz2) will
   be handled automatically.  For path-based openers, such files will result in an
   error before your opener is called.
4. If for some reason the opened file should not appear in the file history,
   set :py:attr:`in_file_history` to ``False``.


``src/io.py``
-------------

.. literalinclude:: ../../../src/examples/tutorials/tut_read/src/io.py
    :language: python
    :linenos:

The :py:func:`open_xyz` function is called from the
:py:meth:`__init__.bundle_api.open_file`
method to open an input file in `XYZ format`_.  The contents of such
a file is a series of blocks, each representing a single molecular
structure.  Each block in an XYZ format file consists of

- a line with the number atoms in the structure,
- a comment line, and
- one line per atom, containing four space-separated fields: element type
  and x, y, and z coordinates.

The return value that ChimeraX expects from :py:meth:`open_xyz` is a 2-tuple
of a list of structures and a status message.  The :py:meth:`open_xyz` code
simply initializes an empty list of structures (line 10) and repeatedly
calls :py:func:`_read_block` until the entire file is read (lines 14-20).
When :py:func:`read_block` successfully reads a block, it returns an instance
of :py:class:`chimerax.atomic.structure.AtomicStructure`,
which is added to the structure list (line 18); otherwise,
it returns **None** which terminates the block-reading loop (lines 16-17).
A status message is constructed from the total number of structures,
atoms, and bonds (lines 21-22).
The structure list and the status message are then returned to
ChimeraX for display (line 23).

:py:func:`_read_block` reads and constructs an atomic
structure in several steps:

#. read the number of atoms in the block (lines 32-43).
#. build an empy atomic structure to which atoms will be added
   (lines 45-51).
   The :py:class:`chimerax.atomic.structure.AtomicStructure`
   instance is created on line 50, and a
   :py:class:`chimerax.atomic.molobject.Residue` instance
   is created on line 51.  The latter is required because ChimeraX
   expects every atom in a structure to be part of exactly one residue
   in the same structure.  Even though XYZ format does not support the
   concept of residues, a *dummy* one is created anyway.
#. skip the comment line (lines 61-63).
#. loop over the expected number of atoms and add them to the structure
   (lines 68-95).  The construction of a
   :py:class:`chimerax.atomic.molobject.Atom`
   instance is somewhat elaborate (lines 83-95).  First, the atom
   parameters are prepared: the atomic coordinates are extracted from
   the input (line 87), and the atom name is constructed from the
   element type and an element-specific running index (lines 88-91).
   The **Atom** instance is created on line 95, using the convenience
   function :py:func:`chimerax.atomic.struct_edit.add_atom` which
   also adds it to the residue, and sets its coordinates.
#. XYZ format files do not have connectivity information, so no bonds
   are created while processing input lines.  Instead, the
   :py:meth:`~chimerax.atomic.molobject.StructureData.connect_structure`
   method of the structure is called to deduce connectivity from
   interatomic distances (line 98).
#. Return success or failure to read a structure to ``open_xyz`` (line 101).


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
