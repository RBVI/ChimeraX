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
see :doc:`tutorial_hello`, :doc:`tutorial_command`,
:doc:`tutorial_tool`, and :doc:`tutorial_read_format`.

.. literalinclude:: ../../../src/examples/tutorials/tut_save/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-23,35-48

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-23).

The ``Providers`` sections on lines 36 through 48 use the
:ref:`Manager <Manager>`/:ref:`Provider <Provider>` protocol to inform
the "data formats" manager about the XYZ format, and the "open command"
and "save command" managers, respectively,  that this bundle can open
and save XYZ files,

The attributes usable with the "data formats" manager are described in
detail in :ref:`data format`.  Note that most formats have a longer
official name than "XYZ" and therefore most formats will also specify
``nicknames`` and ``synopsis`` attributes, whereas they are unneeded
in this example.

The "open command" attributes are described in detail in
:ref:`open command`.
Likewise, the "save command" attributes are described in detail in
:ref:`save command`.
It *is* typical that the only attribute specified is ``name``.


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
    :emphasize-lines: 29-36,45-83

The :py:meth:`run_provider` method is called by a ChimeraX manager
when it needs additional information from a provider or it needs a
provider to execute a task.
The *session* argument is a :py:class:`~chimerax.core.session.Session` instance,
the *name* argument is the same as the ``name`` attribute in your Provider
tag, and the *mgr* argument is the manager instance.
These arguments can be used to decide what to do when your bundle offers
several Provider tags, such as in this example.
The "data formats" manager never calls :py:meth:`run_provider`, so we only
need to know if it's the "open command" or "save command" manager calling
this method.
This "open command" manager is also ``session.open_command`` (and "save command"
is ``session.save_command``), so we use the test on line 36 to decide.

The information needed by the "open command" manager is returned by the code on
lines 37-44 and is described in detail in :doc:`tutorial_read_format`.

When called by the "save command" manager, :py:meth:`run_provider` must return
an instance of a subclass of :py:class:`chimerax.save_command.SaverInfo`.
The methods of the class are thoroughly documented if you click the preceding
link, but briefly:

1. The :py:meth:`save` method is called to actually save the file (and has no
   return value).  The method's *path* is the full path name of the file to save.
2. If there are format-specific keyword arguments that the ``save`` command should
   handle, then a :py:meth:`save_args` property should be implemented, which
   returns a dictionary mapping **Python** keyword names to :ref:`Annotation <Type Annotations>`
   subclasses.  Such keywords will be passed to your :py:meth:`save` method.
3. If your underlying file-writing function uses :py:func:`~chimerax.io.io.open_output`
   to open the path, then compression implied by the file name (*e.g.* a additional
   .gz suffix) will be handled automatically.
4. In the rare case where you save a file type that ChimeraX knows how to open but would
   be inappriate to open for some reason, set :py:attr:`in_file_history` to ``False``
   to exclude it from the file history listing.


``src/io.py``
--------------

.. literalinclude:: ../../../src/examples/tutorials/tut_save/src/io.py
    :language: python
    :linenos:
    :emphasize-lines: 109-147

The ``open_xyz`` and ``_read_block`` functions are described in
detail in :doc:`tutorial_read_format`.

The ``save_xyz`` function performs the following steps:

- open the output file for writing using the ChimeraX function
  :py:func:`~chimerax.io.io.open_output` (lines 112-116),
- if the *structures* keyword was not given, include all atomic structures
  for saving (lines 118-121),
- initialize the total atom count (line 122),
- loop through the structures to save (line 125) and:

  - get the lists of atoms and coordinates for the structure.  (lines 129-130),
  - print the first two lines (number of atoms and comment) for the
    structure to the file (lines 132-135),
  - print one line per atom using the atom and coordinates lists, and
  - update total atom count (lines 136-141).
- close the output file (line 142)
- finally, log a status message to let the user know what was written
  (lines 144-146).


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
