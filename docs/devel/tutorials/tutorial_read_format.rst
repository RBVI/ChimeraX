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

.. literalinclude:: ../../../src/examples/tutorials/tut_read/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-25,41-45


``src``
-------

.. include:: src.rst


``__init__.py``
---------------

.. literalinclude:: ../../../src/examples/tutorials/tut_read/src/__init__.py
    :language: python
    :linenos:


``io.py``
----------

.. literalinclude:: ../../../src/examples/tutorials/tut_read/src/io.py
    :language: python
    :linenos:


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
