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

.. _Bundle Example\: Save a New File Format:


=======================================
Bundle Example: Save a New File Format
=======================================


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

``tut_save`` - bundle folder
    ``bundle_info.xml`` - bundle information read by ChimeraX
    ``src`` - source code to Python package for bundle
        ``__init__.py`` - package initializer and interface to ChimeraX
        ``io.py`` - source code to read and save XYZ format files

The file contents are shown below.


``bundle_info.xml``
-------------------

.. literalinclude:: ../../../src/examples/tutorials/tut_save/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-25,41-45


``src``
-------

.. include:: src.rst


``__init__.py``
---------------

.. literalinclude:: ../../../src/examples/tutorials/tut_save/src/__init__.py
    :language: python
    :linenos:


``io.py``
----------

.. literalinclude:: ../../../src/examples/tutorials/tut_save/src/io.py
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
