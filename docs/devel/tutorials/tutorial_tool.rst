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

.. _Bundle Example\: Add a Tool:


===========================
Bundle Example: Add a Tool
===========================

This tutorial builds on the material from :doc:`tutorial_command`.


Source Code Organization
========================

The source code for this example may be downloaded
as a `zip-format file
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/XXX>`_
containing a folder named `tut_gui`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_gui`` folder are:

``tut_gui`` - bundle folder
    ``bundle_info.xml`` - bundle information read by ChimeraX
    ``src`` - source code to Python package for bundle
        ``__init__.py`` - package initializer and interface to ChimeraX
        ``gui.py`` - source code to implement ``Tutorial GUI`` tool

The file contents are shown below.


``bundle_info.xml``
-------------------

.. literalinclude:: ../../../src/examples/tutorials/tut_gui/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-25,35,38-41,48-50

For explanations of the unhighlighted sections, please
see :doc:`tutorial_hello` and :doc:`tutorial_command`.


``src``
-------

``src`` is the folder containing the source code for the
Python package that implements the bundle functionality.
The ChimeraX ``devel`` command automatically includes all
``.py`` files in ``src`` as part of the bundle.  Additional
files, such as HTML source code, are included using the
bundle information tag ``DataFiles`` as shown above.
The only required file in ``src`` is ``__init__.py``.
Other ``.py`` files are typically arranged to implement
different types of functionality.  For example, ``cmd.py``
is used for command-line commands; ``tool.py`` or ``gui.py``
for graphical interfaces; ``io.py`` for reading and saving
files, etc.


``__init__.py``
---------------

.. literalinclude:: ../../../src/examples/tutorials/tut_gui/src/__init__.py
    :language: python
    :linenos:


``gui.py``
----------

.. literalinclude:: ../../../src/examples/tutorials/tut_gui/src/gui.py
    :language: python
    :linenos:

.. include:: build_test_distribute.rst
