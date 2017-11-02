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

.. _ChimeraX Toolshed: https://cxtoolshed.rbvi.ucsf.edu/
.. _Python wheel: https://wheel.readthedocs.org/
.. _Python package: https://docs.python.org/3/tutorial/modules.html#packages
.. _eXtensible Markup Language: https://en.wikipedia.org/wiki/XML
.. _Python package setup scripts: https://docs.python.org/3/distutils/setupscript.html

.. _Bundle Example\: Add a Command:


==============================
Bundle Example: Add a Command
==============================

This tutorial builds on the material from :doc:`tutorial_hello`.


Source Code Organization
========================

The source code for this example may be downloaded
as a `zip-format file
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_cmd>`_
containing a folder named `tut_cmd`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_cmd`` folder are:

``tut_cmd`` - bundle folder
    ``bundle_info.xml`` - bundle information read by ChimeraX
    ``src`` - source code to Python package for bundle
        ``__init__.py`` - package initializer and interface to ChimeraX
        ``cmd.py`` - source code to implement ``hello`` command

The file contents are shown below.


``bundle_info.xml``
-------------------

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-25,41-45

For explanations of the unhighlighted sections, please
see :doc:`tutorial_hello`.


``src``
-------

``src`` is the folder containing the source code for the
Python package that implements the bundle functionality.
The ChimeraX ``devel`` command automatically includes all
``.py`` files in ``src`` as part of the bundle.  (Additional
files may also be included using bundle information tags
such as ``DataFiles`` as shown in :doc:`tutorials_tool`.)
The only required file in ``src`` is ``__init__.py``.
Other ``.py`` files are typically arranged to implement
different types of functionality.  For example, ``cmd.py``
is used for command-line commands; ``tool.py`` or ``gui.py``
for graphical interfaces; ``io.py`` for reading and saving
files, etc.


``__init__.py``
---------------

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/src/__init__.py
    :language: python
    :linenos:


``cmd.py``
----------

.. literalinclude:: ../../../src/examples/tutorials/tut_cmd/src/cmd.py
    :language: python
    :linenos:


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello` (previous topic)
- :doc:`tutorial_command` (current topic)
- :doc:`tutorial_tool` (next topic)
- :doc:`tutorial_read_format`
- :doc:`tutorial_save_format`
- :doc:`tutorial_fetch`
- :doc:`tutorial_selector`
