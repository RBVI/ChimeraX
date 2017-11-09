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

.. include:: reference.rst

.. _Bundle Example\: Add a Tool:


===========================
Bundle Example: Add a Tool
===========================

This tutorial builds on the material from :doc:`tutorial_command`.

This example describes how to create a ChimeraX bundle
that defines a graphical interface to the two commands,
``tutorial cofm`` and ``tutorial highlight``, defined
in the :doc:`tutorial_command` example.

The ChimeraX user interface is built using `PyQt5`_,
which has a significant learning curve.  However, PyQt5
has very good support for displaying `HTML 5`_ with
`JavaScript`_ in a window, which provides a simpler
avenue for implementing graphical interfaces.  This example
shows how to combine a static HTML page with dynamically
generated JavaScript to create an interface with only
a small amount of code.

The steps in implementing the bundle are:

#. Create a ``bundle_info.xml`` containing information
   about the bundle,
#. Create a Python package that interfaces with ChimeraX
   and implements the command functionality, and
#. Install and test the bundle in ChimeraX.

The final step builds a Python wheel that ChimeraX uses
to install the bundle.  So if the bundle passes testing,
it is immediately available for sharing with other users.


Source Code Organization
========================

The source code for this example may be downloaded
as a `zip-format file
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_gui>`_
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

``bundle_info.xml`` is an `_eXtensible Markup Language`_
format file whose tags are listed in :doc:`bundle_info`.
While there are many tags defined, only a few are needed
for bundles written completely in Python.  The
``bundle_info.xml`` in this example is similar to the one
from the :doc:`tutorial_command` example with changes highlighted.
For explanations of the unhighlighted sections, please
see :doc:`tutorial_hello` and :doc:`tutorial_command`.

.. literalinclude:: ../../../src/examples/tutorials/tut_gui/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,17-25,35,38-41,48-50

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 17-25).  Three other changes are needed
for this bundle to declare that:

#. this bundle depends on the ``ChimeraX-Tutorial_Command`` bundle (line 35),
#. non-Python files need to be included in the bundle (lines 38-41), and
#. a single graphical interface tool is provided in this bundle (lines 48-50).

The ``Dependency`` tag on line 35 informs ChimeraX that the
``ChimeraX-Tutorial_Command`` bundle must be present when this bundle
is installed.  If it is not, it is installed first.

The ``DataFiles`` tag on lines 38-41 informs ChimeraX to include
non-Python files as part of the bundle when building.  In this case,
``gui.html`` (implicitly in the ``src`` folder) should be included.

The ``ChimeraXClassifier`` tag on lines 49-50 informs ChimeraX that
there is one graphical interface *tool* named ``Tutorial GUI`` in
the bundle.  The last two fields (separated by ``::``) are the tool
category and the tool description.  ChimeraX will add a
``Tutorial GUI`` menu entry in its ``Tool`` submenu that matches
the tool category, ``General``; if the submenu does not exist,
it will be created.


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

The tool registration code is 

As described in :doc:`tutorial_hello`, ``__init__.py`` contains
the initialization code that defines the ``bundle_api`` object
that ChimeraX needs in order to invoke bundle functionality.
ChimeraX expects ``bundle_api`` class to be derived from
:py:class:`chimerax.core.toolshed.BundleAPI` with methods
overridden for registering commands, tools, etc.

.. literalinclude:: ../../../src/examples/tutorials/tut_gui/src/__init__.py
    :language: python
    :linenos:

In this example, the ``start_tool`` method is overridden to
invoke a bundle function, ``gui.TutorialGUI``, when the user
selects the ``Tutorial GUI`` menu item from the ``General``
submenu of the ``Tools`` menu.  (The ``Tutorial GUI`` and
``General`` names are from the ``ChimeraXClassifier`` tag
in ``bundle_info.xml`` as described above.)

The arguments to ``start_tool``, in bundle API version 1,
are ``session``, a ``chimerax.core.session.Session`` instance,
``bi``, a ``chimerax.core.toolshed.BundleInfo`` instance, and
``ti``, a ``chimerax.core.toolshed.ToolInfo`` instance.
``session`` is used to access other available data such as
open models, running tasks and the logger for displaying messages,
warnings and errors.  ``bi`` contains the bundle information and
is not used in this example.  ``ti`` contains the tool information;
in this case, it is used to make sure the name of the tool being
invoked is the expected one.  If it is, ``gui.TutorialGUI`` is
called; if not, an exception is thrown, which ChimeraX will turn
into an error message displayed to the user.


``gui.py``
----------

``gui.py`` defines the ``TutorialGUI`` class that is invoked
by ChimeraX (via the ``start_tool`` method of ``bundle_api``
in ``__init__.py``) when the user selects the ``Tutorial GUI``
menu item from the ``Tools`` menu.

.. literalinclude:: ../../../src/examples/tutorials/tut_gui/src/gui.py
    :language: python
    :linenos:


``gui.html``
----------

.. literalinclude:: ../../../src/examples/tutorials/tut_gui/src/gui.html
    :language: html
    :linenos:


.. include:: build_test_distribute.rst

What's Next
===========

- :doc:`tutorial_hello`
- :doc:`tutorial_command` (previous topic)
- :doc:`tutorial_tool` (current topic)
- :doc:`tutorial_read_format` (next topic)
- :doc:`tutorial_save_format`
- :doc:`tutorial_fetch`
- :doc:`tutorial_selector`
