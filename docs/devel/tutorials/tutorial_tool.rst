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

- ``tut_gui`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``gui.py`` - source code to implement ``Tutorial GUI`` tool

The file contents are shown below.


``bundle_info.xml``
-------------------

``bundle_info.xml`` is an `eXtensible Markup Language`_
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

.. include:: src.rst


``__init__.py``
---------------

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
are ``session``, a :py:class:`chimerax.core.session.Session` instance,
``bi``, a :py:class:`chimerax.core.toolshed.BundleInfo` instance, and
``ti``, a :py:class:`chimerax.core.toolshed.ToolInfo` instance.
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

:py:class:`chimerax.ui.htmltool.HtmlToolInstance` is the base class for
simplifying construction of tools with HTML-based graphical
interface.  When an instance of a subclass of
:py:class:`~chimerax.ui.htmltool.HtmlToolInstance`
is created, its constructor must call the
:py:class:`~chimerax.ui.htmltool.HtmlToolInstance`
constructor to set up the graphical interface framework.
The arguments to the
:py:class:`~chimerax.ui.htmltool.HtmlToolInstance`
constructor is the
session and the tool name.  An optional argument, ``size_hint``,
may be supplied to guide the tool layout, but, as the name suggests,
it is only a hint and may not be honored.
The superclass constructor creates a ChimeraX tool which contains
a single widget for displaying an HTML page.  The widget is
accessible using the ``html_view`` attribute, an instance of
:py:class:`chimerax.ui.widgets.htmlview.HtmlView`.  In this example, the
``TutorialGUI`` constructor calls its superclass constructor
and then its own ``_build_ui`` method, which simply constructs
the URL to a static HTML file in the bundle Python package and
displays it in the widget using ``self.html_view.setUrl``.

The :py:class:`~chimerax.ui.htmltool.HtmlToolInstance`
class also helps manage threading
issues that arise from the way HTML is displayed using `PyQt5`_.
The underlying `Qt WebEngine`_ machinery uses a separate thread
for rendering HTML, so developers need to make sure that code
is run in the proper thread.  In particular, access to shared
data must be synchronized between the Qt main and WebEngine
threads.
:py:class:`~chimerax.ui.htmltool.HtmlToolInstance`
simplifies the issues by calling
subclass methods in the main thread when an interesting event
occurs in the WebEngine thread.

The :py:class:`~chimerax.ui.htmltool.HtmlToolInstance` constructor
checks the derived class for the presence of an attribute,
:py:attr:`CUSTOM_SCHEME` and a method, :py:meth:`handle_scheme`.
If both are defined, then the base class will arrange for
py:meth:`handle_scheme` to be called (in the main thread) whenever
a link matching :py:attr:`CUSTOM_SCHEME` is followed. 
In this example, the custom scheme is ``tutorial``
(line 31), so when the user clicks on links such as
``tutorial:cofm`` and ``tutorial:highlight`` (see ``gui.html``
below), :py:meth:`handle_scheme` is called with the clicked URL as
its lone argument.  Currently, the argument is an instance
of :py:class:`PyQt5.QtCore.QUrl` but that may change later to remove
explicit dependency on PyQt.  :py:meth:`handle_scheme` is expected
to parse the URL and take appropriate action depending on
the data.  In this example, the `URL`_ *path* is a command
name and the *query* contains data for command arguments.
Three command names are supported: :py:meth:`update_models`, ``cofm``,
and ``highlight``.  :py:meth:`update_models` is invoked when the page
is loaded (see ``gui.html`` below) and is handled as special case
(see below).
For the other commands, known query fields are ``target``,
``model``, ``color``, ``count``, ``weighted`` and ``transformed``.
The command names and query fields are combined to generate
a ChimeraX command string, which is then executed using
:py:func:`chimerax.core.commands.run.run`.  The main benefit of executing
a command string is automatic display of command and replies
in the ChimeraX log.

The :py:class:`~chimerax.ui.htmltool.HtmlToolInstance` class also
helps monitoring the opening and closing of models.
If the derived class defines a method named :py:meth:`update_models`,
the method will be called whenever a new models is opened or
an existing model is closed.
Note that this is *not* when a model instance is *created*
or *deleted*, because transient models that are not shown to
the user (opened) do not trigger calls to :py:meth:`update_models`.
:py:meth:`update_models` is typically called with two arguments:
the name of the triggering event (either "add models" or
"remove models") and the list of models added or removed.
In this example, :py:meth:`update_models` is used for updating
the HTML drop-down list of models, so only the currently
opened models are important, and neither the trigger
name nor the models added or removed is relevant.
In fact, its arguments are given default values so that
:py:meth:`update_models` can be called with no arguments when
the HTML page is first loaded.  Whether called in response
to model addition/removal or HTML events, :py:meth:`update_models`
does the following:

#. build a list of 2-tuples of (*display text*, *atom_specifier*),
   one for each open model.
#. convert the list into HTML strings of ``option`` elements.
#. concatenated into a single HTML text string.
#. set a string to "true" or "false" depending on whether there are
   any models open.
#. combine the HTML text string and the boolean string with a
   JavaScript template to generate a JavaScript script.
#. execute the JavaScript script in the HTML widget using
   :py:meth:`self.html_view.runJavaScript`.

Note the conversion from Python string to JavaScript string is
accomplished using :py:func:`json.dumps`, which properly handles special
characters such as quotes.  The JavaScript template uses standard
`JavaScript HTML DOM`_ functionality to manipulate the HTML page
contents.  If executing JavaScript results in errors, the messages
should appear in the ChimeraX log.


``gui.html``
------------

``gui.html`` is an `HTML 5`_ file containing the skeleton of
the graphical user interface, consisting of a form with multiple
elements such as check boxes for boolean options and radio
buttons for multiple-choice options.  Even more exotic inputs
like color selection or date and time are supported in
`HTML 5 forms`_.


.. literalinclude:: ../../../src/examples/tutorials/tut_gui/src/gui.html
    :language: html
    :linenos:


The ``name`` attributes in the HTML form elements correspond
to the query field names, and are exactly the same set
of query field names expected by ``handle_scheme`` in
``gui.py``.

The ``select`` element is the drop-down list that is modified when
``update_models`` runs its generated JavaScript script.
To make the element easier to find, it not only has a ``name``
attribute, which does not have to be unique among all elements,
but also an ``id`` attribute, which is (or should be) unique.
The JavaScript ``getElementById`` function returns a single element,
whereas ``getElementsByName`` function returns a list of elements.

The two ``submit`` buttons are tagged with class name ``submit``
so that they can be found using ``getElementsByClassName``.
The buttons are enabled or disabled in the same JavaScript
script that updates the drop-down list of models.


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
