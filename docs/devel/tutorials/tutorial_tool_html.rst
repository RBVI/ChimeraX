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

:orphan:

.. include:: references.rst


======================================
Bundle Example: Add an HTML-based Tool
======================================

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
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_tool_html>`_
containing a folder named `tut_tool_html`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_tool_html`` folder are:

- ``tut_tool_html`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``tool.py`` - source code to implement the ``Tutorial (HTML)`` tool
        - ``docs/users/commands/tutorial.html`` - help file describing
          the graphical tool

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

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_html/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,19-24,35-36,40-43,51-52

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 19-24).  Three other changes are needed
for this bundle to declare that:

#. this bundle depends on the ``ChimeraX-UI`` and
   ``ChimeraX-Tutorial_Command`` bundles (lines 35-36),
#. non-Python files need to be included in the bundle (lines 40-43), and
#. a single graphical interface tool is provided in this bundle (lines 51-52).

The ``Dependency`` tags on lines 35 and 36 inform ChimeraX that the
``ChimeraX-UI`` and ``ChimeraX-Tutorial_Command`` bundles must be
present when this bundle is installed.  If they are not, they are
installed first.  The ``ChimeraX-UI`` bundle is needed to provide
the :py:class:`chimerax.ui.htmltool.HtmlToolInstance` class used
for building the user interface (see `tool.py`` below) and the
``ChimeraX-Tutorial_Command`` bundle is needed to provide the ChimeraX
commands that will be used for actually performing user actions.

The ``DataFiles`` tag on lines 40-43 informs ChimeraX to include
non-Python files as part of the bundle when building.  In this case,
``tool.html`` (implicitly in the ``src`` folder) which provides the
HTML component of our interface should be included.  Also, the
help documentation for our tool, ``tutorial.html``.

The ``ChimeraXClassifier`` tag on lines 51-52 informs ChimeraX that
there is one graphical interface *tool* named ``Tutorial (HTML)`` in
the bundle.  The last two fields (separated by ``::``) are the tool
category and the tool description.  ChimeraX will add a
``Tutorial (HTML)`` menu entry in its ``Tool`` submenu that matches
the tool category, ``General``; if the submenu does not exist,
it will be created.


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

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_html/src/__init__.py
    :language: python
    :linenos:

In this example, the :py:meth:`~chimerax.core.toolshed.BundleAPI.start_tool`
method is overridden to invoke a bundle function, ``tool.TutorialTool``,
when the user selects the ``Tutorial (HTML)`` menu item from the
``General`` submenu of the ``Tools`` menu.  (The ``Tutorial (HTML)``
and ``General`` names are from the ``ChimeraXClassifier`` tag
in ``bundle_info.xml`` as described above.)

The arguments to :py:meth:`~chimerax.core.toolshed.BundleAPI.start_tool`,
in bundle API version 1,
are ``session``, a :py:class:`chimerax.core.session.Session` instance,
``bi``, a :py:class:`chimerax.core.toolshed.BundleInfo` instance, and
``ti``, a :py:class:`chimerax.core.toolshed.ToolInfo` instance.
``session`` is used to access other available data such as
open models, running tasks and the logger for displaying messages,
warnings and errors.  ``bi`` contains the bundle information and
is not used in this example.  ``ti`` contains the tool information;
in this case, it is used to make sure the name of the tool being
invoked is the expected one.  If it is, ``tool.TutorialTool`` is
called; if not, an exception is thrown, which ChimeraX will turn
into an error message displayed to the user.


``src/tool.py``
---------------

``tool.py`` defines the ``TutorialTool`` class that is invoked
by ChimeraX (via the :py:meth:`~chimerax.core.toolshed.BundleAPI.start_tool`
method of ``bundle_api`` in ``__init__.py``) when the user selects the
``Tutorial (HTML)`` menu item from the ``Tools`` menu.

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_html/src/tool.py
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
accessible using the :py:attr:`html_view` attribute, an instance of
:py:class:`chimerax.ui.widgets.htmlview.HtmlView`.  In this example, the
``TutorialTool`` constructor calls its superclass constructor
and then its own ``_build_ui`` method, which simply constructs
the URL to a static HTML file in the bundle Python package and
displays it in the widget using
``self.html_view's``
:py:meth:`~chimerax.ui.widgets.htmlview.HtmlView.setUrl` method.

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
:py:attr:`CUSTOM_SCHEME` and a method,
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.handle_scheme`.
If both are defined, then the base class will arrange for
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.handle_scheme`.
to be called (in the main thread) whenever
a link matching :py:attr:`CUSTOM_SCHEME` is followed. 
In this example, the custom scheme is ``tutorial``
(line 31), so when the user clicks on links such as
``tutorial:cofm`` and ``tutorial:highlight`` (see ``tool.html`` below),
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.handle_scheme`.
is called with the clicked URL as its lone argument.
Currently, the argument is an instance
of :py:class:`PyQt5.QtCore.QUrl` but that may change later to remove
explicit dependency on PyQt.
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.handle_scheme`.
is expected to parse the URL and take appropriate action depending on
the data.  In this example, the `URL`_ *path* is a command
name and the *query* contains data for command arguments.
Three command names are supported:
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`,
``cofm``, and ``highlight``.
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`
is invoked when the page is loaded (see ``tool.html`` below)
and is handled as special case (see below).
For the other commands, known query fields are ``target``,
``model``, ``color``, ``count``, ``weighted`` and ``transformed``.
The command names and query fields are combined to generate
a ChimeraX command string, which is then executed using
:py:func:`chimerax.core.commands.run.run`.  The main benefit of executing
a command string is automatic display of command and replies
in the ChimeraX log.

The :py:class:`~chimerax.ui.htmltool.HtmlToolInstance` class also
helps with monitoring the opening and closing of models.
If the derived class defines a method named
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`,
the method will be called whenever a new model is opened or
an existing model is closed.
Note that this is *not* when a model instance is *created*
or *deleted*, because transient models that are not shown to
the user (opened) do not trigger calls to
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`.
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`
is typically called with two arguments:
the name of the triggering event (either "add models" or
"remove models") and the list of models added or removed.
In this example,
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`
is used for updating
the HTML drop-down list of models, so only the currently
opened models are important, and neither the trigger
name nor the models added or removed is relevant.
In fact, its arguments are given default values so that
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`
can be called with no arguments when
the HTML page is first loaded.  Whether called in response
to model addition/removal or HTML events,
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`
does the following:

#. build a list of 2-tuples of (*display text*, *atom_specifier*),
   one for each open model.
#. convert the list into HTML strings of ``option`` elements.
#. concatenate them into a single HTML text string.
#. set a string to "true" or "false" depending on whether there are
   any models open.
#. combine the HTML text string and the boolean string with a
   JavaScript template to generate a JavaScript script.
#. execute the JavaScript script in the HTML widget using
   ``self.html_view's``
   :py:meth:`~chimerax.ui.widgets.htmlview.HtmlView.runJavaScript` method.

Note the conversion from Python string to JavaScript string is
accomplished using :py:func:`json.dumps`, which properly handles special
characters such as quotes.  The JavaScript template uses standard
`JavaScript HTML DOM`_ functionality to manipulate the HTML page
contents.  If executing JavaScript results in errors, the messages
should appear in the ChimeraX log.


``src/tool.html``
-----------------

``tool.html`` is an `HTML 5`_ file containing the skeleton of
the graphical user interface, consisting of a form with multiple
elements such as check boxes for boolean options and radio
buttons for multiple-choice options.  Even more exotic inputs
like color selection or date and time are supported in
`HTML 5 forms`_.


.. literalinclude:: ../../../src/examples/tutorials/tut_tool_html/src/tool.html
    :language: html
    :linenos:


The ``name`` attributes in the HTML form elements correspond
to the query field names, and are exactly the same set
of query field names expected by
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.handle_scheme`
in ``tool.py``.

The ``select`` element is the drop-down list that is modified when
:py:meth:`~chimerax.ui.htmltool.HtmlToolInstance.update_models`
runs its generated JavaScript script.
To make the element easier to find, it not only has a ``name``
attribute, which does not have to be unique among all elements,
but also an ``id`` attribute, which is (or should be) unique.
The JavaScript ``getElementById`` function returns a single element,
whereas ``getElementsByName`` function returns a list of elements.

The two ``submit`` buttons are tagged with class name ``submit``
so that they can be found using ``getElementsByClassName``.
The buttons are enabled or disabled in the same JavaScript
script that updates the drop-down list of models.


``src/docs/user/commands/tutorial.html``
----------------------------------------

The documentation for the graphical tool should be written
in `HTML 5`_ and saved in a file with a suffix of ``.html``.
For our example, we named the help file ``tutorial.html``.
The location of the help file (relative to ``src/docs``) is expicitly
indicated by setting the :py:attr:`help` attribute of the
:py:class:`~chimerax.ui.htmltool.HtmlToolInstance`,
as shown on line 31 of ``tool.py``:

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_html/src/tool.py
    :language: python
    :linenos:
    :lines: 28-34
    :lineno-start: 28
    :emphasize-lines: 4

When help files are included in bundles, documentation for
the tools may be displayed using the **Help** entry of the
tool's context menu, the same as built-in ChimeraX tools.
The directory structure is chosen to allow for multiple types
of documentation for a bundle.
For example, developer documentation such as
the bundle API are saved in a ``devel`` directory instead of
``user``; documentation for typed commands are saved in
``user/commands`` instead of ``user/tools``.

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_html/src/docs/user/tools/tutorial.html
    :language: html
    :linenos:

While the only requirement for documentation is that it be written
as HTML, it is recommended that developers write tool help files
following the above template, with:

- a banner linking to the documentation index,
- text describing the tool, and
- an address for contacting the bundle author.

Note that the target links used in the HTML file are all relative
to ``..``.
Even though the tool documentation HTML file is stored with the
bundle, ChimeraX treats the links as if the file were located in
the ``tools`` directory in the developer documentation tree.
This creates a virtual HTML documentation tree where tool HTML
files can reference each other without having to be collected
together.

Optional: Session Saving
------------------------
The current session behavior of our example tool (disappearing/closing)
may be fine for some tools, particularly simpler ones.  However, some tools
may prefer to either stay in existence across a session restore, or
to save state in sessions and restore appropriately.

The Models tool is an example of the former behavior.  It does not
close when a session restores, but instead simply displays the
new model information.  It saves no state in the session.  To achieve
this behavior, it just sets the :py:class:`chimerax.core.tools.ToolInstance`
class attribute :py:attr:`SESSION_ENDURING` to True.  In the above example, changing the
:py:attr:`SESSION_ENDURING` :py:class:`~chimerax.ui.htmltool.HtmlToolInstance` class
attribute would have the same effect, since :py:class:`~chimerax.ui.htmltool.HtmlToolInstance`
inherits from :py:class:`~chimerax.core.tools.ToolInstance`.

To achieve the latter behavior, you would instead change the :py:attr:`SESSION_SAVE`
class variable to ``True``, and in addition you would implement a couple of
additional methods in the ``TutorialTool`` class and one in the ``_MyAPI`` class.
Before we get to the details of that, it would be good to go over how the
ChimeraX session-saving mechanism works, so you can have a better
understanding of how these new methods are used and should be implemented...

    When a session is saved, ChimeraX looks through the session object for
    attributes that inherit from :py:class:`chimerax.core.state.StateManager`.
    For such attributes it calls their :py:meth:`~chimerax.core.state.StateManager.take_snapshot`
    method and stows the result.  One of the state managers in the session is the tool manager.
    The tool manager will in turn call :py:meth:`~chimerax.core.state.State.take_snapshot`
    on all running tools
    that inherit from :py:class:`chimerax.core.state.State`.  (which should be
    all of them since :py:class:`~chimerax.core.tools.ToolInstance` inherits
    from :py:class:`~chimerax.core.state.State`) and stow the result.  On restore,
    the class static method :py:meth:`~chimerax.core.state.State.restore_snapshot`
    is called with the data that :py:meth:`~chimerax.core.state.State.take_snapshot`
    produced, and :py:meth:`~chimerax.core.state.State.restore_snapshot` needs to
    return a restored object.

    In practice, :py:meth:`~chimerax.core.state.State.take_snapshot` typically returns
    a dictionary with descriptive
    key names and associated values of various information that would be needed
    during restore.  Frequently one of the keys is ‘version’ so that restore_snapshot
    can do the right thing if the format of various session data items changes.
    The values can be regular Python data (including numpy/tinyarray) or class instances
    that themselves inherit from :py:class:`~chimerax.core.state.State`.

    ``restore_snapshot(session, data)`` uses ``data`` to instantiate an object of that
    class and return it.  If it is difficult to form the constructor arguments for the
    class from the session data, or to completely set the object state via those arguments
    then you will have to use “two pass” initialization, where you call the constructor
    in a way that indicates that it is being restored from a session (e.g. passing ``None``
    to an otherwise mandatory argument) and then calling some method (frequently called
    :py:meth:`set_state_from_snapshot`) to fully initialize the minimally initialized object.

    Session restore knows what bundles various classes came from, but not how to get
    those classes from the bundle so therefore the bundle’s :py:class:`~chimerax.core.toolshed.BundleAPI`
    object needs to implement it’s ``get_class(class_name)`` static method to return the
    class object that corresponds to a string containing the class name.

Now, the ``TutorialTool`` class doesn't really have any state that would need to be
saved into a session.  For the purpose of example, let's suppose that the tool's behavior
somehow depended on the last command it had issued, and that command was saved in an attribute
of ``TutorialTool`` named :py:attr:`prev_command`.  

To save/restore the tool and its :py:attr:`prev_command` attribute, we add
:py:meth:`~chimerax.core.state.State.take_snapshot` and
:py:meth:`~chimerax.core.state.State.restore_snapshot`
methods to the ``TutorialTool`` class, as per the below:

.. literalinclude:: tool_snippet.py
    :language: python

Finally, for the session-restore code to be able to find the ``TutorialTool`` class, we must
implement the :py:meth:`~chimerax.core.toolshed.BundleAPI.get_class` static method in
our ``_MyAPI`` class, like so:

.. literalinclude:: init_snippet.py
    :language: python


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
