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
Bundle Example: Add a Qt-based Tool
======================================

This example describes how to create a ChimeraX bundle
that defines a graphical interface showing a text-input
field that logs text typed by the user via the appropriate
`log command <../../user/commands/log.html>`_.

The ChimeraX user interface is built using `PyQt5`_,
which is a Python wrapping of the `Qt5`_ C++ windowing toolkit.
Bundle writers can themselves use `PyQt5`_ to provide
a graphical interface to their bundle functionality.
This example shows how to build a simple graphical interface,
and is not meant to cover all the capabilities of `Qt`_
in detail (and there are many!).  To learn more you should
explore `PyQt5 tutorials`_
and/or look at the code of other tools that do things
similar to what you want your tool to do.

The steps in implementing the bundle are:

#. Create a ``bundle_info.xml`` file containing information
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
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/bundle_tutorial.zip?name=tut_tool_qt>`_
containing a folder named ``tut_tool_qt``.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``tut_tool_qt`` folder are:

- ``tut_tool_qt`` - bundle folder
    - ``bundle_info.xml`` - bundle information read by ChimeraX
    - ``src`` - source code to Python package for bundle
        - ``__init__.py`` - package initializer and interface to ChimeraX
        - ``tool.py`` - source code to implement the ``Tutorial (Qt)`` tool
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

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/bundle_info.xml
    :language: xml
    :linenos:
    :emphasize-lines: 8-10,19-23,34,38-40,48-49

The ``BundleInfo``, ``Synopsis`` and ``Description`` tags are
changed to reflect the new bundle name and documentation
(lines 8-10 and 19-23).  Three other changes are needed
for this bundle to declare that:

#. this bundle depends on the ``ChimeraX-UI`` bundle (line 34),
#. non-Python files need to be included in the bundle (lines 38-40), and
#. a single graphical interface tool is provided in this bundle (lines 48-49).

The ``Dependency`` tag on line 34 informs ChimeraX that the
``ChimeraX-UI`` bundle must be present when this bundle is installed.
If it is not, it is installed first.  The ``ChimeraX-UI`` bundle is
needed for the :py:class:`chimerax.ui.gui.MainToolWindow` class
that provides the tool window that will contain our interface elements
(see `tool.py`` below).

The ``DataFiles`` tag on lines 38-40 informs ChimeraX to include
non-Python files as part of the bundle when building.  In this case,
``docs/user/tools/tutorial.html`` (implicitly in the ``src`` folder)
which provides the help documentation for our tool.

The ``ChimeraXClassifier`` tag on lines 48-49 informs ChimeraX that
there is one graphical interface tool named ``Tutorial (Qt)`` in
the bundle.  The last two fields (separated by ``::``) are the tool
category and the tool description.  ChimeraX will add a
``Tutorial (Qt)`` menu entry in its ``Tool`` submenu that matches
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

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/__init__.py
    :language: python
    :linenos:

In this example, the :py:meth:`~chimerax.core.toolshed.BundleAPI.start_tool`
method is overridden to invoke a bundle function, :py:meth:`tool.TutorialTool`,
when the user selects the ``Tutorial (Qt)`` menu item from the ``General``
submenu of the ``Tools`` menu.  (The ``Tutorial (Qt)`` and
``General`` names are from the ``ChimeraXClassifier`` tag
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

The :py:meth:`~chimerax.core.toolshed.BundleAPI.get_class` method
is used by the ChimeraX session-saving mechanism to find needed
class objects in the bundle and is discussed in more detail in
the :ref:`session-saving` section below.


``src/tool.py``
---------------

``tool.py`` defines the ``TutorialTool`` class that is invoked
by ChimeraX (via the :py:meth:`~chimerax.core.toolshed.BundleAPI.start_tool`
method of ``bundle_api`` in ``__init__.py``) when the user selects the
``Tutorial (Qt)`` menu item from the ``Tools`` menu.  We will discuss
``tool.py`` in sections.

Class Initialization
++++++++++++++++++++


.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :end-before: __init__

Our ``TutorialTool`` class inherits from :py:class:`chimerax.core.tools.ToolInstance`,
which makes it known to the ChimeraX tool manager, and
it will thereby work correctly in all the generic
ways that tools work, such as being displayed by the
command `tool show <../../user/commands/ui.html>`_.

By declaring :py:attr:`SESSION_ENDURING` as ``False``, we
are telling ChimeraX's session handling that this tool
should be closed/destroyed when a session is closed.
:py:attr:`SESSION_SAVE` = ``True`` tells session handling
that this tool will save state into sessions and should be
restored by sessions.  This is discussed further in the
:ref:`session-saving` section below.
Lastly, setting the class variable :py:attr:`help` informs
the ChimeraX help system where the help documentation for
this tool can be found, and is discussed in more detail in
the :ref:`help` section.

Instance Initialization
+++++++++++++++++++++++


.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :lines: 31-
    :lineno-start: 31
    :end-before: def _build_ui

Our ``TutorialTool`` class constructor is called with ``session`` and
``tool_name`` arguments, because that is how we called it from the
:py:meth:`~chimerax.core.toolshed.BundleAPI.start_tool` method of our
:py:class:`_MyAPI` class:

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/__init__.py
    :language: python
    :lines: 29-30

On line 36, we call our superclass (:py:class:`chimerax.core.tools.ToolInstance`)
constructor.  It also takes a session and tool name as arguments, which is one of
the principal reasons we passed those arguments to our own constructor.
The :py:class:`chimerax.core.tools.ToolInstance` constructor
sets its :py:attr:`session` attribute to be the same as the passed-in ``session``,
so in other parts of our code we can refer to the session with :py:attr:`self.session`.

On lines 50 and 51, we create our :py:class:`~chimerax.ui.MainToolWindow` instance,
which will contain our user interface.  The window will not actually be shown until
we call its :py:meth:`~chimerax.ui.MainToolWindow.manage` method, as discussed in
the following :ref:`interface` section.

Every tool in ChimeraX has a context menu, which will at least contain some generically
useful tool actions (*e.g.* **Hide Tool**).  To add additional tool-specific items to
the context menu, we must override :py:class:`~chimerax.ui.MainToolWindow`'s
:py:meth:`fill_context_menu` method (by default a no-op) with our own routine to add
our custom menu items, as discussed in more detail in the :ref:`context-menu` section.
On line 55 we override that default :py:meth:`fill_context_menu` with :py:meth:`self.fill_context_menu`.

Lastly, on line 61 we call a routine to fill out our user interface, discussed in the
next section.

.. _interface:

Interface Construction
++++++++++++++++++++++

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :lines: 63-
    :lineno-start: 63
    :end-before: def fill_context_menu

The :py:meth:`_build_ui` method adds our user interface widgets to the tool window
and causes the tool window to be shown.  `PyQt5`_ is the windowing toolkit used by ChimeraX.
It is a Python wrapping of the (C++) `Qt5`_ toolkit. This tutorial is in no way meant
to also be a `PyQt5`_/`Qt5`_ tutorial (since those toolkits are *very* extensive) but
merely shows how to use those toolkits in the context of ChimeraX.
To gain additional familarity with those toolkits, there are
`PyQt5 tutorials`_ and
`Qt5 tutorials <http://zetcode.com/gui/qt5/>`_ available on the web.

On line 69 we import the widgets will need for our interface from the `PyQt5`_ toolkit:

* A text-label widget (`QLabel <http://doc.qt.io/qt-5/qlabel.html>`_)
* An editable single-line text entry field (`QLineEdit <http://doc.qt.io/qt-5/qlineedit.html>`_)
* A "metawidget" for laying out the above two widgets side by side (`QHBoxLayout <http://doc.qt.io/qt-5/qhboxlayout.html>`_; "HBox" == "horizontal box")

Line 70 creates our horizontal layout metawidget, and line 71 creates and adds
the label we want next to our entry field to it.  Note that by default widgets
added to an `QHBoxLayout <http://doc.qt.io/qt-5/qhboxlayout.html>`_ will be ordered left to right.
Line 72 creates our text-entry field and line 77 adds it to out layout.

Changes in widgets that the containing interface may care about cause the widget to
emit what `Qt`_ refers to as a "signal".
`returnPressed <http://doc.qt.io/qt-5/qlineedit.html#returnPressed>`_ is the signal that
`QLineEdit <http://doc.qt.io/qt-5/qlineedit.html>`_ emits when the users presses the Return key.
A signal's :py:meth:`connect` method is the way to get a particular routine to be called when
the signal is emitted, which we have done on line 76 to get our :py:meth:`return_pressed`
method called when the `returnPressed <http://doc.qt.io/qt-5/qlineedit.html#returnPressed>`_
signal is emitted.

Lines 86-90 is our handler for the `returnPressed <http://doc.qt.io/qt-5/qlineedit.html#returnPressed>`_
signal.  Some signals also have
arguments (detailed in each widget's signal documentation), but the
`returnPressed <http://doc.qt.io/qt-5/qlineedit.html#returnPressed>`_
signal has no arguments, so therefore our handler has no non-``self`` arguments.
The handler imports the :py:func:`~chimerax.core.commands.run` utility command that
runs a text string as a ChimeraX command, and then calls that routine with the session
and the appropriate `log <../../user/commands/log.html>`_ command, formed based on the
current text in the line editor (*i.e.* :py:meth:`self.line_edit.text`).

We have created both our widgets and added them to the layout.  Line 80 installs
our layout as the layout for the user-interface area of our tool window (the
user-interface area is in fact an instance of `QWidget <http://doc.qt.io/qt-5/qwidget.html>`_).

Line 84 calls our tool window's :py:meth:`manage` method to cause the tool window to be displayed.
The argument to :py:meth:`manage` specifies the general position of the tool window, with
possible values of:

`"side"`
  The user's preferred side of the main window for tools (specified in **Window** preferences)
`"left"` / `"right"` / `"top"` / `"bottom"`
  A specific side of the main window.  Normally, honoring the user's preference with `"side"`
  is preferred, but some tools may work best at the top or bottom of the main window for example.
`None`
  The window should start out "floating", not docked into the ChimeraX main window.

Some tools may use multiple windows (created via the :py:class:`~chimerax.ui.MainToolWindow`'s
:py:meth:`create_child_window` method), and for those tools another possible value for
:py:meth:`manage` is another tool window (typically the tool's main window), in which case
the tool window will start out tabbed with the other window.


.. _context-menu:

Context Menu
++++++++++++

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :lines: 92-
    :lineno-start: 92
    :end-before: def take_snapshot

ChimeraX will create a context menu for every tool, populated with
generically useful tool actions, such as **Dockable Tool** (to control
whether the tool can be docked into the main window).  To add custom
items to this context we have to override the :py:class:`~chimerax.ui.MainToolWindow`'s
default implementation of :py:meth:`fill_context_menu` (which does nothing)
with our own implementation, which we did on line 55:

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :lines: 53-55
    :lineno-start: 53
    :emphasize-lines: 3

Our overriding routine is shown on lines 92-103.  The routine is invoked with
three arguments:

``menu``
  A `QMenu <http://doc.qt.io/qt-5/qmenu.html>`_ instance that we will add our
  custom menu items to.  It is not yet populated with the generic menu items.
``x`` and ``y``
  The x and y position of the click that is bringing up the context menu,
  relative to the entire user-interface area (:py:attr:`self.toolwindow.ui_area`).
  These arguments are only used in the rare case where the contents of the
  context menu depend on exactly where in the tool the user clicked.  These
  values are the :py:meth:`x` and :py:meth:`y` methods of the
  `QContextMenuEvent <http://doc.qt.io/qt-5/qcontextmenuevent.html>`_ that
  is bringing up this menu.

`Qt`_ abstracts actions on widgets (such as button clicks and menu selections)
with its `QAction <http://doc.qt.io/qt-5/qaction.html>`_ class.  In order to
add a **Clear** item to the menu which will clear the text in the input field,
we import the `QAction <http://doc.qt.io/qt-5/qaction.html>`_ class on line 100
and create an instance of it with the text "Clear", and associated with the 
context menu, on line 101.

When the action encapsulated by a `QAction <http://doc.qt.io/qt-5/qaction.html>`_
occurs, its `triggered <http://doc.qt.io/qt-5/qaction.html#triggered>`_ signal is emitted
(in a similar fashion to the `returnPressed <http://doc.qt.io/qt-5/qlineedit.html#returnPressed>`_
signal in the :ref:`interface` section above).
We arrange for our text-input field to be cleared by connecting an anonymous lambda
function (that calls `self.line_edit.clear() <https://doc.qt.io/qt-5/qlineedit.html#clear>`_) to the
`triggered <http://doc.qt.io/qt-5/qaction.html#triggered>`_ signal, 
shown on line 102.
The `triggered <http://doc.qt.io/qt-5/qaction.html#triggered>`_
signal does provide an argument (which the lambda uses
`*args` to ignore) indicating whether the item is checked on or off.  That
isn't relevant in our case because we haven't made our menu item "checkable".
But you may want to add "checkable" menu items in some cases.  To do so,
use `QAction <http://doc.qt.io/qt-5/qaction.html>`_'s
`setCheckable <http://doc.qt.io/qt-5/qaction.html#checkable-prop>`_
method with a value of `True` to make it checkable and then set its initial
checked/unchecked state with the `setChecked <http://doc.qt.io/qt-5/qaction.html#checked-prop>`_
method, with the appropriate boolean argument.

We actually add the action/item to the menu on line 103.

.. _help:

Help Documentation
++++++++++++++++++

All tools will have a **Help** context-menu entry,
one of the "generic" context-menu items that ChimeraX
adds to all tool context menus.  The **Help** menu item
will be disabled unless the tool specifies that it 
provides help by setting the :py:attr:`help` attribute
of the :py:class:`~chimerax.core.tools.ToolInstance` instance
that it creates.  We did do this on line 28:

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :lines: 26-31
    :lineno-start: 26
    :emphasize-lines: 3

The string we set the :py:attr:`help` attribute to is an URL.
The "help:" prefix tells ChimeraX to use its built-in help
system to locate the help page.  It could instead have been "http:"
to have the help page found on web, but this is typically not
recommended since it is best to have the help documentation
match the actual installed version of the tool, and also to
allow help to be accessed even if the user doesn't currently
have Internet connectiity.

The remainder of the string after "help:" is the actual location
of the help page, relative to the package's ``src/docs`` folder.
The directory structure is chosen to allow for multiple types
of documentation for a bundle.
For example, developer documentation such as
the bundle API are saved in a ``devel`` directory instead of
``user``; documentation for typed commands are saved in
``user/commands`` instead of ``user/tools``.

As for the actual contents of the help file...

``src/docs/user/tools/tutorial.html``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/docs/user/tools/tutorial.html
    :language: html

The documentation for the graphical tool should be written
in `HTML 5`_ and saved in a file with a suffix of ``.html``.
For our example, we named the help file ``tutorial.html``.

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

.. _session-saving:

Sessions
++++++++

As mentioned briefly earlier, the behavior of our tool when sessions
are closed, saved, or restored is control by the boolean attributes
:py:attr:`SESSION_ENDURING` and :py:attr:`SESSION_SAVE`, which we
set on lines 26 and 27:

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :lines: 25-28
    :lineno-start: 25
    :emphasize-lines: 2-3

Tools that set :py:attr:`SESSION_ENDURING` to `True` will not be closed
when a session is closed (restoring a session implicitly closes the
existing session).  This behavior can be appropriate for widely used
tools with no particular state to save — such as the Model Panel, which
treats the models closed and opened by the session restore in the same
fashion as other model closures and openings.  Our tool *does* save
state (the current text of the input field), so we set :py:attr:`SESSION_ENDURING` to `False`.

Tools that set :py:attr:`SESSION_SAVE` to `True` will have their state
saved in sessions and need to implement a couple of
additional methods in the :py:class:`~chimerax.core.tools.ToolInstance`
class and one in the :py:class:`~chimerax.core.toolshed.BundleAPI` class.
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

Our implementation of the :py:meth:`~chimerax.core.state.State.take_snapshot`
and :py:meth:`~chimerax.core.state.State.restore_snapshot` methods are on lines 105 to 119:

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/tool.py
    :language: python
    :linenos:
    :lines: 105-
    :lineno-start: 105

The :py:meth:`~chimerax.core.state.State.take_snapshot` method forms and returns a dictionary
encapsulating the tool state.  It has two keys:

version
  An integer indicating what "version" of the state dictionary it is.  This key is not used
  currently during the restore, but if the format of the state dictionary is ever changed,
  it may be useful to use the version key to distinguish between the different formats and
  restore appropriately.
current text
  The text in the input field as the session is saved.

Note that the :py:meth:`~chimerax.core.state.State.take_snapshot` method could return any type
of data, but a dictionary is very flexible, in case additional state needs to be stored in later
versions of the tool.

The `flags` argument of :py:meth:`~chimerax.core.state.State.take_snapshot` can be ignored.
It is intended for use in the future to distinguish between snapshots saved for sessions vs. those
saved for scenes.

The :py:meth:`~chimerax.core.state.State.restore_snapshot` class method constructs an instance
of ``TutorialTool``, and then sets the text in the instance's input field to what was saved in the session,
and then returns the instance.

Note that :py:meth:`~chimerax.core.state.State.restore_snapshot` could have been coded as a
static method (and therefore would not receive a `class_obj` argument), in which case you would
have to use the actual class name in the constructor call.

Lastly, for the session-restore code to be able to find the ``TutorialTool`` class, we must
implement the :py:meth:`~chimerax.core.toolshed.BundleAPI.get_class` static method in
our ``_MyAPI`` class:

.. literalinclude:: ../../../src/examples/tutorials/tut_tool_qt/src/__init__.py
    :language: python
    :linenos:
    :lines: 33-39
    :lineno-start: 33

:py:meth:`~chimerax.core.toolshed.BundleAPI.get_class` is passed the needed class name as a string,
and finds and returns the corresponding class object.
:py:meth:`~chimerax.core.toolshed.BundleAPI.get_class` only needs to handle classes that will be
saved in sessions, not other bundle classes, and should throw an error if it gets a string that
doesn't match a class name it expects to be involved in session saving/restoring.


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
