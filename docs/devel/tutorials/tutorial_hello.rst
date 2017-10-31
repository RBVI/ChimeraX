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

.. _Bundle Example\: Hello World:


============================
Bundle Example: Hello World
============================

This example will describe how to create a ChimeraX bundle
that defines a new command, ``hello``.  The steps in
implementing the bundle are:

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
<http://www.rbvi.ucsf.edu/chimerax/cgi-bin/XXX>`_
containing a folder named `hello_world`.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the source code folder are:

``hello_world`` - bundle folder
    ``bundle_info.xml`` - bundle information read by ChimeraX
    ``src`` - source code to Python package for bundle
        ``__init__.py`` - package initializer and interface to ChimeraX
        ``cmd.py`` - source code to implement ``hello`` command

The file contents are shown below.


``bundle_info.xml``
-------------------

.. literalinclude:: ../../../src/examples/tutorials/hello_world/bundle_info.xml
    :language: xml
    :linenos:


``bundle_info.xml`` is an `_eXtensible Markup Language`_
format file whose tags are listed in :doc:`bundle_info`.
While there are many tags defined, only a few are needed
for bundles written completely in Python.

The document tag (which contains all other tags)
is named ``BundleInfo``, whose required
attributes are:

- ``name``: the name of the bundle,
- ``version``: version of the bundles, usually in the form
  of `major.minor.patch`,
- ``package``: the name of the Python package where ChimeraX
  can find the code for this bundle, and
- ``minSessionVersion`` and ``maxSessionVersion``: the minimum
  and maximum sessionf file versions that the bundle supports.

The next few tags supply information about who wrote the bundle,
where to find more information on the web, as well as short
and long descriptions of what functionality the bundle provides.

The ``Category`` tags list the categories to which the
bundle belong.  These ``Category`` values are used by the
`ChimeraX Toolshed`_ when the bundle is contributed to the
repository.  (Note that these values are completely distinct
from the *category* values described below in
``ChimeraXClassifier``.)

The ``Dependency`` tags list the bundles that must be installed
for this bundle to work.  The ``ChimeraX-Core`` bundle is a
pre-installed bundle that provides much of ChimeraX functionality.
For alpha and beta releases, the version number will start from
"0.1" and slowly approach "1.0".  Because ChimeraX Python API
follows `semantic versioning`_ rules (newer versions of ChimeraX
are compatible with older ones with the same major version number),
bundles written for earlier versions of ChimeraX will typically
work in later versions as well.  This is indicated by the ``>=``
in the ``version`` attribute of the ``Dependency`` tag for
``ChimeraX-Core``.  A ``Dependency`` tag should be present for each
additional bundle that must be installed.  During installation
for this bundle, if any of the bundles listed in ``Dependency``
tags are missing, they are automatically installed as well.

Finally, there are ``Classifier`` tags, of which there are two
flavors: Python and ChimeraX.  Values for Python classifiers
are the same as those found in standard `Python package setup
scripts`_.  Values for ``ChimeraXClassifier`` tags classifiers
follow the same form as Python classifiers, using ``::`` as
separators among data fields.
The first data field must be the string ``ChimeraX``.
The second field specifies the type of functionality supplied,
in this case, a command.
For command classifiers, the third field is the name of the
command, in this case, ``hello``.
The fourth field for command classifiers is its category,
in this case, ``General``.  (The category for a command is
reserved for future use but does not currently affect ChimeraX
behavior.)
The final data field for command classifiers is a synopsis
of what the command does, and is shown as help text in the
ChimeraX interface.

Commands may be a single word or multiple words.
The latter is useful for grouping multiple commands by
sharing the same first word.  ChimeraX also automatically
support unambiguous prefixes as abbreviations.  For example,
the user can use ``hel`` as an abbreviation for ``hello``
if no other command begins with ``hel``; however, ``h``
is not an abbreviation because the ``hide`` command also
starts with ``h``.

All bundle functionality must be listed in in ChimeraX
classifiers in order for ChimeraX to integrate them into
its user interface.  In this example, the bundle only
provides a single new command-line interface command.
Reference documentation for bundle information tags, and
specifically ChimeraX classifiers, is in :doc:`bundle_info`.


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

.. literalinclude:: ../../../src/examples/tutorials/hello_world/src/__init__.py
    :language: python
    :linenos:

``__init__.py`` contains the initialization code that defines
the ``bundle_api`` object that ChimeraX needs in order to
invoke bundle functionality.  ChimeraX expects ``bundle_api``
class to be derived from :py:class:`chimerax.core.toolshed.BundleAPI`,
which has one public attribute, ``api_version``, and these methods:

- ``start_tool`` - invoked to display a graphical interface
- ``register_command`` - invoked the first time a bundle command is used
- ``register_selector`` - invoked the first time a bundle chemical subgroup
  selector is used
- ``open_file`` - invoked when a file of a bundle-supported format is opened
- ``save_file`` - invoked when a file of a bundle-supported format is saved
- ``initialize`` - invoked when ChimeraX starts up and the bundle needs
  custom initialization
- ``finish`` - invoked when ChimeraX exits and the bundle needs custom clean up
- ``get_class`` - invoked when a session is saved and a bundle object needs
  to be serialized

The ``api_version`` attribute should be set to ``1``.  The default
value for ``api_version`` is ``0`` and is supported for older bundles.
New bundles should always use the latest supported API version.

This example only provides a single command, so the only method that
needs to be overridden is ``register_command``.  The other methods
should never be called because there are no ``ChimeraXClassifier``
tags in ``bundle_info.xml`` that mention other types of functionality.

``register_command`` is called once for each command listed in a
``ChimeraXClassifier`` tag.  When ChimeraX starts up, it registers
a placeholder for each command in all bundles, but normally does
import the bundles.  When a command is used and ChimeraX detects
that it is actually a placeholder, it asks the bundle to register
the run-time information regarding what arguments are expected and
which function should be called to process the command, after which
the command line is parsed and the registered function is called.
Once a command is registered, ChimeraX will not call
``register_command`` for it again.

In ``BundleAPI`` version 1, the ``register_command`` method is called
with three arguments:

- ``bi`` - instance of :py:class:`chimerax.core.toolshed.BundleInfo``
- ``ci`` - instance of :py:class:`chimerax.core.toolshed.CommandInfo``
- ``logger`` - instance of :py:class:`chimerax.core.logger.Logger``

``bi`` provides access to bundle information such as its name, version,
and description.  For this example, no bundle information is required
and ``bi`` is unused.  ``ci`` provides access to command information,
and the two attributes used are ``synopsis`` (for setting help text
if none is provided in code) and ``name`` (for notifying ChimeraX of
what function to use to process the command).  ``logger`` may be used
to notify users of warnings and errors; in this example, errors will
be handled by the normal Python exception machinery.

The most important line of code in ``register_command`` is the call
to :py:func:`chimerax.core.commands.register`, whose arguments are:

- ``name`` - a Python string for the command name,
- ``cmd_desc`` - an instance of :py:class:`chimerax.core.commands.CmdDesc`
  which describes what command line arguments are expected, and
- ``function`` - a Python function to process the command.

In this example, the command name comes from the command information
instance, ``ci.name``.  Both the argument description and the Python
function are defined in another package module: ``cmd.py``.
The argument description comes from ``cmd.hello_world_desc``, possibly
augmented with help text from ``ci.synopsis``.  The command-processing
function also comes from the same module, ``cmd.hello_world``.
The arguments that ``cmd.hello_world`` will be called with are
determined by the attributes of ``cmd.hello_world_desc`` and is
described below.

Note that ``register_command`` and other ``BundleAPI`` methods are static
methods and are not associated with the ``bundle_api`` instance.
The intent is that these methods remain simple and should not need
other data.  If necessary, the methods can, of course, refer to
``bundle_api``.


``cmd.py``
----------

.. literalinclude:: ../../../src/examples/tutorials/hello_world/src/cmd.py
    :language: python
    :linenos:

To implement the ``hello`` command, two components are needed:
a function that prints ``Hello world!`` to the ChimeraX log,
and a description to register so that ChimeraX knows how to
parse the typed command text and call the function with the
appropriate arguments.
In this simple example, ``hello_world`` is the name of the function
and ``hello_world_desc`` is the description for the command.
(Note that the function and description names need not match
the command name.)

``hello_world_desc``, the command description, is an
instance of `chimerax.core.commands.CmdDesc`.  No
arguments are passed to the constructor, meaning the
user should not type anything after the command name.
If additional text is entered after the command, ChimeraX will flag
that as an error and display an error message without invoking
the ``hello_world`` function.

If the command is entered correctly, ChimeraX calls the
``hello_world`` function with a single argument, ``session``,
which provides access to session data such as the open models
and current selection.  For this example, ``hello_world`` uses
the session logger, an instance of `chimerax.core.logger.Logger`,
to display the informational message "Hello world!"  The message
is displayed in the log window when the ChimeraX graphical
interface is displayed; otherwise, it is printed to the console.

Later tutorials will discuss how to use the command description
to inform ChimeraX how to convert input text to Python values
and map them to arguments when calling the command-processing
function.

.. include:: build_test_distribute.rst
