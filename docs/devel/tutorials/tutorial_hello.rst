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
and the content folder, named `hello_world` extracted.
Alternatively, one can start with an empty folder
and create source files based on the samples below.
The source folder may be arbitrarily named, as it is
only used during installation; however, avoiding
whitespace characters in the folder name bypasses the
need to type quote characters in some steps.


Sample Files
============

The files in the ``hello_world`` are:

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
command, in this case, ``hello``.  (Commands may be a single
word or multiple words.  The latter is useful for grouping
multiple commands by sharing the same first word.)
The fourth field for command classifiers is its category,
in this case, ``General``.  (The category for a command is
reserved for future use but does not currently affect ChimeraX
behavior.)
The final data field for command classifiers is a synopsis
of what the command does, and is shown as help text in the
ChimeraX interface.

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
conform to :py:class:`chimerax.core.toolshed.BundleAPI`, which
has one public attribute, ``api_version``, and these methods:

- ``start_tool`` - invoked to display a graphical interface
- ``register_command`` - invoked the first time a bundle command is used
- ``register_selector`` - invoked the first time a bundle chemical subgroup selector is used
- ``open_file`` - invoked when a file of a bundle-supported format is opened
- ``save_file`` - invoked when a file of a bundle-supported format is saved
- ``initialize`` - invoked when ChimeraX starts up and the bundle needs custom initialization
- ``finish`` - invoked when ChimeraX exits and the bundle needs custom clean up
- ``get_class`` - invoked when a session is saved and a bundle object needs to be serialized


``cmd.py``
----------

.. literalinclude:: ../../../src/examples/tutorials/hello_world/src/cmd.py
    :language: python
    :linenos:

To implement the ``hello`` command, we need two components:
a function that prints ``Hello world!`` to the ChimeraX log,
and a description to register so that ChimeraX knows how to
call the function.
