..  vim: set expandtab shiftwidth=4 softtabstop=4:

:orphan:

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


===========================
ChimeraX Developer Tutorial
===========================

UCSF ChimeraX is designed to be extensible, much like
Cytoscape_ and `Mozilla Firefox`_.  In ChimeraX, the
units of extension are called **bundles**, which may
be installed at run-time to add **commands**, graphical
interfaces (**tools**), chemical subgroup **selectors**,
support for fetching from network databases, and
reading and writing data files in new formats.

Most bundles can be built using ChimeraX itself.
Once built, a bundle is stored as a single file and
may be exchanged among developers and users, or
made available to the ChimeraX community via
the `ChimeraX Toolshed`_, a web-based bundle
repository based on the `Cytoscape App Store`_.


Prerequisites
=============

Other than developing graphical interfaces, writing
ChimeraX bundles only requires a working knowledge
of Python and XML.  Graphical interfaces may be
written using web development techniques, i.e.,
using HTML and Javascript.  If more graphical
capabilities are needed, developers can use `PyQt`_.
The standard ChimeraX distribution has all the
tools needed to build bundles that only use these
languages and toolkits.

For greater performance, it is possible to include
`C and C++ extensions`_ because ChimeraX is based
on CPython.  However, developers are responsible
for supplying a compatible compilation environment,
e.g., installing Visual Studio 2015 on Microsoft Windows.

.. _pro-tip:

Pro Tip
=======
One the handiest things to know is that if you want
your code to do something that a command already does,
instead of figuring out the equivalent Python call you
can just run the command.  For instance, the code to
run the command ``color red`` is::

    from chimerax.core.commands import run
    run(session, "color red")

The ``run()`` call will return the result of the command,
*e.g.* a list of opened models for the ``open`` command
(in most cases a list of one) or a distance value for the
``distance`` command.  If the "command" is actually a
semi-colon-separated list of commands, the the returned
value will be a list of the individual return values.

If all you need to do is run a fixed set of commands over
a set of data files, you should be aware that the
`forEachFile option <../../user/commands/open.html#forEachFile>`_
to the
`open command <../../user/commands/open.html>`_
does exactly that.

What is a ChimeraX Bundle?
==========================

ChimeraX is implemented in Python 3, with chunks
of C++ thrown in for performance.  A bundle
is simply a Python package that conform to
ChimeraX coding and data conventions.
To shorten start-up time, ChimeraX does *not*
import each bundle on initialization.  Instead, when
a bundle is installed, ChimeraX reads the bundle
(package) data to incorporate the new functionality
into the application user interface
(e.g., registering commands and adding menu items).
When the new functionality is used, ChimeraX
then imports the bundle and invokes the
corresponding code to handle the requests.

The source code for a bundle typically consists
of Python modules, but may
also include C++ for performance reasons.
Additional files, such as license text, icons,
images, etc., may also be included.
The source code must be turned into a bundle before
it is ready for installation into ChimeraX.
The `Source Code Organization`_ section
goes into detail on the recommended file
structure for ChimeraX bundle source code.

ChimeraX expects bundles to be in `Python wheel`_ format.
Python has `standard methods
<https://packaging.python.org/en/latest/distributing/#packaging-your-project>`_
for building wheels from source code.
However, it typically requires writing a ``setup.py``
file that lists the source files, data files
and metadata that should be included in the
package.  To simplify building bundles, ChimeraX
provides the ``devel`` command that reads an
XML file containing the bundle information,
generates the corresponding ``setup.py`` file,
and runs the script to build and/or install
the bundle.  While the XML file contains the
same information as a ``setup.py`` file, it is
simpler to use because it does *not* contain
extraneous Python code and data that is required
(for ``setup.py`` to run successfully) yet is
identical across all bundles.

Once a bundle is built, it can be added to ChimeraX.
Normally, wheels are installed using the ``pip`` module
in a Python-based application.  However, because
some post-processing must be done after a
wheel is installed (i.e., read package data and
integrate functionality into user interface),
ChimeraX provides the ``toolshed install`` command
for use in place of ``pip``.
(If a bundle *must* be installed using ``pip``,
one should run the ``toolshed reload`` command
to properly integrate the bundle into ChimeraX.)


Source Code Organization
========================

The ChimeraX ``devel`` command expects bundle source
code to be arranged in a folder (a.k.a., directory)
in a specific manner: there must be a
``bundle_info.xml`` file and a ``src`` subfolder.


Bundle Information
------------------

The ``bundle_info.xml`` file contains information
read by the ``devel`` command.  It describes both
what functionality is provided in the bundle,
and what source code files need to be included
to build the bundle.

The format of ``bundle_info.xml`` is, as its name
suggests, `eXtensible Markup Language`_ (XML).
Bundle information is grouped into nested *tags*,
which can have zero or more attributes as well
as associated text.

The document tag (the one containing all other tags)
is ``BundleInformation`` whose attributes include
the bundle name, version, and Python package name.
(Other attributes are also supported but are not required.)
``Classifier`` tags provide information about 
supported functionality.  All Python source files
in the ``src`` sub-folder are automatically included,
but additional files, e.g., icons and data files,
must be explicitly listed (e.g., in ``DataFiles`` tags)
for inclusion in the bundle.

Examples of ``bundle_info.xml`` files are provided
in tutorials for building example bundles
(see `Writing Bundles in a Few Easy Steps`_).
The complete set of supported tags are described in
:doc:`bundle_info`.


Python Package
--------------

The ``src`` folder is organized as a `Python package`_,
and implements the bundle functionality.
The ``__init__.py`` file is required.  All other
files, source code or data, are optional.

ChimeraX assumes that the Python package contains
an object named ``bundle_api`` that is an instance
of a subclass of :py:class:`chimerax.core.toolshed.BundleAPI`,
which defines the attributes and methods required
by ChimeraX.  For example,

- the ``api_version`` attribute controls the calling
  conventions for defined methods;
- the ``register_command`` method is called to
  register bundle commands; and
- the ``start_tool`` method is called to start a
  graphical interface tool.

A bundle does not need to provide all bundle API methods,
as the implementation from inherited methods
"do the right thing".

The ``bundle_api`` object, typically defined
in ``__init__.py``, is the only interface that ChimeraX
uses to invoke bundle functionality.  The bundle can,
of course, invoke ChimeraX functionality through the
published APIs for :ref:`ChimeraX Python Modules`.

Because ChimeraX is implemented (mostly) in Python,
all code is available for inspection and introspection,
so developers can essentially use all defined attributes
and methods.  However, using *only* the published APIs has
one great advantage.  Published ChimeraX APIs conform
to `semantic versioning`_ rules, i.e., within the same
major version of ChimeraX, the APIs from newer minor
versions will always be compatible with older minor versions.
That means bundles written for one version of ChimeraX
will always work with new (minor) versions as well.
(Semantic versioning only applies for pure Python
ChimeraX bundles.  Bundles containing code that
compile against ChimeraX C++ APIs are only guaranteed
to work with the exact version used in development.)

.. _A Few Steps:

Writing Bundles in a Few Easy Steps
-----------------------------------

The easiest way to start developing ChimeraX
bundles is to follow these tutorials
for building example bundles:

.. toctree::
   :maxdepth: 1

   tutorial_hello
.. toctree::
   :maxdepth: 1

   tutorial_command
.. toctree::
   :maxdepth: 2

   tutorial_tool
.. toctree::
   :maxdepth: 1

   tutorial_read_format
.. toctree::
   :maxdepth: 1

   tutorial_save_format
.. toctree::
   :maxdepth: 1

   tutorial_fetch
.. toctree::
   :maxdepth: 1

   tutorial_selector
.. toctree::
   :maxdepth: 1

   tutorial_presets

Each tutorial builds on the previous but may also
be used as reference for adding a specific type of
functionality to ChimeraX.

.. include:: build_test_distribute.rst
