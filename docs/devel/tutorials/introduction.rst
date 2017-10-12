..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. _Cytoscape: http://www.cytoscape.org/
.. _Mozilla Firefox: https://www.mozilla.org/firefox/
.. _ChimeraX Toolshed: https://cxtoolshed.rbvi.ucsf.edu/
.. _Cytoscape App Store: http://apps.cytoscape.org/
.. _Python wheel: https://wheel.readthedocs.org/
.. _Python package: https://docs.python.org/3/tutorial/modules.html#packages
.. _eXtensible Markup Language: https://en.wikipedia.org/wiki/XML
.. _PyQt: https://riverbankcomputing.com/software/pyqt/intro
.. _C and C++ Extensions: https://docs.python.org/3/extending/building.html
.. _CPython: https://en.wikipedia.org/wiki/CPython

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
the `ChimeraX Toolshed`_, an extension repository
based on the `Cytoscape App Store`_.


Prerequisites
-------------

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


What is a ChimeraX Bundle?
--------------------------

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
The _`Source Code Organization` section
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
------------------------

The ChimeraX ``devel`` command expects bundle source
code to be arranged in a folder (a.k.a., directory)
in a specific manner: there must be a ``src`` subfolder
and a ``bundle_info.xml`` file.

The ``src`` folder contains the source code for the
bundle, and is organized like a `Python package`_.
The ``__init__.py`` file is required.  All other
files, source code or data, are optional.

The ``bundle_info.xml`` file contains information
read by the ``devel`` command.  It describes both
what functionality is provided in the bundle,
and what source code files need to be included
to build the bundle.  (All Python source files
in ``src`` are automatically included, but
additional files, e.g., icons and data files,
must be explicitly listed for inclusion.)
The format of ``bundle_info.xml`` is, as its name
suggests, `eXtensible Markup Language`_ (XML).
The supported tags are described in
:ref:`Bundle Information XML Tags`.

The easiest way to understand the source code
organization is to follow one of the tutorials
for building sample bundles:

- :ref:`Bundle Tutorial: Hello World`
- :ref:`Bundle Tutorial: Add a Command`
- :ref:`Bundle Tutorial: Add a Tool`
- :ref:`Bundle Tutorial: Read a New File Format`
- :ref:`Bundle Tutorial: Save a New File Format`
- :ref:`Bundle Tutorial: Fetch from Network Database`
- :ref:`Bundle Tutorial: Define a Chemical Subgroup Selector`


Building and Testing Bundles
----------------------------

To build a bundle, start ChimeraX and execute the command:

``devel build PATH_TO_SOURCE_CODE_FOLDER``

Python source code and other resource files are copied
into a ``build`` sub-folder below the source code
folder.  C/C++ source files, if any, are compiled and
also copied into the ``build`` folder.
The files in ``build`` are then assembled into a
Python wheel in the ``dist`` sub-folder.

To test the bundle, execute the ChimeraX command:

``devel install PATH_TO_SOURCE_CODE_FOLDER``

This will build the bundle, if necessary, and install
the bundle in ChimeraX.  Bundle functionality should
be available immediately.

To remove temporary files created while building
the bundle, execute the ChimeraX command:

``devel clean PATH_TO_SOURCE_CODE_FOLDER``

Some files, such as the bundle itself, may still remain
and need to be removed manually.


Distributing Bundles
--------------------

With ChimeraX bundles being packages as standard Python
wheel-format files, they can be distributed as plain files
and installed using the ChimeraX ``toolshed install``
command.  Thus, electronic mail, web sites and file
sharing services can all be used to distribute ChimeraX
bundles.

Private distributions are most useful during bundle
development, when circulation may be limited to testers.
When bundles are ready for public release, they can be
published on the `ChimeraX Toolshed`_, which is designed
to help developers by eliminating the need for custom
distribution channels, and to aid users by providing
a central repository where bundles with a variety of
functionality may be found.

Customizable information for each bundle on the toolshed
includes its description, screen captures, authors,
citation instructions and license terms.
Automatically maintained information
includes release history and download statistics.

To submit a bundle for publication on the toolshed,
you must first sign in.  Currently, only Google
sign in is supported.  Once signed in, use the
``Submit a Bundle`` link at the top of the page
to initiate submission, and follow the instructions.
The first time a bundle is submitted to the toolshed,
approval from ChimeraX staff is needed before it is
published.  Subsequent submissions, using the same
sign in credentials, do not need approval and should
appear immediately on the site.

