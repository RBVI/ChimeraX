..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. _Cytoscape: http://www.cytoscape.org/
.. _Mozilla Firefox: https://www.mozilla.org/firefox/
.. _ChimeraX Toolshed: https://cxtoolshed.rbvi.ucsf.edu/
.. _Cytoscape App Store: http://apps.cytoscape.org/
.. _Python wheel: https://wheel.readthedocs.org/

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
be installed at run-time to add **commands**,
graphical interfaces (**tools**),
chemical subgroup **selectors**,
support for fetching from network databases, and
reading and writing data files in new formats.

Most bundles can be built using ChimeraX itself.
Once built, a bundle is stored as a single file and
may be exchanged among developers and users, or
made available to the ChimeraX community via
the `ChimeraX Toolshed`_, an extension repository
based on the `Cytoscape App Store`_.


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
The _`Bundle Organization` section
goes into detail on the recommended file
organization for ChimeraX bundle source code.

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
