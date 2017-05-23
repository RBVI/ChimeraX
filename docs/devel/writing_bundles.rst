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

Writing and Publishing Bundles
==============================

A *bundle* is a collection of code and data that can be added to
ChimeraX to provide support for new graphical tools, commands,
file formats, web databases and selection specifiers.
This document describes the details of how to create a bundle
and publish it in the ChimeraX toolshed.

Bundle Format
-------------

A ChimeraX bundle is packaged as a Python `wheel
<https://packaging.python.org/wheel_egg/>`_.

A wheel usually contains Python and/or compiled code
along with additional resources such as icons,
data files, and documentation.  While the
general Python wheel specification supports installing
files into arbitrary location, ChimeraX bundles
are limited to provide a single folder/directory,
which may be installed using the ``toolshed install``
command.  Bundle folders are typically placed in a
per-user location:

========    ========
Platform    Location
========    ========
Windows     TBD
macOS       TBD
Linux       TBD
========    ========

It is possible but not recommended to use *pip* to
install a bundle.  ChimeraX maintains a bundle
metadata cache for fast initialization, which
*pip* will not update, and therefore the bundle
functionality may not be available even though
the wheel is installed.  In this event, try running
the ``toolshed refresh`` command to force a cache
update.

*Bundle Organization*

*ChimeraX Classifiers*

Pure Python Bundles
-------------------

Platform-Specific Bundles
-------------------------

Testing Bundles
---------------

Distributing Bundles
--------------------

**Toolshed Submission**
