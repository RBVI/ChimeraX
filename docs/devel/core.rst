..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

.. _core:

ChimeraX Core
=============

There are three major components of the ChimeraX core:
the :doc:`user interface <user_interface>` modules,
the :doc:`data interface <data_interface>` modules,
and the :doc:`tool interface <core/tools>` modules.
The user interface modules support the GUI and the command line interfaces,
the data interface modules support the native data types,
and the tool interface modules support common functionality
and tool registration.

In additons to the core functionality,
there are the :doc:`bundles` that use the core,
and the :doc:`applications` that use the functionality.

.. note::

    The following modules are in the process of being reorganzied.

Contents
========

.. toctree::
    :maxdepth: 2

    user_interface.rst

    data_interface.rst

    infrastructure.rst

    applications.rst

    conventions.rst
