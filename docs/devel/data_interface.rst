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

Data Interface
==============

Architecture
------------

There are modules with the core data structures for each
native data type: :doc:`molecular data <bundles/atomic/src/atomic>`,
sequence data, and volume data.

New data formats are registered with the "data formats" manager,
and functions to read and/or write formats are registered
with the "open command" and "save command" managers respectively.
This is summarized briefly in :ref:`data format`, :ref:`open command`, and
:ref:`save command`, and in more detail in the :doc:`tutorials/tutorial_read_format`,
:doc:`tutorials/tutorial_save_format`, and :doc:`tutorials/tutorial_fetch` tutorials.

.. note::

    Trigger documentation might go here.

Modules
-------

.. toctree::
    :maxdepth: 1

    core/commands/commands.rst

    core/data_events.rst

    core/objects.rst

    core/session.rst

    core/triggerset.rst

    core/scripting.rst

Structure-related Modules
~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    bundles/mmcif/src/mmcif.rst

    bundles/atomic/src/pdbio.rst

Graphics-related Modules
~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    bundles/geometry/src/geometry.rst

    bundles/graphics/src/graphics.rst

C++ helper Modules
~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    core/appdirs_cpp/appdirs_cpp.rst

.. seealso::

    :doc:`Readcif <bundles/mmcif/mmcif_cpp/readcif_cpp/docs/api>`
        C++ library for reading mmCIF files.

Builtin Data formats
--------------------

.. toctree::
    :maxdepth: 2
