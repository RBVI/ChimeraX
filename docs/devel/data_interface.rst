..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

Data Interface
==============

Architecture
------------

There are modules with the core data structures for each
native data type: :doc:`molecular data <modules/atomic/atomic>`,
sequence data, and volume data.

New data formats are registered with the "data formats" manager,
and functions to read and/or write formats are registered
with the "open command" and "save command" managers respectively.
This is summarized briefly in :ref:`data format`, :ref:`open command`, and
:ref:`save command`, and in more detail in the :doc:`tutorials/tutorial_read_format`,
:doc:`tutorials/tutorial_save_format`, and :doc:`tutorials/tutorial_fetch` tutorials.

Bundles may want to be able to react to various kinds of events, such as new models
being opened, or the graphics window background coloring changing.  When such events
occur, a "trigger" will be fired, and functions that have been registered with the
trigger will be executed and given data related to the trigger.  The nuts and bolts of
how to register for a trigger is described in :doc:`modules/core/triggerset`.  Information
about where widely-used triggers can be found, their names, and what data they provide
can be found in :doc:`well_known_triggers`.

Modules
-------

.. toctree::
    :maxdepth: 1

    modules/core/commands/commands.rst

    modules/core/data_events.rst

    modules/core/objects.rst

    modules/core/session.rst

    modules/core/triggerset.rst

    modules/core/scripting.rst

Structure-related Modules
~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    modules/mmcif/mmcif.rst

    modules/atomic/pdbio.rst

Graphics-related Modules
~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    modules/geometry/geometry.rst

    modules/graphics/graphics.rst

C++ helper Modules
~~~~~~~~~~~~~~~~~~

.. seealso::

    :doc:`Readcif <cpp/mmcif/readcif/api>`
        C++ library for reading mmCIF files.

Builtin Data formats
--------------------

.. toctree::
    :maxdepth: 2
