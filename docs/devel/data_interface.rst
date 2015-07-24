..  vim: set expandtab shiftwidth=4 softtabstop=4:

Data Interface
==============

Architecture
------------

There are modules with the core data structures for each
native data type: :doc:`molecular data <molecule>`,
:doc:`sequence data <sequence>`,
and
:doc:`volume data <volume>`.

I/O is managed by the :py:mod:`chimera.core.io` module.
Data formats are registered with the io module with information about how
to recognize files of that type and functions to read and/or write them.

.. note::

    Trigger documentation might go here.

Modules
-------

.. toctree::
    :maxdepth: 2

    core/session.rst

    core/triggerset.rst

    core/data_events.rst

    core/io.rst

Structure-related Modules
~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    core/pdbio.rst

Graphics-related Modules
~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    core/geometry/public.rst

    core/graphics/public.rst

C++ helper Modules
~~~~~~~~~~~~~~~~~~

.. toctree::
    :maxdepth: 2

    core/appdirs_cpp/appdirs_cpp.rst

    core/connectivity.rst

.. seealso::

    :doc:`Readcif <core/readcif_cpp/docs/api>`
        C++ library for reading mmCIF files.

Builtin Data formats
--------------------

.. toctree::
    :maxdepth: 2
