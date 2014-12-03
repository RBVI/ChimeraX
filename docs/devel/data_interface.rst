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

    core/serialize.rst

    core/io.rst

    structaccess.rst

    connectivity.rst

    core/cpp_appdirs/cpp_appdirs.rst

    hydra_geometry.rst

    hydra_graphics.rst

.. seealso::

    :doc:`Readcif <readcif>`
        C++ library for reading mmCIF files.

Builtin Data formats
--------------------

.. toctree::
    :maxdepth: 2
