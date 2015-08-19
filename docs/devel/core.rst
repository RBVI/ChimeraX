..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. _core:

Chimera Core
============

There are three major components of the Chimera core:
the :doc:`user interface <user_interface>` modules,
the :doc:`data interface <data_interface>` modules,
and the :doc:`tool interface <core/tools>` modules.
The user interface modules support the GUI and the command line interfaces,
the data interface modules support the native data types,
and the tool interface modules support common functionality
and tool registration.

In additons to the core functionality,
there are the :doc:`tools` that use the core,
and the :doc:`applications` that bundle the functionality.

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
