******************
Developer's Manual
******************

This manual covers the :ref:`core`, the associated tools and libraries,
and the internals of application(s) built upon them.
The actual applications are documented in the
the :doc:`User Manual </user/index>`.

This code is free for non-commerical use, see the
:doc:`license <../license>` for details.

The various interfaces are implemented in `Python <http://www.python.org/>`_
with occasional help from C or C++ code.

.. toctree::
    :maxdepth: 2

    conventions.rst

.. _core:

Chimera Core
============

There are three major components of the Chimera core:
the :doc:`user interface <user_interface>` modules,
the :doc:`data interface <data_interface>` modules,
and the :doc:`tool interface <tool_interface>` modules.
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

    tool_interface.rst

    tools.rst

    applications.rst
