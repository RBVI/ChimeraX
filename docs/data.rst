Data Interface
==============

Architecture
------------

Data I/O is managed by the :py:mod:`chimera2.data` module.
Data formats need to be registered with information about how
to recognize files of that type and functions to read and/or write them.

.. todo:

    Most data is contained in the universe.
    Ancillary data, like templates, are kept separate.
    Access to the universe is by::

        from chimera2 import universe

    Data in the universe is organized in to Groups.
    While data has no a priori meaning, 
    visualization is an important part of Chimera 2,
    and Groups are where the visualization code hooks in.

    To query the universe for data of a particular type,
    *i.e*., data that supports are particular interface,
    data types should either subclass from a common base class,
    or use Python's Abstract Base Class (ABC) mechanism
    to register common interface support.

Modules
-------

.. toctree::

    datasupport.rst
