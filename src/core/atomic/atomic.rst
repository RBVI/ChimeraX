..  vim: set expandtab shiftwidth=4 softtabstop=4:

atomic: Atomic structures
=========================

Atomic data, such as molecules read from a Protein Databank file, is managed in C++ data structures
which are made available through the following equivalent Python classes:

 * :class:`.AtomicStructure`
 * :class:`.Atom`
 * :class:`.Bond`
 * :class:`.Residue`
 * :class:`.Chain`

Also lines between atoms depicting distances or missing segments of a protein backbone are
represented as pseudobonds:

 * :class:`.Pseudobond`
 * :class:`.PseudobondGroup`
 * :class:`.PseudobondManager`

Efficient collections of molecular objects are also available:

.. toctree::
   :maxdepth: 1

   molarray.rst

Atomic data classes
-------------------

.. automodule:: chimera.core.atomic.structure
    :members:

.. automodule:: chimera.core.atomic.molobject
    :members:

.. automodule:: chimera.core.atomic.pbgroup
    :members:
