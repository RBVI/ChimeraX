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

Efficient collections of molecular objects and molecular surfaces are also available

.. toctree::
   :maxdepth: 1

   molarray.rst
   molsurf.rst

Atomic data classes
-------------------

.. automodule:: chimerax.core.atomic.structure
    :members:

.. automodule:: chimerax.core.atomic.molobject
    :members:

.. automodule:: chimerax.core.atomic.pbgroup
    :members:
