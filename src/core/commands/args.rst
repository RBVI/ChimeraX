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

chimerax.core.commmands Module
******************************

The *chimerax.core.commands* module defines the registration function
for new commands as well as argument types such as ``FloatArg`` and 
``AtomsArg`` which are Python classes used for parsing typed
command arguments.  These functions and classes are actually imported
from a submodule, but the supported API only includes the items
defined in the main *chimerax.core.commands* module.

.. py:module:: chimerax.core.commands

.. automodule:: chimerax.core.commands
    :show-inheritance:
    :members: register, run, AtomicStructuresArg, AtomsArg, AtomSpecArg, AxisArg, BoolArg, CenterArg, ColorArg, ColormapArg, EmptyArg, IntArg, ModelArg, ModelPlacesArg, ModelsArg, NoArg, ObjectsArg, PlaceArg, PseudobondGroupsArg, StringArg, TopModelsArg, ColorArg, ColormapArg
