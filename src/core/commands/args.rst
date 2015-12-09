Command Argument Types
**********************

The *chimerax.core.commands* module defines argument types such as FloatArg, StringArg, AtomsArg,
ModelArg, ColorArg, PlaceArg, ... which are Python classes used for parsing typed command arguments.

| :class:`.AtomicStructuresArg` - Atomic structures
| :class:`.AtomsArg` - Atoms
| :class:`.AtomSpecArg` - Atom specifier
| :class:`.AxisArg` - Axis vector
| :class:`.BoolArg` - True/false values
| :class:`.CenterArg` - Point
| :class:`.ColorArg` - Color
| :class:`.ColormapArg` - Color maps
| :class:`.EmptyArg` - Allows missing argument
| :class:`.IntArg` - Integer
| :class:`.ModelArg` - Single model
| :class:`.ModelPlacesArg` - Models and positions
| :class:`.ModelsArg` - Any number of models
| :class:`.NoArg` - Keyword with no value
| :class:`.ObjectsArg` - Models or atoms
| :class:`.PlaceArg` - Orientation and position
| :class:`.PseudobondGroupsArg` - Pseudobond groups
| :class:`.StringArg` - Axis vector
| :class:`.TopModelsArg` - Top level models in tree

Color arguments
===============
.. autoclass:: chimerax.core.commands.colorarg.ColorArg
.. autoclass:: chimerax.core.commands.colorarg.ColormapArg

Model positions
===============
.. autoclass:: chimerax.core.commands.position.ModelPlacesArg
