..  vim: set expandtab shiftwidth=4 softtabstop=4:

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

Python Classes
==============

Atomic Structures

 * :class:`.Atom` - a molecule atom
 * :class:`.AtomicStructure` - molecular data
 * :class:`.Atoms` - a collection of atoms
 * :class:`.Bond` - a molecule bond
 * :class:`.Chain` - a polymer chain
 * :class:`.MolecularSurface` - molecular surfaces
 * :class:`.Residue` - a molecule residue

Commands and Tools

 * :class:`.CommandInfo` - user command
 * :class:`.ToolInfo` - information about an available tool
 * :class:`.ToolInstance` - running tools
 * :class:`.Toolshed` - manages tool updates

Graphics

 * :class:`.Camera` - position the scene is rendered from
 * :class:`.Drawing` - 3-d graphical object
 * :class:`.Lighting` - lighting paramters
 * :class:`.Material` - material reflectivity parameters
 * :class:`.Model` - Displayed data with id number
 * :class:`.Place` - position and orientation in space
 * :class:`.View` - graphics window

Notifications

 * :class:`.Log` - text command log
 * :class:`.TriggerSet` - event notifications

State

 * :class:`.Session` - application state
 * :class:`.Settings` - save tool preferences

Volume Data

 * :class:`.GridData` - density map data
 * :class:`.Volume` - density map
