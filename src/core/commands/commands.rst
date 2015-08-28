Functions for User Commands
***************************

The *chimera.core.commands* module contains Python functions for the core commands
available from the Chimera command-line.  Tools define additional commands noted with a (*)
and their code location is described with the function descriptions.

| `buriedarea`_ - Compute buried area between atomic structures
| `camera`_ - Change camera settings
| `close`_ - Close models
| `color`_ - Color atoms
| `colordef`_ - Define new colors
| `command_line`_ * - Hide or show the command line
| `contacts`_ * - Make a buried area contact graph
| `crossfade`_ - Fade between scenes for movie making
| `crosslinks`_ - Move atomic structures to minimize crosslinks
| `delete`_ - Delete models
| `display`_ - Display atoms
| `echo`_ - Write message to log
| `ecolor`_ - Experimental color command
| `exit`_ - Quit
| `export`_ - Export files
| `fitmap`_ - fit atomic structures in density maps
| `help`_ - Show documentation
| `ks`_ - Enable keyboard shortcuts
| `2dlabel`_ * - Create text labels
| `lighting`_ - Change lighting parameters
| `list`_ - List open models
| `log`_ * - Hide and show the log
| `material`_ - Change surface material light reflection properties
| `molmap`_ - Calculate a density map from atoms
| `mousemodes`_ * - Hide or show mouse modes panel
| `movie`_ * - Record videos
| `move`_ - Move camera
| `oculus`_ - Enable Oculus Rift headset
| `open`_ - Open data files
| `pdbimages`_ - Render PDB assembly images
| `perframe`_ - Run a command for every rendered frame
| `pwd`_ - Present working directory
| `rcolor`_ - Color ribbons
| `roll`_ - Rotate models
| `ribbon`_ - Display atomic structure ribbons
| `run`_ - Run a user command string
| `sasa`_ - Compute solvent accessible surface area
| `save`_ - Save sessions, images, density maps...
| `scolor`_ - Color surfaces
| `set`_ - Set rendering effects (background color, silhouettes)
| `snav`_ - Enable space navigator input device
| `split`_ - Split atomic structures into pieces
| `stop`_ - Stop motion
| `style`_ - Change atom display style
| `surface`_ - Compute a molecular surface
| `sym`_ - Show molecular assemblies involving symmetry
| `toolshed`_ * - Control the Tool Shed user interface
| `turn`_ - Rotate models
| `volume`_ - Change density map display settings
| `vop`_ - Filter density maps
| `vseries`_ - Play density map time series
| `wait`_ - Wait before executing next command
| `window`_ - Move camera to view all models

buriedarea
==========
.. autofunction:: chimera.core.commands.buriedarea.buriedarea

camera
======
.. autofunction:: chimera.core.commands.camera.camera

close
=====
.. autofunction:: chimera.core.commands.close.close

color
=====
.. autofunction:: chimera.core.commands.color.color

colordef
========
.. autofunction:: chimera.core.commands.color.colordef
.. autofunction:: chimera.core.commands.color.uncolordef

command_line
============
Functions defined in *chimera.cmd_line.cmd*

.. autofunction:: chimera.cmd_line.cmd.command_line_hide
.. autofunction:: chimera.cmd_line.cmd.command_line_show

contacts
========
Function found in *chimera.contacts.cmd*

.. autofunction:: chimera.contacts.cmd.contacts

crossfade
=========
.. autofunction:: chimera.core.commands.crossfade.crossfade

crosslinks
==========
.. autofunction:: chimera.core.commands.crosslinks.crosslinks

delete
======
.. autofunction:: chimera.core.commands.delete.delete

display
=======
.. autofunction:: chimera.core.commands.display.display
.. autofunction:: chimera.core.commands.display.undisplay

echo
====
.. autofunction:: chimera.core.commands.echo.echo

ecolor
======
.. autofunction:: chimera.core.commands.color.ecolor

exit
====
.. autofunction:: chimera.core.commands.exit.exit

export
======
.. autofunction:: chimera.core.commands.export.export

fitmap
======
.. autofunction:: chimera.core.map.fit.fitcmd.fitmap

help
====
.. autofunction:: chimera.core.commands.help.help

ks
==
.. autofunction:: chimera.shortcuts.shortcuts.ks

2dlabel
=======
Function found in *chimera.label.label*

.. autofunction:: chimera.label.label.label_create
.. autofunction:: chimera.label.label.label_change
.. autofunction:: chimera.label.label.label_delete

lighting
========
.. autofunction:: chimera.core.commands.lighting.lighting

list
====
.. autofunction:: chimera.core.commands.list.list

log
===
Function found in *chimera.log.cmd*

.. autofunction:: chimera.log.cmd.log

material
========
.. autofunction:: chimera.core.commands.material.material

molmap
======
.. autofunction:: chimera.core.map.molmap.molmap

mousemodes
==========
Functions defined in *chimera.mouse_modes.cmd*

.. autofunction:: chimera.mouse_modes.cmd.mousemodes_hide
.. autofunction:: chimera.mouse_modes.cmd.mousemodes_show

move
====
.. autofunction:: chimera.core.commands.move.move


movie
=====
Functions defined in *chimera.movie.moviecmd*

.. automodule:: chimera.movie.moviecmd
  :members:
  :member-order: bysource

oculus
======
.. autofunction:: chimera.core.devices.oculus.oculuscmd.oculus

open
====
.. autofunction:: chimera.core.commands.open.open

pdbimages
=========
.. autofunction:: chimera.core.commands.pdbimages.pdbimages

perframe
========
.. autofunction:: chimera.core.commands.perframe.perframe

pwd
===
.. autofunction:: chimera.core.commands.pwd.pwd

rcolor
======
.. autofunction:: chimera.core.commands.color.rcolor

roll
====
.. autofunction:: chimera.core.commands.roll.roll

ribbon
======
.. autofunction:: chimera.core.commands.ribbon.ribbon
.. autofunction:: chimera.core.commands.ribbon.unribbon

run
===
.. autofunction:: chimera.core.commands.run.run

sasa
====
.. autofunction:: chimera.core.commands.sasa.sasa

save
====
.. autofunction:: chimera.core.commands.save.save

scolor
======
.. autofunction:: chimera.core.commands.scolor.scolor

set
===
.. autofunction:: chimera.core.commands.set.set

snav
====
.. autofunction:: chimera.core.devices.spacenavigator.snav.snav

split
=====
.. autofunction:: chimera.core.commands.split.split

stop
====
.. autofunction:: chimera.core.commands.stop.stop

style
=====
.. autofunction:: chimera.core.commands.style.style

surface
=======
.. autofunction:: chimera.core.commands.surface.surface

sym
===
.. autofunction:: chimera.core.commands.sym.sym

toolshed
========
Function defined in *chimera.toolshed.cmd*

.. automodule:: chimera.toolshed.cmd
  :members:
  :member-order: bysource

turn
====
.. autofunction:: chimera.core.commands.turn.turn

volume
======
.. autofunction:: chimera.core.map.volumecommand.volume

vop
===
.. automodule:: chimera.core.map.filter.vopcommand
  :members:		

vseries
=======
.. automodule:: chimera.core.map.series.vseries_command
  :members:

wait
====
.. autofunction:: chimera.core.commands.wait.wait

window
======
.. autofunction:: chimera.core.commands.window.window

