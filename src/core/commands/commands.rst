Functions for User Commands
***************************

The *chimera.core.commands* module contains Python functions for the core commands
available from the Chimera command-line.  Tools define additional commands noted with a (*)
and their code location is described with the function descriptions.

| `alias`_ - Define aliases for commands
| `buriedarea`_ - Compute buried area between atomic structures
| `camera`_ - Change camera settings
| `cartoon`_ * - Display atomic structure ribbons or pipes/planks
| `clip`_ - Clip models
| `close`_ - Close models
| `cofr`_ - Set center of rotation method
| `color`_ - Color atoms, ribbons and surfaces
| `colordef`_ - Define new colors
| `contacts`_ * - Make a buried area contact graph
| `crossfade`_ - Fade between scenes for movie making
| `crosslinks`_ - Move atomic structures to minimize crosslinks
| `delete`_ - Delete models
| `echo`_ - Write message to log
| `exit`_ - Quit
| `export`_ - Export files
| `fitmap`_ - fit atomic structures in density maps
| `help`_ - Show documentation
| `hide`_ - Hide atoms, ribbons, models
| `ks`_ - Enable keyboard shortcuts
| `lighting`_ - Change lighting parameters
| `list`_ - List open models
| `log`_ * - Clear or save the log
| `material`_ - Change surface material light reflection properties
| `molmap`_ - Calculate a density map from atoms
| `mousemode`_ - Set mouse modes
| `movie`_ * - Record videos
| `move`_ - Move camera
| `oculus`_ - Enable Oculus Rift headset
| `open`_ - Open data files
| `pdbimages`_ - Render PDB assembly images
| `perframe`_ - Run a command for every rendered frame
| `pwd`_ - Present working directory
| `roll`_ - Rotate models
| `run`_ - Run a user command string
| `sasa`_ - Compute solvent accessible surface area
| `save`_ - Save sessions, images, density maps...
| `scolor`_ - Color surfaces
| `select`_ - Select objects
| `set`_ - Set rendering effects (background color, silhouettes)
| `show`_ - Show atoms, ribbons, models
| `snav`_ - Enable space navigator input device
| `split`_ - Split atomic structures into pieces
| `stop`_ - Stop motion
| `style`_ - Change atom display style
| `surface`_ - Compute a molecular surface
| `sym`_ - Show molecular assemblies involving symmetry
| `title`_ * - Show text titles with graphics
| `toolshed`_ * - Control the Tool Shed user interface
| `transparency`_ - Set transparency of atoms, ribbons and surfaces
| `turn`_ - Rotate models
| `view`_ - Move camera to view specified objects
| `volume`_ - Change density map display settings
| `vop`_ - Filter density maps
| `vseries`_ - Play density map time series
| `wait`_ - Wait before executing next command
| `zoom`_ - Zoom in or out

alias
=====
.. autofunction:: chimera.core.commands.alias.alias

buriedarea
==========
.. autofunction:: chimera.core.commands.buriedarea.buriedarea

camera
======
.. autofunction:: chimera.core.commands.camera.camera

cartoon
=======
.. autofunction:: chimera.exp_cmd.cartoon.cartoon
.. autofunction:: chimera.exp_cmd.cartoon.uncartoon

clip
=====
.. autofunction:: chimera.core.commands.clip.clip

close
=====
.. autofunction:: chimera.core.commands.close.close

cofr
====
.. autofunction:: chimera.core.commands.cofr.cofr

color
=====
.. autofunction:: chimera.core.commands.color.color

colordef
========
.. autofunction:: chimera.core.commands.colordef.colordef
.. autofunction:: chimera.core.commands.colordef.uncolordef

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

echo
====
.. autofunction:: chimera.core.commands.echo.echo

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
.. autofunction:: chimera.help_viewer.cmd.help

hide
====
.. autofunction:: chimera.core.commands.hide.hide

ks
==
.. autofunction:: chimera.shortcuts.shortcuts.ks

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

mousemode
=========
.. autofunction:: chimera.core.commands.mousemode.mousemode

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

roll
====
.. autofunction:: chimera.core.commands.roll.roll

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

select
======
.. autofunction:: chimera.core.commands.select.select

set
===
.. autofunction:: chimera.core.commands.set.set

show
====
.. autofunction:: chimera.core.commands.show.show

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

title
=====
Function found in *chimera.title.label*

.. autofunction:: chimera.title.label.title_create
.. autofunction:: chimera.title.label.title_change
.. autofunction:: chimera.title.label.title_delete

toolshed
========
Function defined in *chimera.toolshed.cmd*

.. automodule:: chimera.toolshed.cmd
  :members:
  :member-order: bysource

transparency
============
.. autofunction:: chimera.core.commands.transparency.transparency

turn
====
.. autofunction:: chimera.core.commands.turn.turn

view
====
.. autofunction:: chimera.core.commands.view.view

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

zoom
====
.. autofunction:: chimera.core.commands.zoom.zoom

