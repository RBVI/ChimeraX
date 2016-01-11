Functions for User Commands
***************************

The *chimerax.core.commands* module contains Python functions for the core commands
available from the ChimeraX command-line.  Tools define additional commands noted with a (*)
and their code location is described with the function descriptions.

| `alias`_ - Define aliases for commands
| `buriedarea`_ - Compute buried area between atomic structures
| `camera`_ - Change camera settings
| `cartoon`_ * - Display atomic structure ribbons or pipes/planks
| `clip`_ - Clip models
| `close`_ - Close models
| `cofr`_ - Set center of rotation method
| `color`_ - Color atoms, ribbons and surfaces
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
| `position`_ - Set camera and model positions
| `pwd`_ - Present working directory
| `rainbow`_ - Color residues and chains
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
| `time`_ - Time a command
| `title`_ * - Show text titles with graphics
| `toolshed`_ * - Control the Tool Shed user interface
| `transparency`_ - Set transparency of atoms, ribbons and surfaces
| `turn`_ - Rotate models
| `view`_ - Move camera to view specified objects
| `volume`_ - Change density map display settings
| `vop`_ - Filter density maps
| `vseries`_ - Play density map time series
| `wait`_ - Wait before executing next command
| `windowsize`_ - Set or report graphics window size
| `zoom`_ - Zoom in or out

alias
=====
.. autofunction:: chimerax.core.commands.alias.alias

buriedarea
==========
.. autofunction:: chimerax.core.commands.buriedarea.buriedarea

camera
======
.. autofunction:: chimerax.core.commands.camera.camera

cartoon
=======
.. autofunction:: chimerax.exp_cmd.cartoon.cartoon
.. autofunction:: chimerax.exp_cmd.cartoon.uncartoon

clip
=====
.. autofunction:: chimerax.core.commands.clip.clip

close
=====
.. autofunction:: chimerax.core.commands.close.close

cofr
====
.. autofunction:: chimerax.core.commands.cofr.cofr

color
=====
.. autofunction:: chimerax.core.commands.color.color

contacts
========
Function found in *chimerax.contacts.cmd*

.. autofunction:: chimerax.contacts.cmd.contacts

crossfade
=========
.. autofunction:: chimerax.core.commands.crossfade.crossfade

crosslinks
==========
.. autofunction:: chimerax.core.commands.crosslinks.crosslinks

delete
======
.. autofunction:: chimerax.core.commands.delete.delete

echo
====
.. autofunction:: chimerax.core.commands.echo.echo

exit
====
.. autofunction:: chimerax.core.commands.exit.exit

export
======
.. autofunction:: chimerax.core.commands.export.export

fitmap
======
.. autofunction:: chimerax.core.map.fit.fitcmd.fitmap

help
====
.. autofunction:: chimerax.help_viewer.cmd.help

hide
====
.. autofunction:: chimerax.core.commands.hide.hide

ks
==
.. autofunction:: chimerax.shortcuts.shortcuts.ks

lighting
========
.. autofunction:: chimerax.core.commands.lighting.lighting

list
====
.. autofunction:: chimerax.core.commands.list.list

log
===
Function found in *chimerax.log.cmd*

.. autofunction:: chimerax.log.cmd.log

material
========
.. autofunction:: chimerax.core.commands.material.material

molmap
======
.. autofunction:: chimerax.core.map.molmap.molmap

mousemode
=========
.. autofunction:: chimerax.core.commands.mousemode.mousemode

move
====
.. autofunction:: chimerax.core.commands.move.move


movie
=====
Functions defined in *chimerax.movie.moviecmd*

.. automodule:: chimerax.movie.moviecmd
  :members:
  :member-order: bysource

oculus
======
.. autofunction:: chimerax.core.devices.oculus.oculuscmd.oculus

open
====
.. autofunction:: chimerax.core.commands.open.open

pdbimages
=========
.. autofunction:: chimerax.core.commands.pdbimages.pdbimages

perframe
========
.. autofunction:: chimerax.core.commands.perframe.perframe

position
========
.. autofunction:: chimerax.core.commands.position.position
.. autofunction:: chimerax.core.commands.position.position_initial

pwd
===
.. autofunction:: chimerax.core.commands.pwd.pwd

rainbow
=======
.. autofunction:: chimerax.core.commands.rainbow.rainbow

roll
====
.. autofunction:: chimerax.core.commands.roll.roll

run
===
.. autofunction:: chimerax.core.commands.run.run

sasa
====
.. autofunction:: chimerax.core.commands.sasa.sasa

save
====
.. autofunction:: chimerax.core.commands.save.save

scolor
======
.. autofunction:: chimerax.core.commands.scolor.scolor

select
======
.. autofunction:: chimerax.core.commands.select.select

set
===
.. autofunction:: chimerax.core.commands.set.set

show
====
.. autofunction:: chimerax.core.commands.show.show

snav
====
.. autofunction:: chimerax.core.devices.spacenavigator.snav.snav

split
=====
.. autofunction:: chimerax.core.commands.split.split

stop
====
.. autofunction:: chimerax.core.commands.stop.stop

style
=====
.. autofunction:: chimerax.core.commands.style.style

surface
=======
.. autofunction:: chimerax.core.commands.surface.surface

sym
===
.. autofunction:: chimerax.core.commands.sym.sym

time
====
.. autofunction:: chimerax.core.commands.time.time

title
=====
Function found in *chimerax.title.label*

.. autofunction:: chimerax.title.label.title_create
.. autofunction:: chimerax.title.label.title_change
.. autofunction:: chimerax.title.label.title_delete

toolshed
========
Function defined in *chimerax.toolshed.cmd*

.. automodule:: chimerax.toolshed.cmd
  :members:
  :member-order: bysource

transparency
============
.. autofunction:: chimerax.core.commands.transparency.transparency

turn
====
.. autofunction:: chimerax.core.commands.turn.turn

view
====
.. autofunction:: chimerax.core.commands.view.view

volume
======
.. autofunction:: chimerax.core.map.volumecommand.volume

vop
===
.. automodule:: chimerax.core.map.filter.vopcommand
  :members:		

vseries
=======
.. automodule:: chimerax.core.map.series.vseries_command
  :members:

wait
====
.. autofunction:: chimerax.core.commands.wait.wait

windowsize
==========
.. autofunction:: chimerax.core.commands.windowsize.windowsize

zoom
====
.. autofunction:: chimerax.core.commands.zoom.zoom

