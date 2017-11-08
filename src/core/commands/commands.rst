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

Functions for User Commands
***************************

The *chimerax.core.commands* module contains Python functions for the core commands
available from the ChimeraX command-line.  Tools define additional commands noted with a (*)
and their code location is described with the function descriptions.

| `2dlabels`_ * - Show text labels with graphics
| `alias`_ - Define aliases for commands
| `align`_ - Align one set of atoms to another
| `cage`_ * - Place models on polygonal cages
| `camera`_ - Change camera settings
| `cartoon`_ * - Display atomic structure ribbons or pipes/planks
| `cd`_ - Change working directory
| `clip`_ - Clip models
| `close`_ - Close models
| `cofr`_ - Set center of rotation method
| `color`_ - Color atoms, ribbons and surfaces
| `contacts`_ * - Make a buried area contact graph
| `coordset`_ - Play through structure coordinate sets
| `crossfade`_ - Fade between scenes for movie making
| `crosslinks`_ * - Move atomic structures to minimize crosslinks
| `delete`_ - Delete models
| `device`_ * - Enable devices such virtual reality headsets and space navigator
| `echo`_ - Write message to log
| `exit`_ - Quit
| `fitmap`_ - fit atomic structures in density maps
| `graphics`_ - low level graphics control
| `help`_ - Show documentation
| `hide`_ - Hide atoms, ribbons, models
| `info`_ * - Report model info
| `ks`_ * - Enable keyboard shortcuts
| `label`_ * - Show text labels on atoms
| `lighting`_ - Change lighting parameters
| `log`_ * - Clear or save the log
| `material`_ - Change surface material light reflection properties
| `measure`_ - Calculate properties of models
| `mlp`_ * - Color surfaces by molecular lipophilicity
| `molmap`_ - Calculate a density map from atoms
| `morph`_ * - Morph between atomic models
| `mousemode`_ - Set mouse modes
| `movie`_ * - Record videos
| `move`_ - Move camera
| `mseries`_ * - Display a series of models
| `open`_ - Open data files
| `pdbimages`_ - Render PDB assembly images
| `perframe`_ - Run a command for every rendered frame
| `pwd`_ - Print working directory
| `rainbow`_ - Color residues and chains
| `rename`_ - Rename models
| `resfit`_ * - Show density around residues
| `roll`_ - Rotate models
| `run`_ - Run a user command string
| `rungs`_ - Display nucleotides as cylinders
| `save`_ - Save sessions, images, density maps...
| `scolor`_ - Color surfaces
| `select`_ - Select objects
| `set`_ - Set rendering effects (background color, silhouettes)
| `show`_ - Show atoms, ribbons, models
| `size`_ - Change atom, bond, pseudobond sizes
| `smoothlines`_ * - Smooth paths in line drawings
| `split`_ - Split atomic structures into pieces
| `stop`_ - Stop motion
| `struts`_ * - Add struts between atoms for 3d printing
| `style`_ - Change atom display style and sizes
| `surface`_ - Compute a molecular surface
| `sym`_ - Show molecular assemblies involving symmetry
| `time`_ - Time a command
| `toolshed`_ * - Control the Tool Shed user interface
| `transparency`_ - Set transparency of atoms, ribbons and surfaces
| `turn`_ - Rotate models
| `view`_ - Move camera to view specified objects
| `volume`_ - Change density map settings and filter maps
| `vseries`_ - Play density map time series
| `wait`_ - Wait before executing next command
| `windowsize`_ - Set or report graphics window size
| `zoom`_ - Zoom in or out

2dlabels
========
Function found in *chimerax.label.label2d*

.. autofunction:: chimerax.label.label2d.label_create
.. autofunction:: chimerax.label.label2d.label_change
.. autofunction:: chimerax.label.label2d.label_delete

alias
=====
.. autofunction:: chimerax.core.commands.alias.alias

align
=====
.. autofunction:: chimerax.core.commands.align.align

cage
====
.. autofunction:: chimerax.cage_builder.cmd.cage

camera
======
.. autofunction:: chimerax.core.commands.camera.camera

cartoon
=======
.. autofunction:: chimerax.core.commands.cartoon.cartoon
.. autofunction:: chimerax.core.commands.cartoon.uncartoon

cd
=====
.. autofunction:: chimerax.core.commands.cd.cd

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

coordset
========
.. autofunction:: chimerax.core.commands.coordset.coordset

crossfade
=========
.. autofunction:: chimerax.core.commands.crossfade.crossfade

crosslinks
==========
Function found in *chimerax.crosslinks.crosslinks*

.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks
.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks_histogram
.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks_minimize
.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks_network
		  
delete
======
.. autofunction:: chimerax.core.commands.delete.delete

device
======
.. autofunction:: chimerax.spacenavigator.snav.device_snav
.. autofunction:: chimerax.vive.vr.vr

echo
====
Equivalent to the "log text" command.

exit
====
.. autofunction:: chimerax.core.commands.exit.exit

fitmap
======
.. autofunction:: chimerax.core.map.fit.fitcmd.fitmap

help
====
.. autofunction:: chimerax.help_viewer.cmd.help

hide
====
.. autofunction:: chimerax.core.commands.hide.hide

info
====
.. automodule:: chimerax.list_info.cmd
  :members:
  :member-order: bysource

ks
==
.. autofunction:: chimerax.shortcuts.shortcuts.ks

labels
======
Function found in *chimerax.label.label3d*

.. autofunction:: chimerax.label.label3d.label
.. autofunction:: chimerax.label.label3d.label_delete

lighting
========
.. autofunction:: chimerax.core.commands.lighting.lighting

log
===
Function found in *chimerax.log.cmd*

.. autofunction:: chimerax.log.cmd.log

material
========
.. autofunction:: chimerax.core.commands.material.material

measure
=======
.. autofunction:: chimerax.core.commands.measure_buriedarea.measure_buriedarea
.. autofunction:: chimerax.core.commands.measure_sasa.measure_sasa

mlp
===
Function found in *chimerax.mlp.mlp*

.. autofunction:: chimerax.mlp.mlp.mlp

molmap
======
.. autofunction:: chimerax.core.map.molmap.molmap

morph
=====
Function defined in *chimerax.morph.morph*

.. autofunction:: chimerax.morph.morph.morph


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
  :show-inheritance:

mseries
=======
.. autofunction:: chimerax.model_series.mseries.mseries
.. autofunction:: chimerax.model_series.mseries.mseries_slider

open
====
.. autofunction:: chimerax.core.commands.open.open

pdbimages
=========
.. autofunction:: chimerax.core.commands.pdbimages.pdbimages

perframe
========
.. autofunction:: chimerax.core.commands.perframe.perframe

pwd
===
.. autofunction:: chimerax.core.commands.pwd.pwd

rainbow
=======
.. autofunction:: chimerax.core.commands.rainbow.rainbow

rename
======
.. autofunction:: chimerax.core.commands.rename.rename

resfit
======
.. autofunction:: chimerax.residue_fit.cmd.resfit

roll
====
.. autofunction:: chimerax.core.commands.roll.roll

run
===
.. autofunction:: chimerax.core.commands.run.run

rungs
=====
.. autofunction:: chimerax.core.commands.rungs.rungs

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

style
=====
.. autofunction:: chimerax.core.commands.size.size

smoothlines
===========
.. autofunction:: chimerax.smooth_lines.smoothlines.smoothlines

split
=====
.. autofunction:: chimerax.core.commands.split.split

stop
====
.. autofunction:: chimerax.core.commands.stop.stop

struts
======
.. autofunction:: chimerax.struts.struts.struts
.. autofunction:: chimerax.struts.struts.struts_delete

style
=====
.. autofunction:: chimerax.core.commands.style.style

surface
=======
.. autofunction:: chimerax.core.commands.surface.surface
.. autofunction:: chimerax.core.commands.surface.surface_close
.. autofunction:: chimerax.core.commands.sop.surface_dust
.. autofunction:: chimerax.core.commands.surface.surface_hide
.. autofunction:: chimerax.core.commands.surface.surface_show
.. autofunction:: chimerax.core.commands.sop.surface_zone

sym
===
.. autofunction:: chimerax.core.commands.sym.sym

time
====
.. autofunction:: chimerax.core.commands.time.time

toolshed
========
.. autofunction:: chimerax.core.commands.toolshed.toolshed_list
.. autofunction:: chimerax.core.commands.toolshed.toolshed_reload
.. autofunction:: chimerax.core.commands.toolshed.toolshed_install
.. autofunction:: chimerax.core.commands.toolshed.toolshed_uninstall
.. autofunction:: chimerax.core.commands.toolshed.toolshed_url
.. autofunction:: chimerax.core.commands.toolshed.toolshed_show
.. autofunction:: chimerax.core.commands.toolshed.toolshed_hide

transparency
============
.. autofunction:: chimerax.core.commands.transparency.transparency

turn
====
.. autofunction:: chimerax.core.commands.turn.turn

view
====
.. autofunction:: chimerax.core.commands.view.view
.. autofunction:: chimerax.core.commands.view.view_delete
.. autofunction:: chimerax.core.commands.view.view_initial
.. autofunction:: chimerax.core.commands.view.view_list
.. autofunction:: chimerax.core.commands.view.view_matrix
.. autofunction:: chimerax.core.commands.view.view_name

volume
======
.. autofunction:: chimerax.core.map.volumecommand.volume
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_add
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_bin
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_boxes
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_cover
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_falloff
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_flatten
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_flip
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_fourier
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_gaussian
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_laplacian
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_local_correlation
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_maximum
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_median
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_minimum
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_morph
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_multiply
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_octant
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_permute_axes
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_resample
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_ridges
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_scale
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_subtract
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_threshold
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_tile
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_unbend
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_unroll
.. autofunction:: chimerax.core.map.filter.vopcommand.volume_zone

vseries
=======
.. automodule:: chimerax.core.map.series.vseries_command
  :members:
  :show-inheritance:

wait
====
.. autofunction:: chimerax.core.commands.wait.wait

windowsize
==========
.. autofunction:: chimerax.core.commands.windowsize.window_size

zoom
====
.. autofunction:: chimerax.core.commands.zoom.zoom

