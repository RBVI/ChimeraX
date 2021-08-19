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

.. _user_commands:

Python Functions implementing User Commands
*******************************************

The *chimerax.std_commands* and *chimerax.core.commands* modules contain Python functions for the
standard commands available from the ChimeraX command-line.  Tools define additional commands noted with a (*)
and their code location is described with the function descriptions.

| `2dlabels`_ * - Show text labels with graphics
| `alias`_ - Define aliases for commands
| `align`_ - Align one set of atoms to another
| `bumps`_ * - Mark cell protrusions in 3D light microscopy
| `cage`_ * - Place models on polygonal cages
| `camera`_ - Change camera settings
| `cartoon`_ * - Display atomic structure ribbons or pipes/planks
| `cd`_ - Change working directory
| `clip`_ - Clip models
| `close`_ - Close models
| `cofr`_ - Set center of rotation method
| `color`_ - Color atoms, ribbons and surfaces
| `coordset`_ - Play through structure coordinate sets
| `crossfade`_ - Fade between scenes for movie making
| `crosslinks`_ * - Move atomic structures to minimize crosslinks
| `delete`_ - Delete models
| `devel`_ - For developers to package tools
| `device`_ * - Enable devices such virtual reality headsets and space navigator
| `distance`_ * - Show distances between atoms
| `echo`_ - Write message to log
| `exit`_ - Quit
| `fitmap`_ - fit atomic structures in density maps
| `graphics`_ - set graphics rendering parameters
| `help`_ - Show documentation
| `hide`_ - Hide atoms, ribbons, models
| `info`_ * - Report model info
| `interfaces`_ * - Make a buried area contact graph
| `ks`_ * - Enable keyboard shortcuts
| `label`_ * - Show text labels on atoms
| `lighting`_ - Change lighting parameters
| `log`_ * - Clear or save the log
| `material`_ - Change surface material light reflection properties
| `measure`_ - Calculate properties of models
| `mlp`_ * - Color surfaces by molecular lipophilicity
| `molmap`_ - Calculate a density map from atoms
| `morph`_ * - Morph between atomic models
| `move`_ - Move camera
| `movie`_ * - Record videos
| `mseries`_ * - Display a series of models
| `open`_ - Open data files
| `perframe`_ - Run a command for every rendered frame
| `pwd`_ - Print working directory
| `rainbow`_ - Color residues and chains
| `rename`_ - Rename models
| `resfit`_ * - Show density around residues
| `roll`_ - Rotate models
| `run`_ - Run a user command string
| `save`_ - Save sessions, images, density maps...
| `select`_ - Select objects
| `set`_ - Set rendering effects (background color, silhouettes)
| `show`_ - Show atoms, ribbons, models
| `style`_ - Set rendering style for atoms, ribbons, models
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
| `ui`_ * - Control tool windows and set mouse modes
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
.. autofunction:: chimerax.std_commands.alias.alias

align
=====
.. autofunction:: chimerax.std_commands.align.align

bumps
=====
.. autofunction:: chimerax.bumps.bumps.bumps

cage
====
.. autofunction:: chimerax.cage_builder.cmd.cage

camera
======
.. autofunction:: chimerax.std_commands.camera.camera

cartoon
=======
.. autofunction:: chimerax.std_commands.cartoon.cartoon
.. autofunction:: chimerax.std_commands.cartoon.uncartoon

cd
==
.. autofunction:: chimerax.std_commands.cd.cd

clip
=====
.. autofunction:: chimerax.std_commands.clip.clip

close
=====
.. autofunction:: chimerax.std_commands.close.close

cofr
====
.. autofunction:: chimerax.std_commands.cofr.cofr

color
=====
.. autofunction:: chimerax.std_commands.color.color
.. autofunction:: chimerax.surface.colorgeom.color_cylindrical
.. autofunction:: chimerax.surface.colorvol.color_electrostatic
.. autofunction:: chimerax.surface.colorvol.color_gradient
.. autofunction:: chimerax.surface.colorgeom.color_height
.. autofunction:: chimerax.surface.texture.color_image
.. autofunction:: chimerax.surface.colorgeom.color_radial
.. autofunction:: chimerax.surface.colorvol.color_sample
.. autofunction:: chimerax.surface.colorzone.color_zone
		  
coordset
========
.. autofunction:: chimerax.std_commands.coordset.coordset

crossfade
=========
.. autofunction:: chimerax.std_commands.crossfade.crossfade

crosslinks
==========
Function found in *chimerax.crosslinks.crosslinks*

.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks
.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks_histogram
.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks_minimize
.. autofunction:: chimerax.crosslinks.crosslinks.crosslinks_network

delete
======
.. autofunction:: chimerax.std_commands.delete.delete

devel
======
.. autofunction:: chimerax.core.commands.devel.devel_build
.. autofunction:: chimerax.core.commands.devel.devel_install
.. autofunction:: chimerax.core.commands.devel.devel_alias
.. autofunction:: chimerax.core.commands.devel.devel_unalias
.. autofunction:: chimerax.core.commands.devel.devel_clean
.. autofunction:: chimerax.core.commands.devel.devel_dump
		  
device
======
.. autofunction:: chimerax.spacenavigator.snav.device_snav
.. autofunction:: chimerax.vive.vr.vr
		  
distance
========
.. autofunction:: chimerax.dist_monitor.cmd.distance

echo
====
Equivalent to the "log text" command.

exit
====
.. autofunction:: chimerax.std_commands.exit.exit

fitmap
======
.. autofunction:: chimerax.map_fit.fitcmd.fitmap

graphics
========
.. autofunction:: chimerax.std_commands.graphics.graphics

help
====
.. autofunction:: chimerax.help_viewer.cmd.help

hide
====
.. autofunction:: chimerax.std_commands.hide.hide

info
====
.. automodule:: chimerax.list_info.cmd
  :members:
  :member-order: bysource

interfaces
==========
Function found in *chimerax.interfaces.cmd*

.. autofunction:: chimerax.interfaces.cmd.contacts

ks
==
.. autofunction:: chimerax.shortcuts.shortcuts.ks

label
=====
Function found in *chimerax.label.label3d*

.. autofunction:: chimerax.label.label3d.label
.. autofunction:: chimerax.label.label3d.label_delete

lighting
========
.. autofunction:: chimerax.std_commands.lighting.lighting

log
===
Function found in *chimerax.log.cmd*

.. autofunction:: chimerax.log.cmd.log

material
========
.. autofunction:: chimerax.std_commands.material.material

measure
=======
.. autofunction:: chimerax.std_commands.measure_buriedarea.measure_buriedarea
.. autofunction:: chimerax.std_commands.measure_convexity.measure_convexity
.. autofunction:: chimerax.std_commands.measure_length.measure_length
.. autofunction:: chimerax.surface.measure_sasacmd.measure_sasa

mlp
===
Function found in *chimerax.mlp.mlp*

.. autofunction:: chimerax.mlp.mlp.mlp

molmap
======
.. autofunction:: chimerax.map.molmap.molmap

morph
=====
Function defined in *chimerax.morph.morph*

.. autofunction:: chimerax.morph.morph.morph

move
====
.. autofunction:: chimerax.std_commands.move.move

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

.. _open_command:

open
====
Function defined in *chimerax.open_command.manager* and called via `session.open_command.open_data(...)`

.. autoclass:: chimerax.open_command.manager.OpenManager
	:members: open_data
	:noindex:

perframe
========
.. autofunction:: chimerax.std_commands.perframe.perframe

pwd
===
.. autofunction:: chimerax.std_commands.pwd.pwd

rainbow
=======
.. autofunction:: chimerax.std_commands.rainbow.rainbow

rename
======
.. autofunction:: chimerax.std_commands.rename.rename

resfit
======
.. autofunction:: chimerax.residue_fit.cmd.resfit

roll
====
.. autofunction:: chimerax.std_commands.roll.roll

run
===
.. autofunction:: chimerax.core.commands.run
    :noindex:

.. _save_command:

save
====
Function defined in *chimerax.save_command.manager* and called via `session.save_command.save_data(...)`

.. autoclass:: chimerax.save_command.manager.SaveManager
	:members: save_data
	:noindex:

select
======
.. autofunction:: chimerax.std_commands.select.select

set
===
.. autofunction:: chimerax.std_commands.set.set

show
====
.. autofunction:: chimerax.std_commands.show.show

size
=====
.. autofunction:: chimerax.std_commands.size.size

smoothlines
===========
.. autofunction:: chimerax.smooth_lines.smoothlines.smoothlines

split
=====
.. autofunction:: chimerax.std_commands.split.split

stop
====
.. autofunction:: chimerax.std_commands.stop.stop

struts
======
.. autofunction:: chimerax.struts.struts.struts
.. autofunction:: chimerax.struts.struts.struts_delete

style
=====
.. autofunction:: chimerax.std_commands.style.style

surface
=======
.. autofunction:: chimerax.surface.surfacecmds.surface
.. autofunction:: chimerax.surface.surfacecmds.surface_close
.. autofunction:: chimerax.surface.surfacecmds.surface_hide_patches
.. autofunction:: chimerax.surface.surfacecmds.surface_show_patches
.. autofunction:: chimerax.surface.surfacecmds.surface_style
.. autofunction:: chimerax.surface.surfacecmds.surface_cap
.. autofunction:: chimerax.surface.sop.surface_dust
.. autofunction:: chimerax.surface.sop.surface_undust
.. autofunction:: chimerax.surface.sop.surface_zone
.. autofunction:: chimerax.surface.sop.surface_unzone

sym
===
.. autofunction:: chimerax.std_commands.sym.sym

time
====
.. autofunction:: chimerax.std_commands.time.time

toolshed
========
.. autofunction:: chimerax.core.commands.toolshed.toolshed_list
.. autofunction:: chimerax.core.commands.toolshed.toolshed_reload
.. autofunction:: chimerax.core.commands.toolshed.toolshed_install
.. autofunction:: chimerax.core.commands.toolshed.toolshed_uninstall
.. autofunction:: chimerax.core.commands.toolshed.toolshed_url

transparency
============
.. autofunction:: chimerax.std_commands.transparency.transparency

turn
====
.. autofunction:: chimerax.std_commands.turn.turn

ui
==
.. autofunction:: chimerax.ui.cmd.ui_autostart
  :noindex:
.. autofunction:: chimerax.ui.cmd.ui_dockable
  :noindex:                  
.. autofunction:: chimerax.mouse_modes.cmd.mousemode
.. autofunction:: chimerax.ui.cmd.ui_tool_show
  :noindex:                  
.. autofunction:: chimerax.ui.cmd.ui_tool_hide
  :noindex:                  


view
====
.. autofunction:: chimerax.std_commands.view.view
.. autofunction:: chimerax.std_commands.view.view_delete
.. autofunction:: chimerax.std_commands.view.view_initial
.. autofunction:: chimerax.std_commands.view.view_list
.. autofunction:: chimerax.std_commands.view.view_matrix
.. autofunction:: chimerax.std_commands.view.view_name

volume
======
.. autofunction:: chimerax.map.volumecommand.volume
.. autofunction:: chimerax.map_filter.vopcommand.volume_add
.. autofunction:: chimerax.map_filter.vopcommand.volume_bin
.. autofunction:: chimerax.map_filter.vopcommand.volume_boxes
.. autofunction:: chimerax.map_filter.vopcommand.volume_copy
.. autofunction:: chimerax.map_filter.vopcommand.volume_cover
.. autofunction:: chimerax.map_eraser.eraser.volume_erase
.. autofunction:: chimerax.map_filter.vopcommand.volume_falloff
.. autofunction:: chimerax.map_filter.vopcommand.volume_flatten
.. autofunction:: chimerax.map_filter.vopcommand.volume_flip
.. autofunction:: chimerax.map_filter.vopcommand.volume_fourier
.. autofunction:: chimerax.map_filter.vopcommand.volume_gaussian
.. autofunction:: chimerax.map_filter.vopcommand.volume_laplacian
.. autofunction:: chimerax.map_filter.vopcommand.volume_local_correlation
.. autofunction:: chimerax.mask.maskcommand.mask
.. autofunction:: chimerax.map_filter.vopcommand.volume_maximum
.. autofunction:: chimerax.map_filter.vopcommand.volume_median
.. autofunction:: chimerax.map_filter.vopcommand.volume_minimum
.. autofunction:: chimerax.map_filter.vopcommand.volume_morph
.. autofunction:: chimerax.map_filter.vopcommand.volume_multiply
.. autofunction:: chimerax.map_filter.vopcommand.volume_new
.. autofunction:: chimerax.map_filter.vopcommand.volume_octant
.. autofunction:: chimerax.mask.maskcommand.ones_mask
.. autofunction:: chimerax.map_filter.vopcommand.volume_permute_axes
.. autofunction:: chimerax.map_filter.vopcommand.volume_resample
.. autofunction:: chimerax.map_filter.vopcommand.volume_ridges
.. autofunction:: chimerax.map_filter.vopcommand.volume_scale
.. autofunction:: chimerax.surface.colorzone.split_volume_by_color_zone
.. autofunction:: chimerax.map_filter.vopcommand.volume_subtract
.. autofunction:: chimerax.map_filter.vopcommand.volume_threshold
.. autofunction:: chimerax.map_filter.vopcommand.volume_tile
.. autofunction:: chimerax.map_filter.vopcommand.volume_unbend
.. autofunction:: chimerax.map_filter.vopcommand.volume_unroll
.. autofunction:: chimerax.map_filter.vopcommand.volume_unzone
.. autofunction:: chimerax.map_filter.vopcommand.volume_zone

vseries
=======
.. automodule:: chimerax.map_series.vseries_command
  :members:
  :show-inheritance:

wait
====
.. autofunction:: chimerax.std_commands.wait.wait

windowsize
==========
.. autofunction:: chimerax.std_commands.windowsize.window_size

zoom
====
.. autofunction:: chimerax.std_commands.zoom.zoom

