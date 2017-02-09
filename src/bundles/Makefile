# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS = add_missing_atoms align_algs alignments blastprotein \
	  cage_builder cellpack chem_group cmd_line contacts crosslinks \
		cytoscape \
	  exp_cmd file_history help_viewer ihm label list_info \
	  log map_series_gui match_maker mav mlp mmtf model_panel \
	  morph mouse_modes movie remote_control rest_server \
	  residue_fit sim_matrices shell shortcuts sideview \
	  smooth_lines spacenavigator stl storm stringdb struts \
	  test toolshed tug vive volume_viewer vtk

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir

lint:
ifneq (,$(PYSRCS))
	$(PYLINT) $(PYSRCS)
endif

clean:
	rm -rf __pycache__

align_algs.install: alignments.install
sim_matrices.install: alignments.install
match_maker.install: align_algs.install sim_matrices.install
