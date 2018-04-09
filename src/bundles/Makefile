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

SUBDIRS = addh align_algs alignments bild blastprotein bond_rot bumps \
	  bundle_builder cage_builder cellpack chem_group clashes cmd_line \
	  color_globe crosslinks dist_monitor dist_ui exp_cmd file_history \
	  gltf hbonds help_viewer ihm interfaces label list_info \
	  log map_series_gui markers match_maker md_crds meeting mlp model_series \
	  mmtf model_panel mol2 morph mouse_modes movie \
	  nucleotides \
	  registration remote_control rest_server \
	  residue_fit seq_view sim_matrices shell shortcuts sideview \
	  smooth_lines spacenavigator stl storm struts swapaa target \
	  test texture tug ui viewdockx vive volume_viewer vtk wavefront_obj

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax

dir:
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir bundle_builder.install

lint:
ifneq (,$(PYSRCS))
	$(PYLINT) $(PYSRCS)
endif

clean:
	rm -rf __pycache__

align_algs.install: alignments.install
bond_rot.install: ui.install
bumps.install: markers.install
crosslinks.install: interfaces.install
markers.install: ui.install
match_maker.install: align_algs.install sim_matrices.install
sim_matrices.install: alignments.install
