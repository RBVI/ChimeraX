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
SUBDIR_SYNC = 1
include $(TOP)/mk/config.make

SUBDIRS = addh align_algs alignment_hdrs alignments atom_search atomic \
	  basic_actions bild blastprotein bond_rot bug_reporter bumps \
	  bundle_builder buttons cage_builder cellpack chem_group clashes cmd_line \
	  color_globe connect_structure crosslinks crystal \
	  dist_monitor dist_ui dssp exp_cmd file_history \
	  gltf hbonds help_viewer ihm interfaces label list_info log \
	  map map_data map_filter map_fit map_series \
	  markers mask match_maker md_crds meeting mlp model_series \
	  mmcif mmtf model_panel modeller mol2 morph mouse_modes movie \
	  nucleotides pdb read_pbonds \
	  registration remote_control rest_server \
	  residue_fit seq_view sim_matrices shell shortcuts sideview \
	  smooth_lines spacenavigator std_commands stl storm struts surface swapaa \
	  test texture tug ui viewdockx vive vtk wavefront_obj webservices zone

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax

dir:	Makefile.dependencies
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir bundle_builder.install

lint:
ifneq (,$(PYSRCS))
	$(PYLINT) $(PYSRCS)
endif

clean:
	rm -rf __pycache__ Makefile.dependencies

Makefile.dependencies: */bundle_info.xml*
ifneq ($(MAKECMDGOALS),clean)
	$(APP_PYTHON_EXE) Make_dependencies.py

include Makefile.dependencies
endif
