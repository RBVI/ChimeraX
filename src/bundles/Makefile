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

#
# The FIRST_SUBDIRS subdirectories are built before REST_SUBDIRS.
# Because core.models depends on graphics and geometry those are
# installed first to test other bundles.
#
BUNDLE_BUILDER_SUBDIR = bundle_builder

FIRST_SUBDIRS = arrays geometry graphics

REST_SUBDIRS = addh align_algs alignment_hdrs alignments atom_search atomic \
	  basic_actions bild blastprotein bond_rot bug_reporter build_structure \
	  bumps buttons cage_builder \
	  cellpack chem_group clashes cmd_line \
	  color_globe connect_structure core_formats \
	  crosslinks crystal data_formats \
	  dicom dist_monitor dist_ui dssp exp_cmd file_history function_key \
	  gltf hbonds help_viewer hkcage ihm image_formats \
	  interfaces io label list_info log \
	  map map_data map_eraser map_filter map_fit map_series \
	  markers mask match_maker md_crds \
	  medical_toolbar meeting mlp model_series \
	  mmcif mmtf model_panel modeller mol2 morph mouse_modes movie \
	  nucleotides open_command pdb pick_blobs preset_mgr pubchem read_pbonds \
	  registration remote_control rest_server \
	  residue_fit rotamer_lib_mgr rotamer_libs save_command scheme_mgr \
	  sdf segger segment seq_view sim_matrices \
	  shell shortcuts show_attr sideview smiles \
	  smooth_lines spacenavigator std_commands \
	  stl storm struts surface swapaa swap_res \
	  tape_measure test texture toolbar tug ui uniprot \
	  viewdockx vive volume_menu vtk wavefront_obj webservices zone

SUBDIRS = $(BUNDLE_BUILDER_SUBDIR) $(FIRST_SUBDIRS) $(REST_SUBDIRS)

TWO_PASS_INSTALL = 1

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax

dir:	Makefile.dependencies
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_INSTALL): dir

# Make sure bundle_builder is installed before all others.
$(FIRST_SUBDIRS:=.nodep-install): bundle_builder.nodep-install
$(FIRST_SUBDIRS:=.dep-install): bundle_builder.dep-install

# Make sure first subidirectories are built before the remainder.
$(REST_SUBDIRS:=.nodep-install): $(FIRST_SUBDIRS:=.nodep-install)
$(REST_SUBDIRS:=.dep-install): $(FIRST_SUBDIRS:=.dep-install)

lint:
ifneq (,$(PYSRCS))
	$(PYLINT) $(PYSRCS)
endif

clean:
	rm -rf __pycache__ Makefile.dependencies

Makefile.dependencies: */bundle_info.xml* Make_dependencies.py
ifneq ($(MAKECMDGOALS),clean)
	$(APP_PYTHON_EXE) Make_dependencies.py

# Add any custom dependencies that are not captured in bundle_info.xml files
# Segger bundle_info.xml is hidden in a tar file so Make_dependencies.py does not see it.
segger.nodep-install: geometry.nodep-install graphics.nodep-install ui.nodep-install
segger.dep-install: geometry.dep-install graphics.dep-install ui.dep-install

rotamer_libs.nodep-install: rotamer_lib_mgr.nodep-install
rotamer_libs.dep-install: rotamer_lib_mgr.dep-install

include Makefile.dependencies
endif
