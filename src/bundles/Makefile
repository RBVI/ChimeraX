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
SUBDIR_BUILD = 1
include $(TOP)/mk/config.make

SYNC_DIR = $(wildcard $(TOP)/build/sync)

BUNDLE_BUILDER_SUBDIR = bundle_builder
TOOLSHED_UTILS_SUBDIR = toolshed_utils

REST_SUBDIRS = add_charge addh align_algs alignment_hdrs \
	  alignments alphafold altloc_explorer amber_info arrays atom_search \
	  atom_search_lib atomic atomic_lib axes_planes basic_actions bild \
	  blastprotein bond_rot bug_reporter build_structure \
	  bumps buttons cage_builder \
	  cellpack centroids chem_group clashes cmd_line color_actions \
	  color_globe color_key connect_structure core_formats \
	  coulombic crosslinks crystal crystal_contacts data_formats \
	  dicom dist_monitor dist_ui dssp \
	  exp_cmd file_history function_key \
	  geometry gltf graphics \
	  hbonds help_viewer hkcage ihm image_formats imod \
	  interfaces items_inspection io label list_info log looking_glass maestro \
	  map map_data map_eraser map_filter map_fit map_series \
	  markers mask match_align match_maker md_crds \
	  medical_toolbar meeting mlp model_series \
	  mmcif mmtf model_panel modeller mol2 morph mouse_modes movie \
	  neuron nucleotides open_command \
	  pdb pdb_bio pdb_lib pdb_matrices pick_blobs positions preset_mgr pubchem \
	  read_pbonds registration remote_control rest_server \
	  residue_fit rna_layout rotamer_lib_mgr rotamer_libs save_command scheme_mgr \
	  sdf sel_inspector segger segment seq_view sequences shape \
	  shell shortcuts show_attr sideview sim_matrices smiles \
	  smooth_lines spacenavigator std_commands \
	  stl storm struts surface swapaa swap_res \
	  tape_measure test toolbar tug ui uniprot unit_cell \
	  viewdockx vive volume_menu vtk wavefront_obj \
	  webcam webservices zone

ifneq ($(OSARCH),DarwinArm64)
# emdb_sff depends on h5py which is not available for Mac arm64 (July 2021)
REST_SUBDIRS += emdb_sff
endif

ifeq ($(OS),Linux)
REST_SUBDIRS += linux_support
endif

SUBDIRS = $(BUNDLE_BUILDER_SUBDIR) $(TOOLSHED_UTILS_SUBDIR) $(REST_SUBDIRS)

NO_SUBDIR_INSTALL = 1

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax

dir:	Makefile.dependencies
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_BUILD): dir

install:
	$(MAKE) build
	cd $(SYNC_DIR) && $(APP_BOOTSTRAP_PIP) install --upgrade --upgrade-strategy only-if-needed */*.whl
	$(APP_EXE) --exit --nogui --silent --cmd "toolshed reload installed"

lint:
ifneq (,$(PYSRCS))
	$(PYLINT) $(PYSRCS)
endif

clean:
	rm -rf __pycache__ Makefile.dependencies
ifneq (,$(wildcard $(SYNC_DIR)))
	cd $(SYNC_DIR) && rm -f */*.whl
endif

#
# Build Time Dependencies
# -----------------------
#
# Most bundles have no build time dependencies, since only the wheel is
# built, and the declared dependencies are resolved later during a "pip
# install" of all of the wheels.  This allows for circular installation
# dependencies.
#
# But some bundles need other bundles built and installed before they can be
# built (usually for including C++ header files and linking).  Two things have
# to be done for that to work: (1) the bundle needs to declare the build
# dependency by adding build="true" to the Dependency tag on that other bundle
# in their bundle_info.xml file.  And (2), the other bundle needs to put
# BUILD_AND_INSTALL=1 in their Makefile before including ../Makefile.bundle.
#
# The build time dependencies are automatically updated by scanning the bundle
# bundle_info.xml files, and are placed in to the Makefile.dependencies file.
#

Makefile.dependencies: */bundle_info.xml* Make_dependencies.py
ifneq ($(MAKECMDGOALS),clean)
	$(APP_PYTHON_EXE) Make_dependencies.py

# Make sure bundle_builder and toolshed bundles are installed before all others.
$(REST_SUBDIRS:=.build): $(BUNDLE_BUILDER_SUBDIR).build $(TOOLSHED_UTILS_SUBDIR).build

include Makefile.dependencies
endif
