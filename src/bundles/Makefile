# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

TOP = ../..
-include $(TOP)/.makerc
SUBDIR_SYNC = 1
SUBDIR_BUILD = 1
include $(TOP)/mk/config.make
PYTHON ?= $(APP_PYTHON_EXE)

SYNC_DIR = $(wildcard $(TOP)/build/sync)

REST_SUBDIRS = add_charge addh alignment_algs alignment_headers \
	  alphafold altloc_explorer amber_info aniso arrays atom_search \
	  atomic atomic_lib axes_planes basic_actions bild \
	  blastprotein boltz bond_rot bug_reporter build_structure \
	  bumps buttons cage_builder \
	  cellpack centroids change_chains check_waters chem_group \
	  clashes cmd_line color_actions \
	  color_globe color_key connect_structure core_formats \
	  coulombic crosslinks crystal crystal_contacts data_formats \
	  dicom dist_monitor dock_prep dssp \
	  emdb_sff esmfold file_history function_key \
	  geometry gltf graphics \
	  hbonds help_viewer hkcage ihm image_formats imod \
	  interfaces items_inspection io label iupac \
	  kvfinder list_info log looking_glass maestro \
	  map map_data map_eraser map_filter map_fit map_series \
	  markers mask match_maker mcopy md_crds \
	  medical_toolbar meeting minimize mlp model_series \
	  mmcif mmtf model_archive model_panel modeller \
	  mol2 mole morph mouse_modes movie mutation_scores nmrstar \
	  neuron nifti nrrd nucleotides open_command orthopick pdb pdb_bio pdb_lib pdb_matrices \
	  pick_blobs positions preset_mgr profile_grids pubchem read_pbonds registration \
	  remote_control render_by_attr renumber_residues rest_server \
	  residue_fit rna_layout rotamers rotamer_libs \
	  save_command scheme_mgr \
	  sdf sel_inspector segger segment segmentations seqalign seq_view shape \
	  shell shortcuts show_sequences sideview sim_matrices similarstructures smiles \
	  smooth_lines spacenavigator std_commands \
	  stl storm struct_measure struts surface swapaa swap_res \
	  tape_measure task_manager test toolbar topography to_quest tug ui umap uniprot \
	  unit_cell viewdock viperdb vive volume_menu vrml vtk wavefront_obj \
	  webcam webservices zone

ifeq ($(OS),Linux)
REST_SUBDIRS += linux_support
endif

SUBDIRS = $(REST_SUBDIRS)

NO_SUBDIR_INSTALL = 1

include $(TOP)/mk/subdir.make

PKG_DIR = $(PYSITEDIR)/chimerax

dir:	Makefile.dependencies
	if [ ! -d "$(PKG_DIR)" ]; then mkdir -p $(PKG_DIR); fi

$(SUBDIR_BUILD): dir

install: 
	$(MAKE) -C bundle_builder build
	$(MAKE) -C core build
	$(MAKE) -C toolshed_utils build
	$(MAKE) build
	cd $(SYNC_DIR) && $(APP_BOOTSTRAP_PIP) install --upgrade --upgrade-strategy only-if-needed *.whl
	$(APP_PYTHON_EXE) -m chimerax.core --exit --nogui --silent --cmd "toolshed reload installed"

uv-install:
	PYTHON=python $(MAKE) -C bundle_builder uv-build
	PYTHON=python $(MAKE) -C core uv-build
	PYTHON=python $(MAKE) -C toolshed_utils uv-build
	PYTHON=python $(MAKE) uv-build
	cd $(SYNC_DIR) && uv pip install *.whl
	python -m chimerax.core --exit --nogui --silent --cmd "toolshed reload installed"

lint:
ifneq (,$(PYSRCS))
	$(PYLINT) $(PYSRCS)
endif

clean:
	$(MAKE) -C core clean
	$(MAKE) -C bundle_builder clean
	$(MAKE) -C toolshed_utils clean
	rm -rf __pycache__ Makefile.dependencies

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

Makefile.dependencies: */bundle_info.xml* */pyproject.toml* Make_dependencies.py
ifneq ($(MAKECMDGOALS),clean)
	$(PYTHON) Make_dependencies.py

# Make sure bundle_builder and toolshed bundles are installed before all others.
$(REST_SUBDIRS:=.build):

include Makefile.dependencies
endif
