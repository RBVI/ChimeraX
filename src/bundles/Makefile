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
	  alphafold altloc_explorer amber_info animations aniso arrays atom_search \
	  atomic atomic_lib axes_planes basic_actions bild \
	  blastprotein boltz bond_rot bug_reporter build_structure \
	  bumps buttons cage_builder \
	  cellpack centroids change_chains check_waters chem_group \
	  clashes cluster cmd_line color_actions \
	  color_globe color_key connect_structure core_formats \
	  coulombic crosslinks crystal crystal_contacts data_formats \
	  dist_monitor dock_prep dssp \
	  emdb_sff esmfold file_history function_key \
	  geometry gltf graphics \
	  hbonds help_viewer hkcage ihm image_formats imod \
	  interfaces items_inspection io label iupac \
	  kvfinder lighting_gui list_info log looking_glass maestro \
	  map map_data map_eraser map_filter map_fit map_series \
	  markers mask match_align match_maker mcopy mcp_server md_crds \
	  meeting minimize mlp model_series \
	  mmcif model_archive model_panel modeller \
	  mol2 mole morph mouse_modes movie mutation_scores nmrstar \
	  neuron nucleotides open_command openfold orthopick \
	  pdb pdb_bio pdb_lib pdb_matrices \
	  pick_blobs positions preset_mgr profile_grids pubchem read_pbonds registration \
	  remote_control render_by_attr renumber_residues rest_server \
	  residue_fit rna_layout rotamers rotamer_libs \
	  save_command scenes scheme_mgr \
	  sdf sel_inspector segger segment segmentations seqalign seq_view shape \
	  shell shortcuts show_sequences sideview sim_matrices similarstructures smiles \
	  smooth_lines snfg spacenavigator std_commands \
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
	$(MAKE) phased-build
ifdef APP_FAST_PIP_INSTALL
	cd $(SYNC_DIR) && $(APP_FAST_PIP_INSTALL) *.whl
else
	cd $(SYNC_DIR) && $(APP_BOOTSTRAP_PIP) install *.whl
endif
ifndef DEFER_TOOLSHED_RELOAD
	$(APP_PYTHON_EXE) -m chimerax.core --exit --nogui --silent --cmd "toolshed reload installed"
endif

# Same as "build", but split into two sub-makes that share the jobserver, so
# the total number of jobs is unchanged:
#
#   py.phase  Pure Python bundles.  Wheels are built without ChimeraX, so these
#             only need bundle_builder and run alongside core, a few at a
#             time.
#   c.phase   core, then the bundles with compiled code.  When a job finishes,
#             make resumes its walk through the prerequisite list where it left
#             off and only revisits targets that were waiting on a dependency
#             after reaching the end.  In one long list with the pure Python
#             bundles, the arrays -> atomic_lib -> pdb_lib -> mmcif chain waits
#             behind every other bundle; a short list of its own is rescanned
#             quickly.  The parts of bundles that need no other bundle
#             (PREBUILD_SUBDIRS) start right away, alongside core; then the
#             longest chains are listed first, and bundles that are not on one
#             (CRITICAL_SUBDIRS) are kept out of their way (see the c.phase
#             rules at the end).
#
# toolshed_utils only needs setuptools, so it also runs alongside core.
PREBUILD_SUBDIRS = md_crds mmcif
CHAIN_SUBDIRS = $(filter $(SUBDIRS),$(BUILD_DEP_SUBDIRS))
C_REST_SUBDIRS = $(filter-out $(CHAIN_SUBDIRS),$(filter $(SUBDIRS),$(C_SUBDIRS)))
C_BUILD_SUBDIRS = $(CHAIN_SUBDIRS) $(C_REST_SUBDIRS)
PY_SUBDIRS = $(filter-out $(C_BUILD_SUBDIRS) $(C_SUBDIRS),$(SUBDIRS))
.PHONY: phased-build c.phase py.phase py.batch core.build toolshed_utils.build \
	$(PREBUILD_SUBDIRS:=.prebuild)
phased-build: c.phase py.phase toolshed_utils.build
# c.phase is not limited by job slots: the low priority bundles hold theirs
# for a long time, and a critical bundle must not wait behind them for one.
# There are only a few dozen bundles, and each limits its own compiles.
c.phase:
	$(MAKE) -j PHASED_C_BUILD=1 core.build \
		$(PREBUILD_SUBDIRS:=.prebuild) $(C_BUILD_SUBDIRS:=.build)
# Each pure Python bundle takes a fraction of a second, and they have plenty of
# time, so a few at a time is enough and leaves the cores to the C bundles
PY_PHASE_JOBS = 6
# Most of that fraction of a second is importing setuptools, so those whose
# Makefile adds nothing to Makefile.bundle are built by one bundle_builder,
# which imports it once and forks a child for each.  Windows has no fork.
ifndef WIN32
PY_BATCH_SUBDIRS = $(filter $(PLAIN_SUBDIRS),$(PY_SUBDIRS))
endif
PY_MAKE_SUBDIRS = $(filter-out $(PY_BATCH_SUBDIRS),$(PY_SUBDIRS))
py.phase:
	$(MAKE) -j$(PY_PHASE_JOBS) $(PY_MAKE_SUBDIRS:=.build) \
		$(if $(PY_BATCH_SUBDIRS),py.batch)
py.batch:
	PYTHONNOUSERSITE=1 $(LOW_PRIORITY) $(APP_PYTHON_EXE) -m chimerax.bundle_builder \
		-j $(PY_PHASE_JOBS) $(if $(DEBUG),--debug) $(if $(RELEASE),--release) \
		--build $(PY_BATCH_SUBDIRS)
	cp -p $(PY_BATCH_SUBDIRS:=/dist/*.whl) $(SYNC_DIR)
core.build toolshed_utils.build:
	$(MAKE) -C $(subst .build,,$@) build
$(PREBUILD_SUBDIRS:=.prebuild):
	$(LOW_PRIORITY) $(MAKE) -C $(subst .prebuild,,$@) prebuild
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

# c.phase rules (see phased-build above).  They come after the include since
# the target lists use its variables.
ifdef PHASED_C_BUILD
# Only in c.phase, so that "make build" still does not rebuild core
$(addsuffix .build,$(filter $(CORE_DEP_SUBDIRS),$(C_BUILD_SUBDIRS))): core.build
$(foreach d,$(PREBUILD_SUBDIRS),$(eval $(d).build: $(d).prebuild))
# Bundles off the critical path: those that can start alongside its first
# steps have until its end, so they use the efficiency cores; the others wait
# for its step at their own depth (see Make_dependencies.py).  private, so the
# bundles these depend on do not inherit it.
$(addsuffix .build,$(filter $(EARLY_SUBDIRS),$(C_BUILD_SUBDIRS))): \
	private SUBDIR_BUILD_PREFIX = $(BACKGROUND_COMPILE_PRIORITY)
$(addsuffix .build,$(filter-out $(CRITICAL_SUBDIRS) $(EARLY_SUBDIRS),$(C_BUILD_SUBDIRS))): \
	private SUBDIR_BUILD_PREFIX = $(LOW_PRIORITY)
$(foreach w,$(LATE_WAITS),$(eval $(subst :,: ,$(w))))
endif
