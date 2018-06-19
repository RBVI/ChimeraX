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

SUBDIRS = addh active_tools align_algs alignments atomic \
	  basic_actions bild blastprotein bond_rot bumps \
	  bundle_builder cage_builder cellpack chem_group clashes cmd_line \
	  color_globe crosslinks crystal dist_monitor dist_ui dssp exp_cmd file_history \
	  gltf hbonds help_viewer ihm interfaces label list_info log \
	  map map_data map_filter map_fit map_series \
	  markers match_maker md_crds meeting mlp model_series \
	  mmcif mmtf model_panel mol2 morph mouse_modes movie \
	  nucleotides pdb read_pbonds \
	  registration remote_control rest_server \
	  residue_fit seq_view sim_matrices shell shortcuts sideview \
	  smooth_lines spacenavigator std_commands stl storm struts surface swapaa \
	  test texture tug ui viewdockx vive vtk wavefront_obj

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

active_tools.install: ui.install surface.install
addh.install: atomic.install
align_algs.install: alignments.install
bild.install: atomic.install surface.install
bond_rot.install: ui.install
bumps.install: markers.install
chem_group.install: atomic.install
clashes.install: atomic.install
crosslinks.install: interfaces.install
dist_monitor.install: atomic.install
dssp.install: atomic.install
hbonds.install: atomic.install
ihm.install: atomic.install map.install
map.install: map_data.install
map_filter.install: map.install
map_fit.install: map.install
map_series.install: map.install
markers.install: atomic.install ui.install
match_maker.install: align_algs.install sim_matrices.install
md_crds.install: atomic.install
mmcif.install: atomic.install
mmtf-python.install: atomic.install
mol2.install: atomic.install
nucleotides.install: atomic.install
pdbio.install: atomic.install
sim_matrices.install: alignments.install
std_commands.install: atomic.install
viewdockx.install: atomic.install
