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

SUBDIRS = addh active_tools align_algs alignments atom_search atomic \
	  basic_actions bild blastprotein bond_rot bug_reporter bumps \
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
alignments.install: atomic.install
atom_search.install: atomic.install
bild.install: atomic.install surface.install
basic_actions.install: ui.install
blastprotein.install: ui.install
bond_rot.install: ui.install
bumps.install: markers.install
cage_builder.install: ui.install
chem_group.install: atomic.install
clashes.install: atomic.install
cmd_line.install: ui.install
crosslinks.install: interfaces.install
dist_monitor.install: atomic.install
dist_ui.install: ui.install
dssp.install: atomic.install
file_history.install: ui.install
help_viewer.install: ui.install
hbonds.install: atomic.install chem_group.install
ihm.install: atomic.install map.install
interfaces.install: ui.install
log.install: ui.install
map.install: map_data.install ui.install
map_filter.install: map.install
map_fit.install: map.install
map_series.install: map.install ui.install
markers.install: atomic.install ui.install
match_maker.install: align_algs.install sim_matrices.install
md_crds.install: atomic.install
mmcif.install: atomic.install
mmtf-python.install: atomic.install
model_panel.install: ui.install
model_series.install: ui.install
modeller.install: alignments.install
mol2.install: atomic.install
mouse_modes.install: ui.install
nucleotides.install: atomic.install ui.install
pdbio.install: atomic.install
read_pbonds.install: atomic.install
registration.install: ui.install
residue_fit.install: ui.install
seq_view.install: alignments.install ui.install
shell.install: ui.install
shortcuts.install: ui.install
sideview.install: ui.install
signal_viewer.install: ui.install
sim_matrices.install: alignments.install
std_commands.install: atomic.install
swapaa.install: ui.install
tug.install: ui.install
viewdockx.install: atomic.install mol2.install ui.install
