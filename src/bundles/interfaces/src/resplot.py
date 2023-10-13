# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# ------------------------------------------------------------------------------
#
from .graph import Graph
class ResiduePlot(Graph):

    help = 'help:user/commands/contacts.html#residue-plot'
    
    def __init__(self, session, contact, flip = False, interface_residue_area_cutoff = 15):

        self.contact = c = contact

        # Interface residues
        g1, g2 = (c.group2, c.group1) if flip else (c.group1, c.group2)
        min_area = interface_residue_area_cutoff
        self.residues1 = r1 = c.contact_residues(g1, min_area)
        self.residues2 = r2 = c.contact_residues(g2, min_area)
        from chimerax.atomic import concatenate
        self.residues = res = concatenate((r1, r2))

        # Non-interface residues
        self.atoms1, self.atoms2 = a1, a2 = g1.atoms, g2.atoms
        self.noninterface_residues1 = a1.unique_residues.subtract(r1)
        self.noninterface_residues2 = a2.unique_residues.subtract(r2)
        allres = concatenate((a1,a2)).unique_residues
        self.noninterface_residues = allres.subtract(res)
        
        # Create matplotlib panel
        nodes = tuple(ResidueNode(r) for r in res)
        bnodes = tuple(ResidueNode(r, size=800, color=(.7,.7,.7,1), background=True)
                       for r in self.noninterface_residues1)
        edges = ()
        title = '%s %d residues with %s %d residues' % (g2.name, len(r2), g1.name, len(r1))
        Graph.__init__(self, session, nodes+bnodes, edges, "Chain Contacts", title = title)
        self.font_size = 8

        self._interface_shown = False
        self.draw_graph()

    def layout_projection(self):
        c = self.contact
        r1, r2 = (self.residues1, self.residues2)
        xyz1, xyz2 = [r.atoms.scene_coords.mean(axis = 0) for r in (r1,r2)]
        zaxis = xyz2 - xyz1
        center = 0.5 * (xyz1 + xyz2)
        from chimerax.geometry import orthonormal_frame
        f = orthonormal_frame(zaxis, origin = center)
        finv = f.inverse()
        return finv
    
    def mouse_click(self, node, event):
        if node:
            self._select_residue(node.residue)
            if not self._interface_shown:
                self._show_interface()
        else:
            self._clear_selection()

    def fill_context_menu(self, menu, rnode):
        r = rnode.residue if rnode else None
        add = lambda *args: self.add_menu_entry(menu, *args)
        add('Show Interface', self._show_interface)
        if r:
            add('Select Residue', self._select_residue, r)
        add('Save Plot As...', self.save_plot_as)

    def _residue_name(self, r):
        return '%s %d' % (r.name, r.number)
    
    def _show_interface(self):
        aa1 = self.atoms1
        aa1.displays = True
        aa1.draw_modes = aa1.SPHERE_STYLE
        gray = (180,180,180,255)
        self.noninterface_residues1.atoms.colors = gray
        self.noninterface_residues2.atoms.displays = False
        a2 = self.residues2.atoms
        a2.displays = True
        a2.draw_modes = a2.STICK_STYLE
        v = self._session().main_view
        v.camera.position = self.layout_projection().inverse()
        v.view_all(aa1.scene_bounds)
        self._interface_shown = True
    
    def _select_residue(self, r):
        self._clear_selection()
        r.atoms.selected = True
        r.atoms.intra_bonds.selected = True
            
    def _clear_selection(self):
        self._session().selection.clear()

# ------------------------------------------------------------------------------
#
from .graph import Node
class ResidueNode(Node):
    def __init__(self, residue, size=400, color=None, background=False):
        self.residue = residue
        self.size = size 	# Node size on plot in pixel area.
        self.name = '%d' % residue.number
        self._color = color
        self.background = background
    @property
    def position(self):
        return self.residue.atoms.scene_coords.mean(axis = 0)
    @property
    def color(self):
        c = self._color
        return tuple(r255/255 for r255 in self.residue.ribbon_color) if c is None else c
