# vim: set expandtab ts=4 sw=4:

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

# ------------------------------------------------------------------------------
#
from .graph import Graph
class ContactPlot(Graph):

    help = 'help:user/commands/contacts.html#diagram'
    
    def __init__(self, session, groups, contacts, interface_residue_area_cutoff = 5):

        # Create matplotlib panel
        title = '%d Chains %d Contacts' % (len(groups), len(contacts))
        Graph.__init__(self, session, groups, contacts,
                       tool_name = "Chain Contacts", title = title)

        self.groups = groups
        self.contacts = contacts
        self.interface_residue_area_cutoff = interface_residue_area_cutoff

        # Show contact areas less than half maximum as dotted lines.
        small_area = 0.5*max(c.buried_area for c in contacts)
        for c in contacts:
            if c.buried_area < small_area:
                c.style = 'dotted'
                c.width = 2

        self.draw_graph()

        # When group is undisplayed update its node color.
        from chimerax import atomic
        self._handler = atomic.get_triggers(session).add_handler('changes', self._atom_display_change)
        
    def delete(self):
        from chimerax import atomic
        atomic.get_triggers(self._session()).remove_handler(self._handler)
        self._handler = None
        Graph.delete(self)

    def mouse_click(self, item, event):
        nodes = self.item_nodes(item)
        if self.is_ctrl_key_pressed(event):
            self._select_nodes(nodes)
        else:
            n = len(nodes)
            if n == 1:
                self._show_neighbors(nodes[0])
            elif n > 1:
                # Edge clicked, pair of nodes
                self._show_node(nodes)

    def fill_context_menu(self, menu, item):
        nodes = self.item_nodes(item)
        node_names = ' and '.join(n.name for n in nodes)
        nn = len(nodes)

        add = lambda *args: self.add_menu_entry(menu, *args)

        # Show/hide menu entries
        if nodes:
            add('Show Only %s' % node_names, self._show_node, nodes)

        if len(nodes) == 1:
            add('Show %s and Neighbors' % node_names, self._show_neighbors, nodes[0])
            add('Show Contact Residues of Neighbors with %s' % node_names,
                self._show_contact_residues, nodes[0])

        from .cmd import Contact, SphereGroup
        if isinstance(item, Contact):
            c = item
            n1, n2 = (c.group1.name, c.group2.name)
            add('Show Contact Residues of %s with %s' % (n1,n2),
                self._show_interface_residues, c, c.group1)
            add('Show Contact Residues of %s with %s' % (n2,n1),
                self._show_interface_residues, c, c.group2)

        if item is None:
            add('Show All', self._show_all)

        if isinstance(item, Contact):
            flip = True
            add('Residue Plot %s with %s' % (n1,n2), self._show_residue_plot, item, flip)
            add('Residue Plot %s with %s' % (n2,n1), self._show_residue_plot, item)
        
        menu.addSeparator()
        
        # Selection menu entries
        if nodes:
            add('Select %s' % node_names, self._select_nodes, nodes)

        if nn == 1:
            add('Select Neighbors of %s' % node_names, self._select_neighbors, nodes[0])

        if nn == 0:
            clist = self.contacts
            stext = 'Select All Contact Residues'
        elif nn == 1:
            clist = self._node_contacts(nodes[0])
            stext = 'Select Contact Residues of %s and Neighbors' % node_names
        elif nn == 2:
            clist = [item]
            stext = 'Select Contact Residues of %s' % node_names
        add(stext, self._select_contact_residues, clist)

        if isinstance(item, Contact):
            c = item
            n1, n2 = (c.group1.name, c.group2.name)
            add('Select Contact Residues of %s with %s' % (n1,n2),
                self._select_contact_residues, clist, c.group1)
            add('Select Contact Residues of %s with %s' % (n2,n1),
                self._select_contact_residues, clist, c.group2)

        if item is None:
            add('Select All', self._select_nodes, self.groups)
            add('Clear Selection', self._clear_selection)

        menu.addSeparator()

        eargs = ()
        if isinstance(item, Contact):
            explode = item.explode_contact
            ewhat = node_names
        elif isinstance(item, SphereGroup):
            explode = self._explode_neighbors
            eargs = (item,)
            ewhat = 'Neighbors of %s' % node_names
        else:
            explode = self._explode_all
            ewhat = 'All'
        add('Explode ' + ewhat, explode, *eargs)
        add('Unexplode ' + ewhat, self._unexplode_all)

        menu.addSeparator()

        if item is None:
            add('Lay Out to Match Structure', self.draw_graph)
            add('Orient Structure to Match Layout', self._orient)
                    
    def _select_nodes(self, nodes):
        self._clear_selection()
        for g in nodes:
            g.atoms.selected = True

    def _select_neighbors(self, g):
        self._clear_selection()
        from .cmd import neighbors
        for n in neighbors(g, self.contacts):
            n.atoms.selected = True

    def _select_contact_residues(self, contacts, group = None):
        self._clear_selection()
        min_area = self.interface_residue_area_cutoff
        for c in contacts:
            for g in (c.group1, c.group2):
                if g is group or group is None:
                    atoms = c.contact_residue_atoms(g, min_area)
                    atoms.selected = True
            
    def _clear_selection(self):
        self._session().selection.clear()

    def _node_contacts(self, n):
        from .cmd import neighbors
        nc = neighbors(n, self.contacts)	# Map neighbor node to Contact
        return tuple(nc.values())
        
    def _show_node(self, nodes):
        gset = set(nodes)
        for h in self.groups:
            h.show(h in gset)

    def _show_neighbors(self, g):
        from .cmd import neighbors
        ng = neighbors(g, self.contacts)
        ng[g] = None
        for h in self.groups:
            h.show(h in ng)

    def _show_contact_residues(self, g, color = (180,180,180,255)):
        from .cmd import neighbors
        ng = neighbors(g, self.contacts)	# Map neighbor node to Contact
        min_area = self.interface_residue_area_cutoff
        gca = []
        for h in self.groups:
            if h in ng:
                c = ng[h]
                atoms = c.contact_residue_atoms(h, min_area)
                h.show(False)
                atoms.displays = True	# Show only contacting residues
                atoms.draw_modes = atoms.STICK_STYLE
                gatoms = c.contact_residue_atoms(g, min_area)
#                gatoms.draw_modes = gatoms.STICK_STYLE
                gca.append(gatoms)
            else:
                h.show(h is g)
        # Color non-contact atoms gray.
        from chimerax.atomic import concatenate, Atoms
        g.color_atoms(g.atoms - concatenate(gca,Atoms), color)

    def _show_interface_residues(self, c, g, color = (180,180,180,255)):
        for go in self.groups:
            go.show(False)
            
        g1, g2 = c.group1, c.group2
        gf, gb = (g1,g2) if g is g1 else (g2,g1)
        min_area = self.interface_residue_area_cutoff
        af = c.contact_residue_atoms(gf, min_area)
        ab = c.contact_residue_atoms(gb, min_area)

        af.displays = True	# Show only contacting residues
        af.draw_modes = af.STICK_STYLE
        gf.restore_atom_colors()

        allb = gb.atoms
        allb.displays = True
        allb.draw_modes = allb.SPHERE_STYLE
        gb.color_atoms(allb - ab, color)

        v = self._session().main_view
        v.camera.position = c.interface_frame(gb)
        v.view_all(allb.scene_bounds)

    def _show_all(self):
        for g in self.groups:
            g.show(True)

    def _show_residue_plot(self, c, flip=False):
        g = c.group1 if flip else c.group2
        self._show_interface_residues(c, g)
        from .resplot import ResiduePlot
        ResiduePlot(self._session(), c, flip, self.interface_residue_area_cutoff)
        
    def _explode_all(self, scale = 2):
        gc = [(g,g.centroid()) for g in self.groups if g.shown()]
        if len(gc) < 2:
            return
        from numpy import mean
        center = mean([c for g,c in gc], axis = 0)
        for g,c in gc:
            g.move((scale-1)*(c - center))

    def _unexplode_all(self):
        for g in self.groups:
            g.unmove()

    def _explode_neighbors(self, n):
        for c in self._node_contacts(n):
            g = c.group2 if n is c.group1 else c.group1
            c.explode_contact(move_group = g)

    def _orient(self):
        gc = []
        la = []
        for g,(x,y) in zip(self.groups, self._layout_positions):
            if g.shown():
                gc.append(g.centroid())
                la.append((x,y,0))
        if len(gc) < 2:
            return
        from chimerax.geometry import align_points, translation
        from numpy import array, mean
        p, rms = align_points(array(gc), array(la))
        ra = p.zero_translation()
        center = mean(gc, axis=0)
        v = self._session().main_view
        rc = v.camera.position.zero_translation()
        rot = translation(center) * rc * ra * translation(-center)
        v.move(rot)
        
    def _atom_display_change(self, name, changes):
        if 'display changed' in changes.atom_reasons():
            # Atoms shown or hidden.  Color hidden nodes gray.
            self.recolor_nodes()

        # Close plot if any of its atoms were deleted since
        # the contacts are then invalid, and Contact index lists
        # are now incorrect.
        if changes.num_deleted_atoms() > 0:
            for g in self.groups:
                if g.atoms_deleted:
                    self.delete()
                    return
