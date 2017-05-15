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
from chimerax.contacts.graph import Plot
class LengthsPlot(Plot):

    help = 'help:user/commands/crosslinks.html#histogram'
    
    def __init__(self, session, pbonds):

        # Create matplotlib panel
        title = '%d Crosslink Lengths' % len(pbonds)
        Plot.__init__(self, session, tool_name = "Crosslinks", title = title)

        self.pbonds = pbonds

        self._bin_edges = None
        self._patches = []
        self._last_picked_bin = None
        self._fattened_pbonds = None
        self._unfattened_radii = None

        self._make_histogram()

        self.figure.set_tight_layout(True)	# Scale plot to fill window.

        self.canvas.mouseMoveEvent = self._mouse_move
        
        self.show()

    def _make_histogram(self, bins=50):
        a = self.axes
        a.set_title('Crosslink lengths')
        a.set_xlabel(r'length ($\AA$)')
        a.set_ylabel('crosslink count')

        d = self.pbonds.lengths
        n, be, self._patches = a.hist(d, bins=bins)
        self._bin_edges = be
        
        if bins > 0:
            # Rightmost bin edge is exactly at max data value.
            # This makes it fall outside bins in numpy.digitize().
            # Remedy this by extending the rightmost bin edge a little.
            be[bins] += 0.01 * (be[bins]-be[bins-1])

        # Map bin to pbonds in bin.
        from numpy import digitize
        pbi = digitize(d, self._bin_edges)-1
        ipb = {}
        for i,pb in zip(pbi, self.pbonds):
            ipb.setdefault(i, []).append(pb)
        from chimerax.core.atomic import Pseudobonds
        self._bin_pbonds = {i:Pseudobonds(pbs) for i,pbs in ipb.items()}
            
    def _mouse_move(self, event):
        e = self.matplotlib_mouse_event(event.x(), event.y())
        for i,p in enumerate(self._patches):
            c,details = p.contains(e)
            if c:
                self._pick_bin(i)
                return
        self._pick_bin(None)

    def _pick_bin(self, b):
        
        if b == self._last_picked_bin:
            return
        
        self._last_picked_bin = b
        fpb = self._fattened_pbonds
        if fpb is not None:
            fpb.radii = self._unfattened_radii

        if b is None:
            self._fattened_pbonds = None
            self._unfattened_radii = None
            self.pbonds.displays = True
        else:
            pb = self.pbonds
            bpb = self._bin_pbonds.get(b, None)
            if bpb is None or len(pb) == 0:
                return
            self._fattened_pbonds = bpb
            self._unfattened_radii = bpb.radii
            bpb.radii *= 3.0
            pb.displays = False
            bpb.displays = True
    
# ------------------------------------------------------------------------------
#
from chimerax.contacts.graph import Plot
class EnsemblePlot(Plot):
    
    def __init__(self, session, pbond, ensemble_model):

        # Create matplotlib panel
        e = ensemble_model
        title = 'Crosslink length for %d models %s' % (e.num_coord_sets, e.name)
        Plot.__init__(self, session, tool_name = "Crosslinks", title = title)

        self.pbond = pbond
        self.ensemble_model = e

        self._bin_edges = None
        self._patches = []
        self._last_picked_bin = None
        self._fattened_pbonds = None
        self._unfattened_radii = None

        self._make_histogram()

        self.figure.set_tight_layout(True)	# Scale plot to fill window.

        self.canvas.mouseMoveEvent = self._mouse_move
        
        self.show()

    def _make_histogram(self, bins=50):
        a = self.axes
        a.set_title('Crosslink lengths')
        a.set_xlabel(r'length ($\AA$)')
        a.set_ylabel('model count')

        a1, a2 = self._crosslink_atoms()
        e = self.ensemble_model
        acid = e.active_coordset_id
        cset_ids = e.coordset_ids
        from numpy import empty, float32
        d = empty((len(cset_ids),), float32)
        from chimerax.core.geometry import distance
        # TODO: Optimize. Changing coordset scans all coordsets.
        #       Make routine to return coords for all coordsets for an atom?
        for i, id in enumerate(cset_ids):
            e.active_coordset_id = id
            d[i] = distance(a1.scene_coord, a2.scene_coord)
        e.active_coordset_id = acid
        
        n, be, self._patches = a.hist(d, bins=bins)
        self._bin_edges = be
        
        if bins > 0:
            # Rightmost bin edge is exactly at max data value.
            # This makes it fall outside bins in numpy.digitize().
            # Remedy this by extending the rightmost bin edge a little.
            be[bins] += 0.01 * (be[bins]-be[bins-1])

        # Map bin to list of coordset ids.
        from numpy import digitize
        bi = digitize(d, self._bin_edges)-1
        self._bin_coordset_ids = bcs = {}
        for i,cs in zip(bi, cset_ids):
            bcs.setdefault(i, []).append(cs)

    def _crosslink_atoms(self):
        pb = self.pbond
        e = self.ensemble_model
        atoms = []
        for a in pb.atoms:
            if a.structure is not e:
                # Find matching atom in e.
                ea = e.atoms
                from numpy import logical_and
                mask = logical_and.reduce(((ea.names == a.name),
                                           (ea.chain_ids == a.chain_id),
                                           (ea.residues.numbers == a.residue.number)))
                matom = ea.filter(mask)
                if len(matom) != 1:
                    from chimerax.core.errors import UserError
                    raise UserError('Require one atom in ensemble %s matching pseudobond atom /%s:%d@%s, got %d'
                                    % (e.name, a.chain_id, a.residue.number, a.name, len(matom)))
                a = matom[0]
            atoms.append(a)
        return atoms

        
    def _mouse_move(self, event):
        e = self.matplotlib_mouse_event(event.x(), event.y())
        for i,p in enumerate(self._patches):
            c,details = p.contains(e)
            if c:
                self._pick_bin(i)
                return
        self._pick_bin(None)

    def _pick_bin(self, b):
        
        if b == self._last_picked_bin:
            return
        self._last_picked_bin = b

        cset_ids = self._bin_coordset_ids.get(b, None)
        if cset_ids:
            e = self.ensemble_model
            e.active_coordset_id = cset_ids[0]
