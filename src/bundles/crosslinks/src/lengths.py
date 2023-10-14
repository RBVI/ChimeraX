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
from chimerax.interfaces.graph import Plot
class LengthsPlot(Plot):

    help = 'help:user/commands/crosslinks.html#histogram'
    
    def __init__(self, session, pbonds,
                 bins = 50, max_length = None, min_length = None, height = None):

        # Create matplotlib panel
        title = '%d Crosslink Lengths' % len(pbonds)
        Plot.__init__(self, session, tool_name = "Crosslinks", title = title)
        self.tool_window.fill_context_menu = self._fill_context_menu

        self.pbonds = pbonds
        self._bins = bins
        self._max_length = max_length
        self._min_length = min_length
        self._height = height
        
        self._bin_edges = None
        self._patches = []
        self._last_picked_bin = None
        self._fattened_pbonds = None
        self._unfattened_radii = None

        self._make_histogram()

        self.figure.set_tight_layout(True)	# Scale plot to fill window.

        self.canvas.mouseMoveEvent = self._mouse_move
        
        self.show()

    def _make_histogram(self):
        a = self.axes
        a.set_title('Crosslink lengths')
        a.set_xlabel(r'length ($\AA$)')
        a.set_ylabel('crosslink count')
        if self._height is not None:
            a.set_ylim([0, self._height])

        d = self.pbonds.lengths
        bins = self._bins
        
        range = [self._min_length, self._max_length]
        if range[0] is None:
            range[0] = d.min()
        if range[1] is None:
            range[1] = d.max()

        n, be, self._patches = a.hist(d, bins=bins, range=range)
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
        from chimerax.atomic import Pseudobonds
        self._bin_pbonds = {i:Pseudobonds(pbs) for i,pbs in ipb.items()}
            
    def _mouse_move(self, event):
        pos = event.pos()
        e = self.matplotlib_mouse_event(pos.x(), pos.y())
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

    def _fill_context_menu(self, menu, x, y):
        self.add_menu_entry(menu, 'Save Plot As...', self.save_plot_as)
    
# ------------------------------------------------------------------------------
#
from chimerax.interfaces.graph import Plot
class EnsemblePlot(Plot):
    
    def __init__(self, session, pbond, ensemble_model,
                 bins = 50, max_length = None, min_length = None, height = None):
        
        # Create matplotlib panel
        e = ensemble_model
        title = 'Crosslink length for %d models %s' % (e.num_coordsets, e.name)
        Plot.__init__(self, session, tool_name = "Crosslinks", title = title)
        self.tool_window.fill_context_menu = self._fill_context_menu

        self.pbond = pbond
        self.ensemble_model = e
        self._bins = bins
        self._max_length = max_length
        self._min_length = min_length
        self._height = height

        self._bin_edges = None
        self._patches = []
        self._last_picked_bin = None
        self._fattened_pbonds = None
        self._unfattened_radii = None

        self._make_histogram()

        self.figure.set_tight_layout(True)	# Scale plot to fill window.

        self.canvas.mouseMoveEvent = self._mouse_move
        
        self.show()

    def _make_histogram(self):
        a = self.axes
        a.set_title('Crosslink lengths')
        a.set_xlabel(r'length ($\AA$)')
        a.set_ylabel('model count')
        if self._height is not None:
            a.set_ylim([0, self._height])

        a1, a2 = self._crosslink_atoms()
        e = self.ensemble_model
        acid = e.active_coordset_id
        cset_ids = e.coordset_ids
        from numpy import empty, float32
        d = empty((len(cset_ids),), float32)
        from chimerax.geometry import distance
        # TODO: Optimize. Changing coordset scans all coordsets.
        #       Make routine to return coords for all coordsets for an atom?
        for i, id in enumerate(cset_ids):
            e.active_coordset_id = id
            d[i] = distance(a1.scene_coord, a2.scene_coord)
        e.active_coordset_id = acid
        
        bins = self._bins

        range = [self._min_length, self._max_length]
        if range[0] is None:
            range[0] = d.min()
        if range[1] is None:
            range[1] = d.max()

        n, be, self._patches = a.hist(d, bins=bins, range=range)
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
                                           (ea.chain_ids == a.residue.chain_id),
                                           (ea.residues.numbers == a.residue.number)))
                matom = ea.filter(mask)
                if len(matom) != 1:
                    from chimerax.core.errors import UserError
                    raise UserError('Require one atom in ensemble %s matching pseudobond atom /%s:%d@%s, got %d'
                                    % (e.name, a.residue.chain_id, a.residue.number, a.name, len(matom)))
                a = matom[0]
            atoms.append(a)
        return atoms

        
    def _mouse_move(self, event):
        pos = event.pos()
        e = self.matplotlib_mouse_event(pos.x(), pos.y())
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

    def _fill_context_menu(self, menu, x, y):
        self.add_menu_entry(menu, 'Save Plot As...', self.save_plot_as)
