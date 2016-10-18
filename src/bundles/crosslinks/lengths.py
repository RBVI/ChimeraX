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
        from numpy import digitize
        self._pbond_bin_index = digitize(d, self._bin_edges)-1
        
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
            bpb = pb.filter(self._pbond_bin_index == b)
            self._fattened_pbonds = bpb
            self._unfattened_radii = bpb.radii
            bpb.radii *= 3.0
            pb.displays = False
            bpb.displays = True
