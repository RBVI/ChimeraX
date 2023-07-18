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

# -----------------------------------------------------------------------------
# Show only part of the surface model within specified distances of the given
# list of points.  The points are in model coordinates.
#
metrics = ('size', 'area', 'volume', 'size rank', 'area rank', 'volume rank')

# -----------------------------------------------------------------------------
#
def hide_dust(surface, metric, limit, auto_update = False, use_cached_geometry = False):

    # Hide dust for this model and all children.
    for s in surface.all_drawings():
        hide_surface_dust(s, metric, limit, auto_update, use_cached_geometry)

# -----------------------------------------------------------------------------
#
def hide_surface_dust(surface, metric, limit, auto_update = False, use_cached_geometry = False):

    s = surface
    if s.vertices is None:
        return

    if getattr(s, 'is_clip_cap', False):
        return    # Don't hide surface cap on a visible blob.
    
    b = getattr(s, 'blobs', None) if use_cached_geometry else None
    if b is None or len(s.vertices) != b.vertex_count:
        s.blobs = b = Blob_Masker(s.vertices, s.triangles)
    m = b.triangle_mask(metric, limit)
    s.triangle_mask = m

    if auto_update:
        remask = Redust(s, metric, limit)
        from .updaters import add_updater_for_session_saving
        add_updater_for_session_saving(surface.session, remask)
    else:
        remask = None
    s.auto_remask_triangles = remask

# -----------------------------------------------------------------------------
#
def largest_blobs_triangle_mask(vertices, triangles, triangle_mask, blob_count = 1,
                                rank_metric = 'size rank'):
    '''
    Return a triangle mask which includes the N largest connected surface components
    using a specified metric "size rank", "area rank", or "volume rank".  Size rank
    measures maximum extent along x, y and z axes.
    '''
    b = Blob_Masker(vertices, triangles, triangle_mask)
    tmask = b.triangle_mask(metric = rank_metric, limit = blob_count)
    return tmask

# -----------------------------------------------------------------------------
#
def show_only_largest_blobs(surface, visible_only = False, blob_count = 1,
                            rank_metric = 'size rank'):
    s = surface
    # Handle surfaces with duplicate vertices, such as molecular
    # surfaces with sharp edges between atoms.
    t = surface.joined_triangles if hasattr(surface, 'joined_triangles') else surface.triangles
    tmask = s.triangle_mask if visible_only else None
    b = Blob_Masker(s.vertices, t, tmask)
    tmask = b.triangle_mask(metric = rank_metric, limit = blob_count)
    s.triangle_mask = tmask

# -----------------------------------------------------------------------------
#
from chimerax.core.state import State
class Redust(State):
    def __init__(self, surface, metric, limit):
        self.surface = surface
        self.metric = metric
        self.limit = limit

    # -------------------------------------------------------------------------
    #
    def __call__(self):
        self.set_surface_mask()

    # -------------------------------------------------------------------------
    #
    def active(self):
        s = self.surface
        return s is not None and s.auto_remask_triangles is self

    # -------------------------------------------------------------------------
    #
    def set_surface_mask(self):
        surf = self.surface
        hide_dust(surf, self.metric, self.limit, auto_update = False)
        surf.auto_remask_triangles = self

    # -------------------------------------------------------------------------
    #
    def take_snapshot(self, session, flags):
        data = {
            'surface': self.surface,
            'metric': self.metric,
            'limit': self.limit,
            'version': 1,
        }
        return data

    # -------------------------------------------------------------------------
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        surf = data['surface']
        if surf is None:
            return None		# Surface to mask is gone.
        c = cls(surf, data['metric'], data['limit'])
        c.set_surface_mask()
        return c

# -----------------------------------------------------------------------------
#
def dusting(surface):
    rd = surface.auto_remask_triangles
    return rd if isinstance(rd, Redust) else None

# -----------------------------------------------------------------------------
# Stop updating dust hiding.
#
def unhide_dust(model):
    for s in model.all_drawings():
        s.triangle_mask = None
        s.auto_remask_triangles = None

# -----------------------------------------------------------------------------
#
class Blob_Masker:

    def __init__(self, vertices, triangles, triangle_mask = None):

        self.tsubset = triangle_mask
                
        self.varray = va = vertices
        self.tarray = ta = triangles if triangle_mask is None else triangles[triangle_mask,:]
        self.vertex_count = len(va)
        self.triangle_count = len(ta)

        self.blist = None
        self.tbindex = None
        self.tbvalues = {}
        self.tbsizes = None
        self.tbranks = None
        self.tbvolumes = None
        self.tbareas = None
        self.tmask = None

    def triangle_mask(self, metric, limit):

        mask = self.mask_array()
        from numpy import greater
        tvalues = self.triangle_values(metric)
        if metric.endswith('rank'):
            r = len(self.blob_list()) - int(limit+1)
            greater(tvalues, r, mask)
        else:
            greater(tvalues, limit, mask)

        ts = self.tsubset
        if ts is None:
            tmask = mask
        else:
            from numpy import zeros
            tmask = zeros((len(ts),), bool)
            tmask[ts.nonzero()[0]] = mask
        return tmask

    def triangle_values(self, metric):

        if metric == 'size':
            v = self.blob_sizes()
        elif metric == 'area':
            v = self.blob_areas()
        elif metric == 'volume':
            v = self.blob_volumes()
        elif metric.endswith('rank'):
            v = self.blob_ranks(metric)
        return v

    def triangle_blob_indices(self):
        if self.tbindex is None:
            from numpy import empty, intc
            tbi = empty((self.triangle_count,), intc)
            for i, (vi, ti) in enumerate(self.blob_list()):
                tbi.put(ti, i)
            self.tbindex = tbi
        return self.tbindex

    def blob_sizes(self):
        tv = self.tbvalues
        if not 'size' in tv:
            bsizes = self.blob_values(self.blob_size)
            tv['size'] = bsizes[self.triangle_blob_indices()]
        return tv['size']

    def blob_size(self, vi, ti):
        v = self.varray[vi,:]
        return max(v.max(axis=0) - v.min(axis=0))

    def blob_ranks(self, metric):
        tv = self.tbvalues
        if not metric in tv:
            m = {'size rank':self.blob_size,
                 'area rank':self.blob_area,
                 'volume rank':self.blob_volume}[metric]
            border = self.blob_values(m).argsort()
            branks = border.copy()
            from numpy import arange
            branks.put(border, arange(len(border)))
            tv[metric] = branks[self.triangle_blob_indices()]
        return tv[metric]

    def blob_volumes(self):
        tv = self.tbvalues
        if not 'volume' in tv:
            bvolumes = self.blob_values(self.blob_volume)
            tv['volume'] = bvolumes[self.triangle_blob_indices()]
        return tv['volume']

    def blob_volume(self, vi, ti):
        from .area import enclosed_volume
        t = self.tarray[ti,:]
        vol, holes = enclosed_volume(self.varray, t)
        if vol is None:
            vol = 0
        return vol

    def blob_areas(self):
        tv = self.tbvalues
        if not 'area' in tv:
            bareas = self.blob_values(self.blob_area)
            tv['area'] = bareas[self.triangle_blob_indices()]
        return tv['area']

    def blob_area(self, vi, ti):
        from .area import surface_area
        t = self.tarray[ti,:]
        area = surface_area(self.varray, t)
        return area

    def blob_values(self, value_func):
        blist = self.blob_list()
        from numpy import empty, single as floatc
        bv = empty((len(blist),), floatc)
        for i, (vi,ti) in enumerate(blist):
            bv[i] = value_func(vi, ti)
        return bv

    def blob_list(self):
        if self.blist is None:
            from ._surface import connected_pieces
            self.blist = connected_pieces(self.tarray)
        return self.blist

    def mask_array(self):
        if self.tmask is None:
            from numpy import empty
            self.tmask = empty((self.triangle_count,), bool)
        return self.tmask
