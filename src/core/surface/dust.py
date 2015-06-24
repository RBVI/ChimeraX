# -----------------------------------------------------------------------------
# Show only part of the surface model within specified distances of the given
# list of points.  The points are in model coordinates.
#
metrics = ('size', 'area', 'volume', 'size rank', 'area rank', 'volume rank')

# -----------------------------------------------------------------------------
#
def hide_dust(model, metric, limit, auto_update, use_cached_geometry = False):

    plist = model.surfacePieces
    for p in plist:
        hide_dust_piece(p, metric, limit, use_cached_geometry)
    if auto_update:
        dust_updater.auto_dust(model, metric, limit)

# -----------------------------------------------------------------------------
#
def hide_dust_piece(p, metric, limit, use_cached_geometry = False):

    # Don't hide surface cap on a visible blob.
    import SurfaceCap
    if SurfaceCap.is_surface_cap(p):
        return

    b = getattr(p, 'blobs', None) if use_cached_geometry else None
    if b is None or p.vertexCount != b.vertex_count:
        p.blobs = b = Blob_Masker(p)
    import Surface
    if Surface.visibility_method(p.model) != 'hide dust':
        Surface.set_visibility_method('hide dust', p.model , None)
    m = b.triangle_mask(metric, limit)
    Surface.set_triangle_mask(p, m)

# -----------------------------------------------------------------------------
#
def largest_blobs_triangle_mask(vertices, triangles, triangle_mask, blob_count = 1,
                                rank_metric = 'size rank'):

    b = Blob_Masker(vertices, triangles, triangle_mask)
    tmask = b.triangle_mask(metric = rank_metric, limit = blob_count)
    return tmask
#    import Surface
#    Surface.set_visibility_method('hide dust', surf , None)
        
# -----------------------------------------------------------------------------
# Stop updating dust hiding.
#
def unhide_dust(model):
    
    dust_updater.stop_hiding_dust(model)
    import Surface
    Surface.reshow_surface(model)

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
            from numpy import zeros, bool
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
            from numpy  import empty, bool
            self.tmask = empty((self.triangle_count,), bool)
        return self.tmask
            
# -----------------------------------------------------------------------------
#
class Dust_Updater:

    def __init__(self):

        self.models = {}

        import SimpleSession
        import chimera
        chimera.triggers.addHandler(SimpleSession.SAVE_SESSION,
                                    self.save_session_cb, None)
            
    # -------------------------------------------------------------------------
    #
    def auto_dust(self, model, metric, limit):

        add_callback = not self.models.has_key(model)
        self.models[model] = (metric, limit)
        if add_callback:
            from Surface import set_visibility_method
            set_visibility_method('hide dust', model, self.stop_hiding_dust)
            model.addGeometryChangedCallback(self.surface_changed_cb)
            import chimera
            chimera.addModelClosedCallback(model, self.model_closed_cb)
            
    # -------------------------------------------------------------------------
    #
    def stop_hiding_dust(self, model):

        if model in self.models:
            del self.models[model]
            model.removeGeometryChangedCallback(self.surface_changed_cb)
            for p in model.surfacePieces:
                if hasattr(p, 'blobs'):
                    delattr(p, 'blobs')
            
    # -------------------------------------------------------------------------
    #
    def surface_changed_cb(self, p, detail):

        if detail == 'removed':
            return

        m = p.model
        (metric, limit) = self.models[m]
        hide_dust_piece(p, metric, limit)
            
    # -------------------------------------------------------------------------
    #
    def model_closed_cb(self, model):

        if model in self.models:
            del self.models[model]
    
    # -------------------------------------------------------------------------
    #
    def save_session_cb(self, trigger, x, file):

        import session
        session.save_hide_dust_state(self.models, file)

# -----------------------------------------------------------------------------
#
def hiding_dust(model):
    return model in dust_updater.models
def dust_limit(model):
    return dust_updater.models[model]

# -----------------------------------------------------------------------------
#
#dust_updater = Dust_Updater()
