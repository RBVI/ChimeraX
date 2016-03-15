# -----------------------------------------------------------------------------
# User interface for building cages.
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class CageBuilder(ToolInstance):

    SIZE = (-1, -1)
    SESSION_SKIP = True

    def __init__(self, session, bundle_info):
        ToolInstance.__init__(self, session, bundle_info)

        self.minimize_steps = 10
        self.edge_thickness = 0.1  # Edge diameter as fraction of edge length.

        self.display_name = 'Cage Builder'

        from chimerax.core.ui.gui import MainToolWindow
        tw = MainToolWindow(self, size=self.SIZE)
        self.tool_window = tw
        parent = tw.ui_area

        import wx
        bl = wx.StaticText(parent, label="Create polygon")

        b5 = wx.Button(parent, label='5', style=wx.BU_EXACTFIT)
        b5.Bind(wx.EVT_BUTTON, lambda e: self.attach_polygons(5))
        b6 = wx.Button(parent, label='6', style=wx.BU_EXACTFIT)
        b6.Bind(wx.EVT_BUTTON, lambda e: self.attach_polygons(6))
        bm = wx.Button(parent, label='Minimize', style=wx.BU_EXACTFIT)
        bm.Bind(wx.EVT_BUTTON, self.minimize_cb)
        bel = wx.StaticText(parent, label="Edge length")
        self.edge_length = be = wx.TextCtrl(parent, value = '50', size = (30,-1))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(bl, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(b5, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(b6, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(bm, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(bel, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        sizer.Add(be, 0, wx.FIXED_MINSIZE | wx.ALIGN_CENTER)
        parent.SetSizerAndFit(sizer)

        tw.manage(placement="top")

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def minimize_cb(self, n):
        pass

    def attach_polygons(self, n):
        d = self.vertex_degree()
        length, thickness, inset = self.edge_size()
        from . import cage
        cage.attach_polygons(self.session, 'selected', n, length, thickness, inset,
                             vertex_degree = d)

    def edge_size(self):
        e = float(self.edge_length.GetValue())
        et = self.edge_thickness*e	# Edge thickness
        ei = 0.5*et	# Edge inset
        return e, et, ei

    def vertex_degree(self):
        return 3
    
    def minimize_cb(self, event):
        from . import cage
        for i in range(self.minimize_steps):
            cage.optimize_shape(self.session)

def cage_builder_panel(session, bundle_info):
  cb = getattr(session, '_cage_builder', None)
  if cb is None:
    session._cage_builder = cb = CageBuilder(session, bundle_info)
  return cb
