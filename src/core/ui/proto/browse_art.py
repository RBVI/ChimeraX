import wx
class MyFrame(wx.Frame):
    def __init__(self):
        title = "wx.ART browser"
        wx.Frame.__init__(self, None, -1, title, size = (200,300))

        li = ["wx.%s" % x for x in dir(wx) if x.startswith("ART")]

        lb = wx.ListBox(self, -1, choices = li, style=wx.LB_SINGLE)
        self.sb = wx.StaticBitmap(self, -1, wx.ArtProvider.GetBitmap(eval(li[0])))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(lb)
        sizer.Add( [ 20, 20 ] , 0, wx.ALIGN_CENTER|wx.ALL, 5 )
        sizer.Add(self.sb, wx.ALIGN_CENTER|wx.ALL, 5)
        self.SetSizer(sizer)

        self.Bind(wx.EVT_LISTBOX, self.OnUpdateBitmap, lb)

    def OnUpdateBitmap(self, event):
        name = event.GetString()
        im = wx.ArtProvider.GetBitmap(eval(name))
        w, h = im.GetSize()
        self.sb.SetSize((w,h))
        self.sb.SetBitmap(im)


app = wx.PySimpleApp()
f = MyFrame()
f.Center()
f.Show()
app.MainLoop()
