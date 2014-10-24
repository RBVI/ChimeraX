# vim: set expandtab ts=4 sw=4:

import wx

class SceneThumbnails:
    def __init__(self, session):
        self.session = session

        from ..tool_api import ToolWindow
        self._scenes_window = ToolWindow("Scenes", "General", session,
            destroy_hides=True)
        #from wx.html2 import WebView, EVT_WEBVIEW_NAVIGATING, EVT_WEBVIEW_ERROR
        #self._scenes = WebView.New(self._scenes_window.ui_area, size=(700, 200))
        from wx.html import HtmlWindow, EVT_HTML_LINK_CLICKED
        self._scenes = HtmlWindow(self._scenes_window.ui_area, size=(700, 200))
        self._scenes.SetHTMLBackgroundColour(wx.Colour(0,0,0))
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self._scenes, 1, wx.EXPAND)
        self._scenes_window.ui_area.SetSizerAndFit(sizer)
        self._scenes_window.manage("top")
        self._scenes_window.shown = False
        #self._scenes.Bind(EVT_WEBVIEW_NAVIGATING,
        #    self.OnWebViewNavigating, self._scenes)
        #self._scenes.Bind(EVT_WEBVIEW_ERROR,
        #    self.OnWebViewError, self._scenes)
        wx.FileSystem.AddHandler(wx.MemoryFSHandler())
        #self._scenes.RegisterHandler(wx.MemoryFSHandler())
        self._scenes.Bind(EVT_HTML_LINK_CLICKED, self.OnHtmlLinkClicked)
        self.memory_files = []

    """
    def OnWebViewNavigating(self, event):
        url = event.GetURL()
        scene_id = wx.FileSystem.URLToFileName(url)
        while not scene_id[0].isdigit():
            scene_id = scene_id[1:]
            if not scene_id:
                return
        self.session.scenes.show_scene(int(scene_id))
        event.Veto()

    def OnWebViewError(self, event):
        import sys
        print("types of Int/String: {} {}".format(type(event.Int), type(event.String)), file=sys.stderr)
    """

    def OnHtmlLinkClicked(self, event):
        scene_id = event.GetLinkInfo().GetHref()
        self.session.scenes.show_scene(int(scene_id))

    def show(self, scenes):
        while self.memory_files:
            wx.MemoryFSHandler.RemoveFile(self.memory_files.pop())
        lines = ['<html>', '<head>', '<style>',
           'body { background-color: black; }',
           'a { text-decoration: none; }',      # No underlining of links
           'a:link { color: #FFFFFF; }',        # Link text color white.
           'table { float:left; }',     # Multiple image/caption tables per row.
           'td { font-size:large; }',
           #'td { text-align:center; }',        # Does not work in Qt 5.0.2
           '</style>', '</head>', '<body bgcolor="black">',
           '<table style="float:left;">', '<tr>',
        ]
        import io
        for s in scenes:
            img = s.image
            mem_name = "image_{}.png".format(s.id)
            self.memory_files.append(mem_name)
            img_io = io.BytesIO()
            img.save(img_io, format='PNG')
            png_data = img_io.getvalue()
            bitmap = wx.Bitmap.NewFromPNGData(png_data, len(png_data))
            image = bitmap.ConvertToImage()
            fs_name = "/Users/pett/rm/" + mem_name
            image.SaveFile(fs_name, wx.BITMAP_TYPE_PNG)
            wx.MemoryFSHandler.AddFile(mem_name, image, wx.BITMAP_TYPE_PNG)
            w, h = img.size
            lines.append('<td width={} valign=bottom><a href="{}">'
                '<img src="memory:{}", width={} height={}</a>'.format(
                w + 10, s.id, mem_name, w, h))
            #'<img src="file://{}", width={} height={}</a>'.format(
            #w + 10, s.id, fs_name, w, h))
        lines.append('<tr>')
        for s in scenes:
            lines.append('<td><a href="{}"><center>{}</center></a>'.format(
                s.id, s.id))
        if [s for s in scenes if s.description]:
            lines.append('<tr>')
            import cgi
            for s in scenes:
                line = ('<td><a href="{}">{}</a>'.format(s.id,
                    cgi.escape(s.description))) if s.description else '<td>'
                lines.append(line)
        lines.extend(['</table>', '</body>', '</html>'])
        #self._scenes.SetPage('\n'.join(lines), "")
        self._scenes.SetPage('\n'.join(lines))
        self._scenes_window.shown = True
        #fs = wx.FileSystem()
        #fs.ChangePathTo("memory:")
        #name = fs.FindFirst("*")
        #while name:
        #    name = fs.FindNext()

    def shown(self):
        return self._scenes_window.shown

    def hide(self):
        self._scenes_window.shown = False

    def set_height(self, **kw):
        import sys
        print("set_height to {}".format(kw), file=sys.stderr)
