#!/usr/bin/env python
# vim: set expandtab ts=4 sw=4:

xrange = range

import wx
import wx.grid
import wx.html
#import wx.aui
import wx.lib.agw.aui
wx.aui = wx.lib.agw.aui

from wx.lib.six import BytesIO

ID_CreateTree = wx.NewId()
ID_CreateGrid = wx.NewId()
ID_CreateText = wx.NewId()
ID_CreateHTML = wx.NewId()
ID_CreateSizeReport = wx.NewId()
ID_GridContent = wx.NewId()
ID_TextContent = wx.NewId()
ID_TreeContent = wx.NewId()
ID_HTMLContent = wx.NewId()
ID_SizeReportContent = wx.NewId()
ID_CreatePerspective = wx.NewId()
ID_CopyPerspective = wx.NewId()

ID_TransparentHint = wx.NewId()
ID_VenetianBlindsHint = wx.NewId()
ID_RectangleHint = wx.NewId()
ID_NoHint = wx.NewId()
ID_HintFade = wx.NewId()
ID_AllowFloating = wx.NewId()
ID_NoVenetianFade = wx.NewId()
ID_TransparentDrag = wx.NewId()
ID_AllowActivePane = wx.NewId()
ID_NoGradient = wx.NewId()
ID_VerticalGradient = wx.NewId()
ID_HorizontalGradient = wx.NewId()

ID_Settings = wx.NewId()
ID_About = wx.NewId()
ID_FirstPerspective = ID_CreatePerspective+1000



#----------------------------------------------------------------------
def GetMondrianData():
    return \
b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 \x08\x06\x00\
\x00\x00szz\xf4\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\x00\x00qID\
ATX\x85\xed\xd6;\n\x800\x10E\xd1{\xc5\x8d\xb9r\x97\x16\x0b\xad$\x8a\x82:\x16\
o\xda\x84pB2\x1f\x81Fa\x8c\x9c\x08\x04Z{\xcf\xa72\xbcv\xfa\xc5\x08 \x80r\x80\
\xfc\xa2\x0e\x1c\xe4\xba\xfaX\x1d\xd0\xde]S\x07\x02\xd8>\xe1wa-`\x9fQ\xe9\
\x86\x01\x04\x10\x00\\(Dk\x1b-\x04\xdc\x1d\x07\x14\x98;\x0bS\x7f\x7f\xf9\x13\
\x04\x10@\xf9X\xbe\x00\xc9 \x14K\xc1<={\x00\x00\x00\x00IEND\xaeB`\x82'


def GetMondrianBitmap():
    return wx.Bitmap(GetMondrianImage())


def GetMondrianImage():
    stream = BytesIO(GetMondrianData())
    return wx.Image(stream)


def GetMondrianIcon():
    icon = wx.Icon()
    icon.CopyFromBitmap(GetMondrianBitmap())
    return icon


class PyAUIFrame(wx.Frame):

    def __init__(self, parent, id=-1, title="", pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE |
                                            wx.SUNKEN_BORDER |
                                            wx.CLIP_CHILDREN):

        wx.Frame.__init__(self, parent, id, title, pos, size, style)

        # tell FrameManager to manage this frame
        self._mgr = wx.aui.AuiManager()
        self._mgr.SetManagedWindow(self)

        self._perspectives = []
        self.n = 0
        self.x = 0

        self.SetIcon(GetMondrianIcon())

        # create menu
        mb = wx.MenuBar()

        file_menu = wx.Menu()
        file_menu.Append(wx.ID_EXIT, "Exit")

        view_menu = wx.Menu()
        view_menu.Append(ID_CreateText, "Create Text Control")
        view_menu.Append(ID_CreateHTML, "Create HTML Control")
        view_menu.Append(ID_CreateTree, "Create Tree")
        view_menu.Append(ID_CreateGrid, "Create Grid")
        view_menu.Append(ID_CreateSizeReport, "Create Size Reporter")
        view_menu.AppendSeparator()
        view_menu.Append(ID_GridContent, "Use a Grid for the Content Pane")
        view_menu.Append(ID_TextContent, "Use a Text Control for the Content Pane")
        view_menu.Append(ID_HTMLContent, "Use an HTML Control for the Content Pane")
        view_menu.Append(ID_TreeContent, "Use a Tree Control for the Content Pane")
        view_menu.Append(ID_SizeReportContent, "Use a Size Reporter for the Content Pane")

        options_menu = wx.Menu()
        options_menu.AppendRadioItem(ID_TransparentHint, "Transparent Hint")
        options_menu.AppendRadioItem(ID_VenetianBlindsHint, "Venetian Blinds Hint")
        options_menu.AppendRadioItem(ID_RectangleHint, "Rectangle Hint")
        options_menu.AppendRadioItem(ID_NoHint, "No Hint")
        options_menu.AppendSeparator();
        options_menu.AppendCheckItem(ID_HintFade, "Hint Fade-in")
        options_menu.AppendCheckItem(ID_AllowFloating, "Allow Floating")
        options_menu.AppendCheckItem(ID_NoVenetianFade, "Disable Venetian Blinds Hint Fade-in")
        options_menu.AppendCheckItem(ID_TransparentDrag, "Transparent Drag")
        options_menu.AppendCheckItem(ID_AllowActivePane, "Allow Active Pane")
        options_menu.AppendSeparator();
        options_menu.AppendRadioItem(ID_NoGradient, "No Caption Gradient")
        options_menu.AppendRadioItem(ID_VerticalGradient, "Vertical Caption Gradient")
        options_menu.AppendRadioItem(ID_HorizontalGradient, "Horizontal Caption Gradient")
        options_menu.AppendSeparator();
        options_menu.Append(ID_Settings, "Settings Pane")

        self._perspectives_menu = wx.Menu()
        self._perspectives_menu.Append(ID_CreatePerspective, "Create Perspective")
        self._perspectives_menu.Append(ID_CopyPerspective, "Copy Perspective Data To Clipboard")
        self._perspectives_menu.AppendSeparator()
        self._perspectives_menu.Append(ID_FirstPerspective+0, "Default Startup")
        self._perspectives_menu.Append(ID_FirstPerspective+1, "All Panes")
        self._perspectives_menu.Append(ID_FirstPerspective+2, "Vertical Toolbar")

        help_menu = wx.Menu()
        help_menu.Append(ID_About, "About...")

        mb.Append(file_menu, "File")
        mb.Append(view_menu, "View")
        mb.Append(self._perspectives_menu, "Perspectives")
        mb.Append(options_menu, "Options")
        mb.Append(help_menu, "Help")

        self.SetMenuBar(mb)

        self.statusbar = self.CreateStatusBar(2, wx.STB_SIZEGRIP)
        self.statusbar.SetStatusWidths([-2, -3])
        self.statusbar.SetStatusText("Status", 0)
        self.statusbar.SetStatusText("Welcome To Chimera 2", 1)

        # min size for the frame itself isn't completely done.
        # see the end up FrameManager::Update() for the test
        # code. For now, just hard code a frame minimum size
        self.SetMinSize(wx.Size(400, 300))

        # create some toolbars
        tb1 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                         wx.TB_FLAT | wx.TB_NODIVIDER)
        tb1.SetToolBitmapSize(wx.Size(48,48))
        tb1.AddTool(101, "Test", wx.ArtProvider.GetBitmap(wx.ART_ERROR))
        tb1.AddSeparator()
        tb1.AddTool(102, "Test", wx.ArtProvider.GetBitmap(wx.ART_QUESTION))
        tb1.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_INFORMATION))
        tb1.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_WARNING))
        tb1.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_MISSING_IMAGE))
        tb1.Realize()

        tb2 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                         wx.TB_FLAT | wx.TB_NODIVIDER)
        tb2.SetToolBitmapSize(wx.Size(32,32))
        tb2_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_QUESTION, wx.ART_OTHER, wx.Size(16, 16))
        for fn in ["back.png", "book.png", "gauge.png", "grid.png",
                "select.png", "icecube.png", "rotmol.png"]:
            image = wx.Image("../hydra/ui/icons/" + fn)
            w, h = image.GetWidth(), image.GetHeight()
            scaling = 32.0 / max(w, h)
            image.Rescale(int(w * scaling + 0.5), int(h * scaling + 0.5),
                wx.IMAGE_QUALITY_HIGH)
            tb2.AddTool( 101, "", wx.Bitmap(image))

        #tb2.AddTool(101, "", wx.Bitmap(wx.Image("../hydra/ui/icons/select.png")))
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddSeparator()
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddSeparator()
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddTool(101, "Test", tb2_bmp1)
        #tb2.AddTool(101, "Test", tb2_bmp1)
        tb2.Realize()

        tb3 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                         wx.TB_FLAT | wx.TB_NODIVIDER)
        tb3.SetToolBitmapSize(wx.Size(48,48))
        tb3_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, wx.Size(16, 16))
        tb3.AddTool(101, "", wx.Bitmap(wx.Image("icons/scenes/image1.png")))
        tb3.AddTool(101, "", wx.Bitmap(wx.Image("icons/scenes/image2.png")))
        tb3.AddTool(101, "", wx.Bitmap(wx.Image("icons/scenes/image3.png")))
        tb3.AddTool(101, "", wx.Bitmap(wx.Image("icons/scenes/image4.png")))
        #tb3.AddTool(101, "Test", tb3_bmp1)
        #tb3.AddTool(101, "Test", tb3_bmp1)
        #tb3.AddTool(101, "Test", tb3_bmp1)
        #tb3.AddTool(101, "Test", tb3_bmp1)
        #tb3.AddSeparator()
        #tb3.AddTool(101, "Test", tb3_bmp1)
        #tb3.AddTool(101, "Test", tb3_bmp1)
        tb3.Realize()

        tb4 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                     wx.TB_FLAT | wx.TB_NODIVIDER | wx.TB_HORZ_TEXT)
        #tb4 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
        #             wx.TB_FLAT | wx.TB_NODIVIDER | wx.TB_HORZ_TEXT)
        tb4.SetToolBitmapSize(wx.Size(48,48))
        tb4_bmp1 = wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, wx.Size(16, 16))
        tb4.AddTool(101, "hbond", wx.Bitmap(wx.Image("icons/tools/hbond.png")))
        tb4.AddTool(101, "match", wx.Bitmap(wx.Image("icons/tools/matchmaker.png")))
        tb4.AddTool(101, "dock prep", wx.Bitmap(wx.Image("icons/tools/dockprep.png")))
        #tb4.AddTool(101, "Item 1", tb4_bmp1)
        #tb4.AddTool(101, "Item 2", tb4_bmp1)
        #tb4.AddTool(101, "Item 3", tb4_bmp1)
        #tb4.AddTool(101, "Item 4", tb4_bmp1)
        #tb4.AddSeparator()
        tb4.AddTool(101, "volume", wx.Bitmap(wx.Image("icons/tools/volumeviewer.gif")))
        #tb4.AddTool(101, "Item 5", tb4_bmp1)
        #tb4.AddTool(101, "Item 6", tb4_bmp1)
        #tb4.AddTool(101, "Item 7", tb4_bmp1)
        #tb4.AddTool(101, "Item 8", tb4_bmp1)
        tb4.Realize()

        tb5 = wx.ToolBar(self, -1, wx.DefaultPosition, wx.DefaultSize,
                         wx.TB_FLAT | wx.TB_NODIVIDER | wx.TB_VERTICAL)
        tb5.SetToolBitmapSize(wx.Size(48, 48))
        tb5.AddTool(101, "Test", wx.ArtProvider.GetBitmap(wx.ART_ERROR))
        tb5.AddSeparator()
        tb5.AddTool(102, "Test", wx.ArtProvider.GetBitmap(wx.ART_QUESTION))
        tb5.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_INFORMATION))
        tb5.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_WARNING))
        tb5.AddTool(103, "Test", wx.ArtProvider.GetBitmap(wx.ART_MISSING_IMAGE))
        tb5.Realize()

        # add a bunch of panes
        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test1").Caption("Pane Caption").Top().
                          CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test2").Caption("Client Size Reporter").
                          Bottom().Position(1).CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test3").Caption("Client Size Reporter").
                          Bottom().CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test4").Caption("Pane Caption").
                          Left().CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test5").Caption("Pane Caption").
                          Right().CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test6").Caption("Client Size Reporter").
                          Right().Row(1).CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test7").Caption("Client Size Reporter").
                          Left().Layer(1).CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateTreeCtrl(), wx.aui.AuiPaneInfo().
                          Name("test8").Caption("Model Panel").
                          Left().Layer(1).Position(1).CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateCommandLine(), wx.aui.AuiPaneInfo().
                          Name("test9").Caption("Command Line").
                          Bottom().Layer(0).Position(0).CloseButton(True).MaximizeButton(True))

        #self._mgr.AddPane(self.CreateTextCtrl(), wx.aui.AuiPaneInfo().
        self._mgr.AddPane(self.CreateLog(), wx.aui.AuiPaneInfo().
                          Name("test10").Caption("Log").
                          Bottom().Layer(1).Position(1).CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Name("test11").Caption("Fixed Pane").
                          Bottom().Layer(1).Position(2).Fixed().CloseButton(True).MaximizeButton(True))

        self._mgr.AddPane(SettingsPanel(self, self), wx.aui.AuiPaneInfo().
                          Name("settings").Caption("Dock Manager Settings").
                          Dockable(False).Float().Hide().CloseButton(True).MaximizeButton(True))

        # create some center panes

        self._mgr.AddPane(self.CreateGrid(), wx.aui.AuiPaneInfo().Name("grid_content").
                          CenterPane().Hide())

        self._mgr.AddPane(self.CreateTreeCtrl(), wx.aui.AuiPaneInfo().Name("tree_content").
                          CenterPane().Hide())

        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().Name("sizereport_content").
                          CenterPane().Hide())

        self._mgr.AddPane(self.CreateTextCtrl(), wx.aui.AuiPaneInfo().Name("text_content").
                          CenterPane().Hide())

        #self._mgr.AddPane(self.CreateHTMLCtrl(), wx.aui.AuiPaneInfo().Name("html_content").
        self._mgr.AddPane(self.CreateOpenGL(), wx.aui.AuiPaneInfo().Name("html_content").
                          CenterPane())

        # add the toolbars to the manager

        self._mgr.AddPane(tb1, wx.aui.AuiPaneInfo().
                          Name("tb1").Caption("Big Toolbar").
                          ToolbarPane().Top().
                          LeftDockable(False).RightDockable(False))

        self._mgr.AddPane(tb2, wx.aui.AuiPaneInfo().
                          Name("tb2").Caption("Actions").
                          ToolbarPane().Top().Row(1).
                          LeftDockable(False).RightDockable(False))

        self._mgr.AddPane(tb3, wx.aui.AuiPaneInfo().
                          Name("tb3").Caption("Scenes").
                          ToolbarPane().Top().Row(1).Position(1).
                          LeftDockable(False).RightDockable(False))

        self._mgr.AddPane(tb4, wx.aui.AuiPaneInfo().
                          Name("tb4").Caption("Tools").
                          ToolbarPane().Top().Row(1).Position(2).
                          LeftDockable(False).RightDockable(False))

        self._mgr.AddPane(tb5, wx.aui.AuiPaneInfo().
                          Name("tbvert").Caption("Sample Vertical Toolbar").
                          ToolbarPane().Left().GripperTop().
                          TopDockable(False).BottomDockable(False))

        self._mgr.AddPane(wx.Button(self, -1, "Test Button"),
                          wx.aui.AuiPaneInfo().Name("tb5").
                          ToolbarPane().Top().Row(2).Position(1).
                          LeftDockable(False).RightDockable(False))

        # make some default perspectives

        self._mgr.GetPane("tbvert").Hide()

        perspective_all = self._mgr.SavePerspective()

        all_panes = self._mgr.GetAllPanes()

        for ii in xrange(len(all_panes)):
            if not all_panes[ii].IsToolbar():
                all_panes[ii].Hide()

        self._mgr.GetPane("tb1").Hide()
        self._mgr.GetPane("tb5").Hide()
        self._mgr.GetPane("test8").Show().Left().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("test10").Show().Bottom().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("test9").Show().Bottom().Layer(1).Row(1).Position(1)
        self._mgr.GetPane("html_content").Show()

        perspective_default = self._mgr.SavePerspective()

        for ii in xrange(len(all_panes)):
            if not all_panes[ii].IsToolbar():
                all_panes[ii].Hide()

        self._mgr.GetPane("tb1").Hide()
        self._mgr.GetPane("tb5").Hide()
        self._mgr.GetPane("tbvert").Show()
        self._mgr.GetPane("grid_content").Show()
        self._mgr.GetPane("test10").Show().Left().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("test8").Show().Bottom().Layer(0).Row(0).Position(0)
        self._mgr.GetPane("test9").Show().Bottom().Layer(1).Row(1).Position(1)
        self._mgr.GetPane("html_content").Show()

        perspective_vert = self._mgr.SavePerspective()

        self._perspectives.append(perspective_default)
        self._perspectives.append(perspective_all)
        self._perspectives.append(perspective_vert)

        self._mgr.GetPane("tbvert").Hide()
        self._mgr.GetPane("grid_content").Hide()

        # "commit" all changes made to FrameManager
        self._mgr.Update()

        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        # Show How To Use The Closing Panes Event
        self.Bind(wx.aui.EVT_AUI_PANE_CLOSE, self.OnPaneClose)

        self.Bind(wx.EVT_MENU, self.OnCreateTree, id=ID_CreateTree)
        self.Bind(wx.EVT_MENU, self.OnCreateGrid, id=ID_CreateGrid)
        self.Bind(wx.EVT_MENU, self.OnCreateText, id=ID_CreateText)
        self.Bind(wx.EVT_MENU, self.OnCreateHTML, id=ID_CreateHTML)
        self.Bind(wx.EVT_MENU, self.OnCreateSizeReport, id=ID_CreateSizeReport)
        self.Bind(wx.EVT_MENU, self.OnCreatePerspective, id=ID_CreatePerspective)
        self.Bind(wx.EVT_MENU, self.OnCopyPerspective, id=ID_CopyPerspective)

        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_AllowFloating)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_TransparentHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_VenetianBlindsHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_RectangleHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_NoHint)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_HintFade)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_NoVenetianFade)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_TransparentDrag)
        self.Bind(wx.EVT_MENU, self.OnManagerFlag, id=ID_AllowActivePane)

        self.Bind(wx.EVT_MENU, self.OnGradient, id=ID_NoGradient)
        self.Bind(wx.EVT_MENU, self.OnGradient, id=ID_VerticalGradient)
        self.Bind(wx.EVT_MENU, self.OnGradient, id=ID_HorizontalGradient)
        self.Bind(wx.EVT_MENU, self.OnSettings, id=ID_Settings)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_GridContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_TreeContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_TextContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_SizeReportContent)
        self.Bind(wx.EVT_MENU, self.OnChangeContentPane, id=ID_HTMLContent)
        self.Bind(wx.EVT_MENU, self.OnExit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.OnAbout, id=ID_About)

        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_TransparentHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_VenetianBlindsHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_RectangleHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_NoHint)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_HintFade)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_AllowFloating)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_NoVenetianFade)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_TransparentDrag)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_AllowActivePane)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_NoGradient)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_VerticalGradient)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateUI, id=ID_HorizontalGradient)


        self.Bind(wx.EVT_MENU_RANGE, self.OnRestorePerspective, id=ID_FirstPerspective,
                  id2=ID_FirstPerspective+1000)


    def OnPaneClose(self, event):

        caption = event.GetPane().caption

        if caption in ["Model Panel", "Dock Manager Settings", "Fixed Pane"]:
            msg = "Are You Sure You Want To Close This Pane?"
            dlg = wx.MessageDialog(self, msg, "AUI Question",
                                   wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION)

            if dlg.ShowModal() in [wx.ID_NO, wx.ID_CANCEL]:
                event.Veto()
            dlg.Destroy()


    def OnClose(self, event):
        self._mgr.UnInit()
        del self._mgr
        self.Destroy()


    def OnExit(self, event):
        self.Close()

    def OnAbout(self, event):

        msg = "wx.aui Demo\n" + \
              "An advanced window management library for wxWidgets\n" + \
              "(c) Copyright 2005-2006, Kirix Corporation"
        dlg = wx.MessageDialog(self, msg, "About wx.aui Demo",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()


    def GetDockArt(self):

        return self._mgr.GetArtProvider()


    def DoUpdate(self):

        self._mgr.Update()


    def OnEraseBackground(self, event):

        event.Skip()


    def OnSize(self, event):

        event.Skip()


    def OnSettings(self, event):

        # show the settings pane, and float it
        floating_pane = self._mgr.GetPane("settings").Float().Show()

        if floating_pane.floating_pos == wx.DefaultPosition:
            floating_pane.FloatingPosition(self.GetStartPosition())

        self._mgr.Update()


    def OnGradient(self, event):

        gradient = 0

        if event.GetId() == ID_NoGradient:
            gradient = wx.aui.AUI_GRADIENT_NONE
        elif event.GetId() == ID_VerticalGradient:
            gradient = wx.aui.AUI_GRADIENT_VERTICAL
        elif event.GetId() == ID_HorizontalGradient:
            gradient = wx.aui.AUI_GRADIENT_HORIZONTAL

        self._mgr.GetArtProvider().SetMetric(wx.aui.AUI_DOCKART_GRADIENT_TYPE, gradient)
        self._mgr.Update()


    def OnManagerFlag(self, event):

        flag = 0
        eid = event.GetId()

        if eid in [ ID_TransparentHint, ID_VenetianBlindsHint, ID_RectangleHint, ID_NoHint ]:
            flags = self._mgr.GetFlags()
            flags &= ~wx.aui.AUI_MGR_TRANSPARENT_HINT
            flags &= ~wx.aui.AUI_MGR_VENETIAN_BLINDS_HINT
            flags &= ~wx.aui.AUI_MGR_RECTANGLE_HINT
            self._mgr.SetFlags(flags)

        if eid == ID_AllowFloating:
            flag = wx.aui.AUI_MGR_ALLOW_FLOATING
        elif eid == ID_TransparentDrag:
            flag = wx.aui.AUI_MGR_TRANSPARENT_DRAG
        elif eid == ID_HintFade:
            flag = wx.aui.AUI_MGR_HINT_FADE
        elif eid == ID_NoVenetianFade:
            flag = wx.aui.AUI_MGR_NO_VENETIAN_BLINDS_FADE
        elif eid == ID_AllowActivePane:
            flag = wx.aui.AUI_MGR_ALLOW_ACTIVE_PANE
        elif eid == ID_TransparentHint:
            flag = wx.aui.AUI_MGR_TRANSPARENT_HINT
        elif eid == ID_VenetianBlindsHint:
            flag = wx.aui.AUI_MGR_VENETIAN_BLINDS_HINT
        elif eid == ID_RectangleHint:
            flag = wx.aui.AUI_MGR_RECTANGLE_HINT

        self._mgr.SetFlags(self._mgr.GetFlags() ^ flag)


    def OnUpdateUI(self, event):

        flags = self._mgr.GetFlags()
        eid = event.GetId()

        if eid == ID_NoGradient:
            event.Check(self._mgr.GetArtProvider().GetMetric(wx.aui.AUI_DOCKART_GRADIENT_TYPE) == wx.aui.AUI_GRADIENT_NONE)

        elif eid == ID_VerticalGradient:
            event.Check(self._mgr.GetArtProvider().GetMetric(wx.aui.AUI_DOCKART_GRADIENT_TYPE) == wx.aui.AUI_GRADIENT_VERTICAL)

        elif eid == ID_HorizontalGradient:
            event.Check(self._mgr.GetArtProvider().GetMetric(wx.aui.AUI_DOCKART_GRADIENT_TYPE) == wx.aui.AUI_GRADIENT_HORIZONTAL)

        elif eid == ID_AllowFloating:
            event.Check((flags & wx.aui.AUI_MGR_ALLOW_FLOATING) != 0)

        elif eid == ID_TransparentDrag:
            event.Check((flags & wx.aui.AUI_MGR_TRANSPARENT_DRAG) != 0)

        elif eid == ID_TransparentHint:
            event.Check((flags & wx.aui.AUI_MGR_TRANSPARENT_HINT) != 0)

        elif eid == ID_VenetianBlindsHint:
            event.Check((flags & wx.aui.AUI_MGR_VENETIAN_BLINDS_HINT) != 0)

        elif eid == ID_RectangleHint:
            event.Check((flags & wx.aui.AUI_MGR_RECTANGLE_HINT) != 0)

        elif eid == ID_NoHint:
            event.Check(((wx.aui.AUI_MGR_TRANSPARENT_HINT |
                          wx.aui.AUI_MGR_VENETIAN_BLINDS_HINT |
                          wx.aui.AUI_MGR_RECTANGLE_HINT) & flags) == 0)

        elif eid == ID_HintFade:
            event.Check((flags & wx.aui.AUI_MGR_HINT_FADE) != 0);

        elif eid == ID_NoVenetianFade:
            event.Check((flags & wx.aui.AUI_MGR_NO_VENETIAN_BLINDS_FADE) != 0);




    def OnCreatePerspective(self, event):

        dlg = wx.TextEntryDialog(self, "Enter a name for the new perspective:", "AUI Test")

        dlg.SetValue(("Perspective %d")%(len(self._perspectives)+1))
        if dlg.ShowModal() != wx.ID_OK:
            return

        if len(self._perspectives) == 0:
            self._perspectives_menu.AppendSeparator()

        self._perspectives_menu.Append(ID_FirstPerspective + len(self._perspectives), dlg.GetValue())
        self._perspectives.append(self._mgr.SavePerspective())


    def OnCopyPerspective(self, event):

        s = self._mgr.SavePerspective()

        if wx.TheClipboard.Open():

            wx.TheClipboard.SetData(wx.TextDataObject(s))
            wx.TheClipboard.Close()

    def OnRestorePerspective(self, event):

        self._mgr.LoadPerspective(self._perspectives[event.GetId() - ID_FirstPerspective])


    def GetStartPosition(self):

        self.x = self.x + 20
        x = self.x
        pt = self.ClientToScreen(wx.Point(0, 0))

        return wx.Point(pt.x + x, pt.y + x)


    def OnCreateTree(self, event):
        self._mgr.AddPane(self.CreateTreeCtrl(), wx.aui.AuiPaneInfo().
                          Caption("Tree Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          FloatingSize(wx.Size(150, 300)).CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnCreateGrid(self, event):
        self._mgr.AddPane(self.CreateGrid(), wx.aui.AuiPaneInfo().
                          Caption("Grid").
                          Float().FloatingPosition(self.GetStartPosition()).
                          FloatingSize(wx.Size(300, 200)).CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnCreateHTML(self, event):
        self._mgr.AddPane(self.CreateHTMLCtrl(), wx.aui.AuiPaneInfo().
                          Caption("HTML Content").
                          Float().FloatingPosition(self.GetStartPosition()).
                          FloatingSize(wx.Size(300, 200)).CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnCreateText(self, event):
        self._mgr.AddPane(self.CreateTextCtrl(), wx.aui.AuiPaneInfo().
                          Caption("Text Control").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnCreateSizeReport(self, event):
        self._mgr.AddPane(self.CreateSizeReportCtrl(), wx.aui.AuiPaneInfo().
                          Caption("Client Size Reporter").
                          Float().FloatingPosition(self.GetStartPosition()).
                          CloseButton(True).MaximizeButton(True))
        self._mgr.Update()


    def OnChangeContentPane(self, event):

        self._mgr.GetPane("grid_content").Show(event.GetId() == ID_GridContent)
        self._mgr.GetPane("text_content").Show(event.GetId() == ID_TextContent)
        self._mgr.GetPane("tree_content").Show(event.GetId() == ID_TreeContent)
        self._mgr.GetPane("sizereport_content").Show(event.GetId() == ID_SizeReportContent)
        self._mgr.GetPane("html_content").Show(event.GetId() == ID_HTMLContent)
        self._mgr.Update()


    def CreateCommandLine(self):
        return wx.TextCtrl(self,-1, "open 1gcn", wx.Point(0, 0),
            wx.Size(150, 25), wx.TE_NOHIDESEL)


    def CreateTextCtrl(self):

        text = ("This is text box %d")%(self.n + 1)

        return wx.TextCtrl(self,-1, text, wx.Point(0, 0), wx.Size(150, 90),
                           wx.NO_BORDER | wx.TE_MULTILINE)



    def CreateGrid(self):

        grid = wx.grid.Grid(self, -1, wx.Point(0, 0), wx.Size(150, 250),
                            wx.NO_BORDER | wx.WANTS_CHARS)

        grid.CreateGrid(50, 20)

        return grid


    def CreateTreeCtrl(self):

        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        tree = wx.TreeCtrl(panel, -1, wx.Point(0, 0), wx.Size(160, 250),
                           wx.TR_HAS_BUTTONS | wx.TR_HIDE_ROOT)
        sizer.Add(tree, 1, wx.EXPAND)

        root = tree.AddRoot("Models")

        chimera_img = wx.Image("chimera48.png")
        chimera_bmp = wx.Bitmap(chimera_img.Rescale(16, 16))
        imglist = wx.ImageList(16, 16, True, 4)
        imglist.Add(wx.ArtProvider.GetBitmap(wx.ART_FOLDER, wx.ART_OTHER, wx.Size(16,16)))
        imglist.Add(wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, wx.Size(16,16)))
        imglist.Add(chimera_bmp)
        imglist.Add(wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_OTHER, wx.Size(16,16)))
        tree.AssignImageList(imglist)

        id = tree.AppendItem(root, "#1: 3wmg — Transport protein/inhibitor", 2)
        tree.AppendItem(id, "chain A: ATP-binding cassette, sub-family B, member 1", 3)
        tree.AppendItem(id, "chain B: anti-CMABCB1 peptide", 3)
        id = tree.AppendItem(root, "#2: 1mtx — Toxin [all submodels]", 2)
        tree.AppendItem(id, "chain A: margatoxin", 3)
        id = tree.AppendItem(root, "#3: 1bna — DNA", 2)
        tree.AppendItem(id, "chain A: DNA (5'-D(*CP*GP*CP*GP*AP*AP*TP*TP*CP*GP*CP*G)-3')", 3)
        tree.AppendItem(id, "chain B: DNA (5'-D(*CP*GP*CP*GP*AP*AP*TP*TP*CP*GP*CP*G)-3')", 3)
        id = tree.AppendItem(root, "#4: 4kdn — Viral protein", 2)
        tree.AppendItem(id, "chain A: hemagglutinin", 3)
        tree.AppendItem(id, "chain B: hemagglutinin", 3)
        tree.AppendItem(id, "chain C: hemagglutinin", 3)
        tree.AppendItem(id, "chain D: hemagglutinin", 3)
        tree.AppendItem(id, "chain E: hemagglutinin", 3)
        tree.AppendItem(id, "chain F: hemagglutinin", 3)
        id = tree.AppendItem(root, "#5: 1bhq — Cell adhesion", 2)
        tree.AppendItem(id, "chain 1: cell surface glycoprotein MAC-1 \N{GREEK SMALL LETTER ALPHA} subunit", 3)
        tree.AppendItem(id, "chain 2: cell surface glycoprotein MAC-1 \N{GREEK SMALL LETTER ALPHA} subunit", 3)

        from wx.lib.scrolledpanel import ScrolledPanel
        button_panel = ScrolledPanel(panel)
        sizer.Add(button_panel, 0, wx.EXPAND)
        button_sizer = wx.BoxSizer(wx.VERTICAL)
        from wx.lib.agw.balloontip import BalloonTip
        for button_text, tip in [
                ("activate", "Make selected models active\n(responsive to"
                    " mouse motions"),
                ("edit description...", "Edit description shown in model list"),
                ("clipping...", "Adjust per-model clipping plane"),
                ("close", "Close models"),
                ("copy/combine...", "Combine molecular models into new model"),
                ("expand submodels", "Show separate line in model table for"
                    " each submodel"),
                ("ramachandran plot...", None),
                ("show all atoms", None)
        ]:
            but = wx.Button(button_panel, -1, button_text)
            if tip is not None:
                bt = BalloonTip(message=tip)
                bt.SetTarget(but)
                bt.SetMessageFont(wx.Font(12, wx.FONTFAMILY_SWISS,
                    wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False))
                bt.SetStartDelay(1000)
            button_sizer.Add(but, 0, wx.EXPAND)
        button_panel.SetSizerAndFit(button_sizer)
        button_panel.SetAutoLayout(1)
        button_panel.SetupScrolling(False)

        panel.SetSizerAndFit(sizer)

        return panel


    def CreateSizeReportCtrl(self, width=80, height=80):

        ctrl = SizeReportCtrl(self, -1, wx.DefaultPosition,
                              wx.Size(width, height), self._mgr)
        return ctrl


    def CreateHTMLCtrl(self):
        """
        ctrl = wx.html.HtmlWindow(self, -1, wx.DefaultPosition, wx.Size(400, 300))
        if "gtk2" in wx.PlatformInfo:
            ctrl.SetStandardFonts()
        ctrl.SetPage(self.GetIntroText())
        """
        import wx.html2
        ctrl = wx.html2.WebView.New(self)
        ctrl.SetPage(self.GetIntroText(), "")
        return ctrl


    def CreateOpenGL(self):
        panel = wx.Panel(self)
        from GLCanvas import CubeCanvas
        canvas = CubeCanvas(panel)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(canvas, 1, wx.EXPAND)
        panel.SetSizerAndFit(sizer)
        return panel


    def CreateLog(self):
        import wx.html2
        ctrl = wx.html2.WebView.New(self, size=wx.Size(200,400))
        ctrl.SetPage(log, "")
        return ctrl


    def GetIntroText(self):
        return overview


# -- wx.SizeReportCtrl --
# (a utility control that always reports it's client size)

class SizeReportCtrl(wx.Control):

    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, mgr=None):

        wx.Control.__init__(self, parent, id, pos, size, wx.NO_BORDER)

        self._mgr = mgr

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)


    def OnPaint(self, event):

        dc = wx.PaintDC(self)

        size = self.GetClientSize()
        s = ("Size: %d x %d")%(size.x, size.y)

        dc.SetFont(wx.NORMAL_FONT)
        w, height = dc.GetTextExtent(s)
        height = height + 3
        dc.SetBrush(wx.WHITE_BRUSH)
        dc.SetPen(wx.WHITE_PEN)
        dc.DrawRectangle(0, 0, size.x, size.y)
        dc.SetPen(wx.LIGHT_GREY_PEN)
        dc.DrawLine(0, 0, size.x, size.y)
        dc.DrawLine(0, size.y, size.x, 0)
        dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2))

        if self._mgr:

            pi = self._mgr.GetPane(self)

            s = ("Layer: %d")%pi.dock_layer
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*1))

            s = ("Dock: %d Row: %d")%(pi.dock_direction, pi.dock_row)
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*2))

            s = ("Position: %d")%pi.dock_pos
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*3))

            s = ("Proportion: %d")%pi.dock_proportion
            w, h = dc.GetTextExtent(s)
            dc.DrawText(s, (size.x-w)/2, ((size.y-(height*5))/2)+(height*4))


    def OnEraseBackground(self, event):
        # intentionally empty
        pass


    def OnSize(self, event):

        self.Refresh()
        event.Skip()


ID_PaneBorderSize = wx.ID_HIGHEST + 1
ID_SashSize = ID_PaneBorderSize + 1
ID_CaptionSize = ID_PaneBorderSize + 2
ID_BackgroundColor = ID_PaneBorderSize + 3
ID_SashColor = ID_PaneBorderSize + 4
ID_InactiveCaptionColor =  ID_PaneBorderSize + 5
ID_InactiveCaptionGradientColor = ID_PaneBorderSize + 6
ID_InactiveCaptionTextColor = ID_PaneBorderSize + 7
ID_ActiveCaptionColor = ID_PaneBorderSize + 8
ID_ActiveCaptionGradientColor = ID_PaneBorderSize + 9
ID_ActiveCaptionTextColor = ID_PaneBorderSize + 10
ID_BorderColor = ID_PaneBorderSize + 11
ID_GripperColor = ID_PaneBorderSize + 12

class SettingsPanel(wx.Panel):

    def __init__(self, parent, frame):

        wx.Panel.__init__(self, parent, wx.ID_ANY, wx.DefaultPosition,
                          wx.DefaultSize)

        self._frame = frame

        vert = wx.BoxSizer(wx.VERTICAL)

        s1 = wx.BoxSizer(wx.HORIZONTAL)
        self._border_size = wx.SpinCtrl(self, ID_PaneBorderSize, "", wx.DefaultPosition, wx.Size(50,20))
        s1.Add((1, 1), 1, wx.EXPAND)
        s1.Add(wx.StaticText(self, -1, "Pane Border Size:"))
        s1.Add(self._border_size)
        s1.Add((1, 1), 1, wx.EXPAND)
        s1.SetItemMinSize(1, (180, 20))
        #vert.Add(s1, 0, wx.EXPAND | wxLEFT | wxBOTTOM, 5)

        s2 = wx.BoxSizer(wx.HORIZONTAL)
        self._sash_size = wx.SpinCtrl(self, ID_SashSize, "", wx.DefaultPosition, wx.Size(50,20))
        s2.Add((1, 1), 1, wx.EXPAND)
        s2.Add(wx.StaticText(self, -1, "Sash Size:"))
        s2.Add(self._sash_size)
        s2.Add((1, 1), 1, wx.EXPAND)
        s2.SetItemMinSize(1, (180, 20))
        #vert.Add(s2, 0, wx.EXPAND | wxLEFT | wxBOTTOM, 5)

        s3 = wx.BoxSizer(wx.HORIZONTAL)
        self._caption_size = wx.SpinCtrl(self, ID_CaptionSize, "", wx.DefaultPosition, wx.Size(50,20))
        s3.Add((1, 1), 1, wx.EXPAND)
        s3.Add(wx.StaticText(self, -1, "Caption Size:"))
        s3.Add(self._caption_size)
        s3.Add((1, 1), 1, wx.EXPAND)
        s3.SetItemMinSize(1, (180, 20))
        #vert.Add(s3, 0, wx.EXPAND | wxLEFT | wxBOTTOM, 5)

        #vert.Add(1, 1, 1, wx.EXPAND)

        b = self.CreateColorBitmap(wx.BLACK)

        s4 = wx.BoxSizer(wx.HORIZONTAL)
        self._background_color = wx.BitmapButton(self, ID_BackgroundColor, b, wx.DefaultPosition, wx.Size(50,25))
        s4.Add((1, 1), 1, wx.EXPAND)
        s4.Add(wx.StaticText(self, -1, "Background Color:"))
        s4.Add(self._background_color)
        s4.Add((1, 1), 1, wx.EXPAND)
        s4.SetItemMinSize(1, (180, 20))

        s5 = wx.BoxSizer(wx.HORIZONTAL)
        self._sash_color = wx.BitmapButton(self, ID_SashColor, b, wx.DefaultPosition, wx.Size(50,25))
        s5.Add((1, 1), 1, wx.EXPAND)
        s5.Add(wx.StaticText(self, -1, "Sash Color:"))
        s5.Add(self._sash_color)
        s5.Add((1, 1), 1, wx.EXPAND)
        s5.SetItemMinSize(1, (180, 20))

        s6 = wx.BoxSizer(wx.HORIZONTAL)
        self._inactive_caption_color = wx.BitmapButton(self, ID_InactiveCaptionColor, b,
                                                       wx.DefaultPosition, wx.Size(50,25))
        s6.Add((1, 1), 1, wx.EXPAND)
        s6.Add(wx.StaticText(self, -1, "Normal Caption:"))
        s6.Add(self._inactive_caption_color)
        s6.Add((1, 1), 1, wx.EXPAND)
        s6.SetItemMinSize(1, (180, 20))

        s7 = wx.BoxSizer(wx.HORIZONTAL)
        self._inactive_caption_gradient_color = wx.BitmapButton(self, ID_InactiveCaptionGradientColor,
                                                                b, wx.DefaultPosition, wx.Size(50,25))
        s7.Add((1, 1), 1, wx.EXPAND)
        s7.Add(wx.StaticText(self, -1, "Normal Caption Gradient:"))
        s7.Add(self._inactive_caption_gradient_color)
        s7.Add((1, 1), 1, wx.EXPAND)
        s7.SetItemMinSize(1, (180, 20))

        s8 = wx.BoxSizer(wx.HORIZONTAL)
        self._inactive_caption_text_color = wx.BitmapButton(self, ID_InactiveCaptionTextColor, b,
                                                            wx.DefaultPosition, wx.Size(50,25))
        s8.Add((1, 1), 1, wx.EXPAND)
        s8.Add(wx.StaticText(self, -1, "Normal Caption Text:"))
        s8.Add(self._inactive_caption_text_color)
        s8.Add((1, 1), 1, wx.EXPAND)
        s8.SetItemMinSize(1, (180, 20))

        s9 = wx.BoxSizer(wx.HORIZONTAL)
        self._active_caption_color = wx.BitmapButton(self, ID_ActiveCaptionColor, b,
                                                     wx.DefaultPosition, wx.Size(50,25))
        s9.Add((1, 1), 1, wx.EXPAND)
        s9.Add(wx.StaticText(self, -1, "Active Caption:"))
        s9.Add(self._active_caption_color)
        s9.Add((1, 1), 1, wx.EXPAND)
        s9.SetItemMinSize(1, (180, 20))

        s10 = wx.BoxSizer(wx.HORIZONTAL)
        self._active_caption_gradient_color = wx.BitmapButton(self, ID_ActiveCaptionGradientColor,
                                                              b, wx.DefaultPosition, wx.Size(50,25))
        s10.Add((1, 1), 1, wx.EXPAND)
        s10.Add(wx.StaticText(self, -1, "Active Caption Gradient:"))
        s10.Add(self._active_caption_gradient_color)
        s10.Add((1, 1), 1, wx.EXPAND)
        s10.SetItemMinSize(1, (180, 20))

        s11 = wx.BoxSizer(wx.HORIZONTAL)
        self._active_caption_text_color = wx.BitmapButton(self, ID_ActiveCaptionTextColor,
                                                          b, wx.DefaultPosition, wx.Size(50,25))
        s11.Add((1, 1), 1, wx.EXPAND)
        s11.Add(wx.StaticText(self, -1, "Active Caption Text:"))
        s11.Add(self._active_caption_text_color)
        s11.Add((1, 1), 1, wx.EXPAND)
        s11.SetItemMinSize(1, (180, 20))

        s12 = wx.BoxSizer(wx.HORIZONTAL)
        self._border_color = wx.BitmapButton(self, ID_BorderColor, b, wx.DefaultPosition,
                                             wx.Size(50,25))
        s12.Add((1, 1), 1, wx.EXPAND)
        s12.Add(wx.StaticText(self, -1, "Border Color:"))
        s12.Add(self._border_color)
        s12.Add((1, 1), 1, wx.EXPAND)
        s12.SetItemMinSize(1, (180, 20))

        s13 = wx.BoxSizer(wx.HORIZONTAL)
        self._gripper_color = wx.BitmapButton(self, ID_GripperColor, b, wx.DefaultPosition,
                                              wx.Size(50,25))
        s13.Add((1, 1), 1, wx.EXPAND)
        s13.Add(wx.StaticText(self, -1, "Gripper Color:"))
        s13.Add(self._gripper_color)
        s13.Add((1, 1), 1, wx.EXPAND)
        s13.SetItemMinSize(1, (180, 20))

        grid_sizer = wx.GridSizer(cols=2)
        grid_sizer.SetHGap(5)
        grid_sizer.Add(s1)
        grid_sizer.Add(s4)
        grid_sizer.Add(s2)
        grid_sizer.Add(s5)
        grid_sizer.Add(s3)
        grid_sizer.Add(s13)
        grid_sizer.Add((1, 1))
        grid_sizer.Add(s12)
        grid_sizer.Add(s6)
        grid_sizer.Add(s9)
        grid_sizer.Add(s7)
        grid_sizer.Add(s10)
        grid_sizer.Add(s8)
        grid_sizer.Add(s11)

        cont_sizer = wx.BoxSizer(wx.VERTICAL)
        cont_sizer.Add(grid_sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(cont_sizer)
        self.GetSizer().SetSizeHints(self)

        self._border_size.SetValue(frame.GetDockArt().GetMetric(wx.aui.AUI_DOCKART_PANE_BORDER_SIZE))
        self._sash_size.SetValue(frame.GetDockArt().GetMetric(wx.aui.AUI_DOCKART_SASH_SIZE))
        self._caption_size.SetValue(frame.GetDockArt().GetMetric(wx.aui.AUI_DOCKART_CAPTION_SIZE))

        self.UpdateColors()

        self.Bind(wx.EVT_SPINCTRL, self.OnPaneBorderSize, id=ID_PaneBorderSize)
        self.Bind(wx.EVT_SPINCTRL, self.OnSashSize, id=ID_SashSize)
        self.Bind(wx.EVT_SPINCTRL, self.OnCaptionSize, id=ID_CaptionSize)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_BackgroundColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_SashColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_InactiveCaptionColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_InactiveCaptionGradientColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_InactiveCaptionTextColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_ActiveCaptionColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_ActiveCaptionGradientColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_ActiveCaptionTextColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_BorderColor)
        self.Bind(wx.EVT_BUTTON, self.OnSetColor, id=ID_GripperColor)


    def CreateColorBitmap(self, c):
        image = wx.Image(25, 14)
        
        for x in xrange(25):
            for y in xrange(14):
                pixcol = c
                if x == 0 or x == 24 or y == 0 or y == 13:
                    pixcol = wx.BLACK

                image.SetRGB(x, y, pixcol.Red(), pixcol.Green(), pixcol.Blue())

        return image.ConvertToBitmap()


    def UpdateColors(self):

        bk = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_BACKGROUND_COLOUR)
        self._background_color.SetBitmapLabel(self.CreateColorBitmap(bk))

        cap = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_INACTIVE_CAPTION_COLOUR)
        self._inactive_caption_color.SetBitmapLabel(self.CreateColorBitmap(cap))

        capgrad = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_INACTIVE_CAPTION_GRADIENT_COLOUR)
        self._inactive_caption_gradient_color.SetBitmapLabel(self.CreateColorBitmap(capgrad))

        captxt = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_INACTIVE_CAPTION_TEXT_COLOUR)
        self._inactive_caption_text_color.SetBitmapLabel(self.CreateColorBitmap(captxt))

        acap = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_ACTIVE_CAPTION_COLOUR)
        self._active_caption_color.SetBitmapLabel(self.CreateColorBitmap(acap))

        acapgrad = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_ACTIVE_CAPTION_GRADIENT_COLOUR)
        self._active_caption_gradient_color.SetBitmapLabel(self.CreateColorBitmap(acapgrad))

        acaptxt = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_ACTIVE_CAPTION_TEXT_COLOUR)
        self._active_caption_text_color.SetBitmapLabel(self.CreateColorBitmap(acaptxt))

        sash = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_SASH_COLOUR)
        self._sash_color.SetBitmapLabel(self.CreateColorBitmap(sash))

        border = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_BORDER_COLOUR)
        self._border_color.SetBitmapLabel(self.CreateColorBitmap(border))

        gripper = self._frame.GetDockArt().GetColour(wx.aui.AUI_DOCKART_GRIPPER_COLOUR)
        self._gripper_color.SetBitmapLabel(self.CreateColorBitmap(gripper))


    def OnPaneBorderSize(self, event):

        self._frame.GetDockArt().SetMetric(wx.aui.AUI_DOCKART_PANE_BORDER_SIZE,
                                           event.GetInt())
        self._frame.DoUpdate()


    def OnSashSize(self, event):

        self._frame.GetDockArt().SetMetric(wx.aui.AUI_DOCKART_SASH_SIZE,
                                           event.GetInt())
        self._frame.DoUpdate()


    def OnCaptionSize(self, event):

        self._frame.GetDockArt().SetMetric(wx.aui.AUI_DOCKART_CAPTION_SIZE,
                                           event.GetInt())
        self._frame.DoUpdate()


    def OnSetColor(self, event):

        dlg = wx.ColourDialog(self._frame)

        dlg.SetTitle("Color Picker")

        if dlg.ShowModal() != wx.ID_OK:
            return

        var = 0
        if event.GetId() == ID_BackgroundColor:
            var = wx.aui.AUI_DOCKART_BACKGROUND_COLOUR
        elif event.GetId() == ID_SashColor:
            var = wx.aui.AUI_DOCKART_SASH_COLOUR
        elif event.GetId() == ID_InactiveCaptionColor:
            var = wx.aui.AUI_DOCKART_INACTIVE_CAPTION_COLOUR
        elif event.GetId() == ID_InactiveCaptionGradientColor:
            var = wx.aui.AUI_DOCKART_INACTIVE_CAPTION_GRADIENT_COLOUR
        elif event.GetId() == ID_InactiveCaptionTextColor:
            var = wx.aui.AUI_DOCKART_INACTIVE_CAPTION_TEXT_COLOUR
        elif event.GetId() == ID_ActiveCaptionColor:
            var = wx.aui.AUI_DOCKART_ACTIVE_CAPTION_COLOUR
        elif event.GetId() == ID_ActiveCaptionGradientColor:
            var = wx.aui.AUI_DOCKART_ACTIVE_CAPTION_GRADIENT_COLOUR
        elif event.GetId() == ID_ActiveCaptionTextColor:
            var = wx.aui.AUI_DOCKART_ACTIVE_CAPTION_TEXT_COLOUR
        elif event.GetId() == ID_BorderColor:
            var = wx.aui.AUI_DOCKART_BORDER_COLOUR
        elif event.GetId() == ID_GripperColor:
            var = wx.aui.AUI_DOCKART_GRIPPER_COLOUR
        else:
            return

        self._frame.GetDockArt().SetColor(var, dlg.GetColourData().GetColour())
        self._frame.DoUpdate()
        self.UpdateColors()



#----------------------------------------------------------------------

class TestPanel(wx.Panel):
    def __init__(self, parent, log):
        self.log = log
        wx.Panel.__init__(self, parent, -1)
        b = wx.Button(self, -1, "Show the wx.aui Demo Frame", (50,50))
        self.Bind(wx.EVT_BUTTON, self.OnButton, b)

    def OnButton(self, evt):
        frame = PyAUIFrame(self, wx.ID_ANY, "wx.aui wxPython Demo", size=(750, 590))
        frame.Show()

#----------------------------------------------------------------------

def runTest(frame, nb, log):
    win = TestPanel(nb, log)
    return win

#----------------------------------------------------------------------


overview = """\
<html>
<head>
<style>
html, body {
    height:100%;
    width:100%;
    margin:0px;
}
body {
    display:table;
    vertical-align:middle;
    text-align:center;
    background-color:black;
    font-size:x-large;
    color:white;
}
div {
    display:table-cell;
    vertical-align:middle;
}
</style>
</head>
<body>
<div>Graphics Window</div>
</body></html>
"""

log = """\
<html>
<head>
</head>
<body>
<p style="font-size:12px">Thu Apr 7 17:48 PDT 2014<p>
<ol style="font-size:12px;margin-left:5px;padding-left:10px">
<li><img src="file://localhost/Users/pett/src/chimera2/src/ui-proto/image1.png"></img>
Opened <a href=none.html">emd_1862.map</a> (#0), size 255<sup>3</sup>,
grid spacing 1.27&Aring;, contour level -363, minimum -17270,
maximum 14818, mean -6649.5, sd 3176.6, rms 7369.3, 16-bit integer
<img src="file://localhost/Users/pett/src/chimera2/src/ui-proto/image2.png" width="87" height="15"></img>
<li><img src="file://localhost/Users/pett/src/chimera2/src/ui-proto/image3.png"></img>
Opened <a href=none.html">1em9.pdb</a> (#1), 2 chains A B,
2257 atoms, rous sarcoma virus capsid protein: n-terminal domain,
weight 30.04 KDa, x-ray 2.05 &Aring;
</ol>
<p style="font-size:12px">The Apr 8 00:14 PDT 2014<p>
</body></html>
"""




if __name__ == '__main__':
    import sys,os
    import run
    run.main(['', os.path.basename(sys.argv[0])] + sys.argv[1:])

