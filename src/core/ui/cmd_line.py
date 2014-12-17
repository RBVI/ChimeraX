# vim: set expandtab ts=4 sw=4:

class CmdLine:

    SIZE = (500, 25)

    def __init__(self, session):
        self.session = session
        import wx
        from .tool_api import ToolWindow
        self.tool_window = ToolWindow("Command Line", "General", session,
            size=self.SIZE)
        parent = self.tool_window.ui_area
        self.text = wx.TextCtrl(parent, size=self.SIZE,
            style=wx.TE_PROCESS_ENTER | wx.TE_NOHIDESEL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.text, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.text.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)
        self.tool_window.manage(placement="bottom")

    def OnEnter(self, event):
        text = self.text.GetLineText(0)
        self.text.SelectAll()
        from chimera.core import cli
        try:
             cmd = cli.Command(self.session, text, final=True)
        except SystemExit as e:
            pass  # TODO: somehow quit application
        except cli.UserError:
            rest = cmd.current_text[cmd.amount_parsed:]
            spaces = len(rest) - len(rest.lstrip())
            error_at = cmd.amount_parsed + spaces
            # TODO: send the following to the reply log?
            print(cmd.current_text)
            print("%s^" % ('.' * error_at))
            print(err)
