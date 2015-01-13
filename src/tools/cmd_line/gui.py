# vim: set expandtab ts=4 sw=4:

class CmdLine:

    SIZE = (500, 25)

    def __init__(self, session):
        import weakref
        self._session = weakref.ref(session)
        import wx
        from chimera.core.ui.tool_api import ToolWindow
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
        session.ui.register_for_keystrokes(self)

    def forwarded_keystroke(self, event):
        if event.KeyCode == 13:
            self.OnEnter(event)
        else:
            self.text.EmulateKeyPress(event)

    def OnEnter(self, event):
        session = self._session()  # resolve back reference
        text = self.text.GetLineText(0)
        self.text.SelectAll()
        from chimera.core import cli
        import sys
        try:
            cmd = cli.Command(session, text, final=True)
            cmd.execute()
        except SystemExit as e:
            # TODO: somehow quit application
            raise
        except cli.UserError as err:
            rest = cmd.current_text[cmd.amount_parsed:]
            spaces = len(rest) - len(rest.lstrip())
            error_at = cmd.amount_parsed + spaces
            session.logger.info(cmd.current_text)
            session.logger.info("%s^" % ('.' * error_at))
            session.logger.info(str(err))
            session.logger.status(str(err))
        except:
            import traceback
            session.logger.error(traceback.format_exc())
