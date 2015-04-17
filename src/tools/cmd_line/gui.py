# vi: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance


class CmdLine(ToolInstance):

    SIZE = (500, 25)
    VERSION = 1

    def __init__(self, session, **kw):
        super().__init__(session, **kw)
        from chimera.core.ui.tool_api import ToolWindow
        self.tool_window = ToolWindow("Command Line", session,
                                      size=self.SIZE, destroy_hides=True)
        parent = self.tool_window.ui_area
        import wx
        self.text = wx.TextCtrl(parent, size=self.SIZE,
                                style=wx.TE_PROCESS_ENTER | wx.TE_NOHIDESEL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.text, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.text.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)
        self.tool_window.manage(placement="bottom")
        session.ui.register_for_keystrokes(self)
        session.tools.add([self])
        self._last_thumb = None

    def forwarded_keystroke(self, event):
        if event.KeyCode == 13:
            self.OnEnter(event)
        elif event.KeyCode == 315:        # Up arrow
            self.session.models.promote_selection()
        elif event.KeyCode == 317:        # Down arrow
            self.session.models.demote_selection()
        else:
            self.text.EmulateKeyPress(event)

    def OnEnter(self, event):
        session = self.session
        logger = session.logger
        text = self.text.GetLineText(0)
        self.text.SelectAll()
        logger.status("")
        from chimera.core import cli
        try:
            cmd = cli.Command(session, text, final=True)
            cmd.execute()
        except SystemExit:
            # TODO: somehow quit application
            raise
        except cli.UserError as err:
            rest = cmd.current_text[cmd.amount_parsed:]
            spaces = len(rest) - len(rest.lstrip())
            error_at = cmd.amount_parsed + spaces
            text = "<pre>%s<br>\n%s^<br>\n%s\n</pre>" % (
                cmd.current_text, '.' * error_at, str(err))
            logger.info(text, is_html=True)
            logger.status(str(err))
        except:
            import traceback
            session.logger.error(traceback.format_exc())
        else:
            thumb = session.main_view.image(width=100, height=100)
            log_thumb = False
            from time import time
            s_t = time()
            if thumb.getcolors(1) is None:
                # image not just a solid background color;
                # ensure it differs from previous thumbnail
                thumb_data = thumb.tostring()
                if thumb_data != self._last_thumb:
                    self._last_thumb = thumb_data
                    log_thumb = True
            else:
               self._last_thumb = None
            session.logger.info(text)
            if log_thumb:
                session.logger.info("graphics image", image=thumb)

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        version = self.VERSION
        data = {"shown": self.tool_window.shown}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        from chimera.core.session import State
        if phase == State.PHASE1:
            # All the action is in phase 2 because we do not
            # want to restore until all objects have been resolved
            pass
        else:
            self.display(data["shown"])

    def reset_state(self):
        self.tool_window.shown = True

    #
    # Override ToolInstance methods
    #
    def delete(self):
        session = self.session
        session.ui.deregister_for_keystrokes(self)
        self.tool_window.shown = False
        self.tool_window.destroy()
        session.tools.remove([self])
        super().delete()

    def display(self, b):
        """Show or hide command line user interface."""
        self.tool_window.shown = b
