# vi: set expandtab ts=4 sw=4:

from chimera.core.tools import ToolInstance


class CommandLine(ToolInstance):

    SIZE = (500, 25)
    VERSION = 1

    record_label = "Command History..."
    compact_label = "Remove duplicate consecutive commands"

    def __init__(self, session, **kw):
        super().__init__(session, **kw)
        self.tool_window = session.ui.create_main_tool_window(self,
                                      size=self.SIZE, destroy_hides=True)
        parent = self.tool_window.ui_area
        import wx
        self.text = wx.ComboBox(parent, size=self.SIZE,
                                style=wx.TE_PROCESS_ENTER | wx.TE_NOHIDESEL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.text, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.history_dialog = _HistoryDialog(self)
        self.text.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)
        self.text.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.tool_window.manage(placement="bottom")
        self.history_dialog.populate()
        session.ui.register_for_keystrokes(self)
        session.tools.add([self])
        self._last_thumb = None

    def forwarded_keystroke(self, event):
        if event.KeyCode == 13:
            self.OnEnter(event)
        elif event.KeyCode == 315:        # Up arrow
            self.session.selection.promote()
        elif event.KeyCode == 317:        # Down arrow
            self.session.selection.demote()
        elif event.KeyCode == 27:         # Escape
            self.session.keyboard_shortcuts.enable_shortcuts()
        else:
            self.text.EmulateKeyPress(event)

    def OnEnter(self, event):
        session = self.session
        logger = session.logger
        text = self.text.Value
        logger.status("")
        from chimera.core import cli
        for cmd_text in text.split("\n"):
            if not cmd_text:
                continue
            self.text.SetItems([cmd_text] + list(self.text.GetItems()[:19]))
            try:
                cmd = cli.Command(session, cmd_text, final=True)
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
                session.logger.info(cmd_text)
                if log_thumb:
                    session.logger.info("graphics image", image=thumb)
        self.text.SetValue(text)
        self.text.SelectAll()

    def OnKeyDown(self, event):
        # intercept up/down arrow
        if event.KeyCode == 315:  # up arrow
            self.session.logger.info("Up arrow")
        elif event.KeyCode == 317:  # down arrow
            self.session.logger.info("Down arrow")
        else:
            # pass through other keys
            event.Skip()

    #
    # Implement session.State methods if deriving from ToolInstance
    #
    def take_snapshot(self, session, flags):
        version = self.VERSION
        data = {"shown": self.tool_window.shown}
        return [version, data]

    def restore_snapshot(self, phase, session, version, data):
        from chimera.core.session import State, RestoreError
        if version != self.VERSION:
            raise RestoreError("unexpected version")
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

class _HistoryDialog:
    def __init__(self, controller):
        # make dialog hidden initially
        self.controller = controller
        self.window = controller.session.ui.create_child_tool_window(
            controller, title="Command History", destroy_hides=True)

        parent = self.window.ui_area
        import wx
        self.listbox = wx.ListBox(parent,
                                style=wx.LB_EXTENDED | wx.LB_NEEDED_SB)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.listbox, 1, wx.EXPAND)
        parent.SetSizerAndFit(sizer)
        self.window.manage(placement=None)
        self.window.shown = False

    def populate(self):
        pass # waiting for Greg to implement non-auto-unique-ifying history
