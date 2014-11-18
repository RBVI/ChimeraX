# vim: set expandtab ts=4 sw=4:

import wx
from ..mousemodes import MouseModes

class WxMouseModes(MouseModes):

    def set_mouse_event_handlers(self):
        gw = self.graphics_window
        gw.opengl_canvas.Bind(wx.EVT_LEFT_DOWN,
            lambda e: self.dispatch_mouse_event(e, "left", 0))
        gw.opengl_canvas.Bind(wx.EVT_MIDDLE_DOWN,
            lambda e: self.dispatch_mouse_event(e, "middle", 0))
        gw.opengl_canvas.Bind(wx.EVT_RIGHT_DOWN,
            lambda e: self.dispatch_mouse_event(e, "right", 0))
        gw.opengl_canvas.Bind(wx.EVT_MOTION,
            lambda e: self.dispatch_mouse_event(e, None, 1))
        gw.opengl_canvas.Bind(wx.EVT_LEFT_UP,
            lambda e: self.dispatch_mouse_event(e, "left", 2))
        gw.opengl_canvas.Bind(wx.EVT_MIDDLE_UP,
            lambda e: self.dispatch_mouse_event(e, "middle", 2))
        gw.opengl_canvas.Bind(wx.EVT_RIGHT_UP,
            lambda e: self.dispatch_mouse_event(e, "right", 2))
        gw.opengl_canvas.Bind(wx.EVT_MOUSEWHEEL, self.wheel_event)

    def dispatch_mouse_event(self, event, button, fnum):
        if fnum == 0:
            # remember button for later drag events
            self.graphics_window.opengl_canvas.CaptureMouse()
        elif fnum == 1:
            if not event.Dragging():
                return
        elif fnum == 2:
            self.graphics_window.opengl_canvas.ReleaseMouse()
        if button is None:
            if event.LeftIsDown():
                button = "left"
            elif event.MiddleIsDown():
                button = "middle"
            elif event.RightIsDown():
                button = "right"
            else:
                return
            if not self.graphics_window.opengl_canvas.HasCapture():
                # a Windows thing; can lose mouse capture w/o mouse up
                return
        f = self.mouse_modes.get(button)
        if f and f[fnum]:
            f[fnum](event)

    def shift_down(self, event):
        return event.ShiftDown()

    def event_position(self, event):
        return event.GetPosition()

    def cursor_position(self):
        return self.graphics_window.ScreenToClient(wx.GetMousePosition())

    def wheel_value(self, event):
        return event.GetWheelRotation()/120.0   # Usually one wheel click is delta of 120
