def start_oculus(session):

    oc = session.oculus
    if oc is None:
        # Create separate graphics window for rendering to Oculus Rift.
        # Don't show window until after oculus started, otherwise rendering uses wrong viewport.
        from ...ui.qt import graphicswindow as gw
        session.oculus_window = win = gw.Secondary_Graphics_Window('Oculus Rift View', session, show = False)
        from track import Oculus_Rift
        session.oculus = oc = Oculus_Rift(session.view)
        if oc.connected:
            # Move window to oculus screen and switch to full screen mode.
            w,h = oc.display_size()
            win.full_screen(w,h)
            # Set redraw timer for 1 msec to minimize dropped frames.
            # In Qt 5.2 interval of 5 or 10 mseconds caused dropped frames on 2 million triangle surface,
            # but 1 or 2 msec produced no dropped frames.
            session.main_window.graphics_window.set_redraw_interval(1)
        msg = 'started oculus head tracking ' if oc.connected else 'failed to start oculus head tracking'
        session.show_status(msg)
        session.show_info(msg)

def stop_oculus(session):

    oc = session.oculus
    if oc:
        oc.close()
        session.oculus = None
        session.oculus_window.close()
        session.oculus_window = None
        session.main_window.graphics_window.set_redraw_interval(10)

def oculus_command(enable = None, panSpeed = None, session = None):

    if not enable is None:
        if enable:
            start_oculus(session)
        else:
            stop_oculus(session)

    if not panSpeed is None:
        oc = session.oculus
        if oc:
            oc.panning_speed = panSpeed
