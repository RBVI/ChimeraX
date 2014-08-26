def show_status(msg, append = False):
    print(msg)

def set_show_status(show_status_func):
    global show_status
    show_status = show_status_func

def show_info(msg, append = False):
    print(msg)

def set_show_info(show_info_func):
    global show_info
    show_status = show_info_func

def choose_window_toolkit():
    from sys import argv
    if argv and argv[-1] == 'wx':
        from .. import wx_ui as api
    else:
        from . import qt as api
    g = globals()
    for name in dir(api):
        if not name.startswith('_'):
            g[name] = getattr(api,name)
