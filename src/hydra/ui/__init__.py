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
