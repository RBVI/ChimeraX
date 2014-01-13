def show_status(msg, append = False):
    print(msg)

def set_show_status(show_status_func):
    global show_status
    show_status = show_status_func
