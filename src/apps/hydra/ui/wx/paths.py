# vim: set expandtab ts=4 sw=4:

def user_settings_path(app_name="ChimeraX", vendor="RBVI"):
    import appdirs
    return appdirs.user_data_dir(app_name, vendor)
