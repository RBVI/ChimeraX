# vim: set expandtab shiftwidth=4 softtabstop=4:

def start_tool(session, ti):
    from . import fetch_cellpack
    fetch_cellpack.register()	# Register file reader
