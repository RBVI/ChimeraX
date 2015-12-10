# vim: set expandtab shiftwidth=4 softtabstop=4:

def start_tool(session, bi):
    from . import fetch_cellpack
    fetch_cellpack.register_cellpack_fetch(session)
