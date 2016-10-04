# vim: set expandtab shiftwidth=4 softtabstop=4:

_server = None

def _get_server():
    global _server
    if _server is None:
        return None
    if _server.httpd is None:
        _server = None
    return _server

def start_server(session, port=None, ssl=None):
    global _server
    server = _get_server()
    if server is not None:
        session.logger.error("REST server is already running")
    else:
        from .server import RESTServer
        _server = RESTServer(session)
        # Run code will report port number
        _server.start(port, ssl)
from chimerax.core.commands import CmdDesc, IntArg, BoolArg
start_desc = CmdDesc(keyword=[("port", IntArg),
                              ("ssl", BoolArg),
                             ],
                     synopsis="Start REST server")

def report_port(session):
    server = _get_server()
    addr = server.server_address if _server else None
    if addr is None:
        session.logger.info("REST server is not running")
    else:
        session.logger.info("REST server is listening on host %s port %s" % addr)
from chimerax.core.commands import CmdDesc, IntArg
port_desc = CmdDesc(synopsis="Report REST server port")

def stop_server(session):
    global _server
    server = _get_server()
    if server is None:
        session.logger.info("REST server is not running")
    else:
        server.terminate()
        _server = None
        session.logger.info("REST server stopped")
from chimerax.core.commands import CmdDesc, IntArg
stop_desc = CmdDesc(synopsis="Stop REST server")
