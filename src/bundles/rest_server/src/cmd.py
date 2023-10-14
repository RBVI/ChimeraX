# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

_server = None

def _get_server():
    global _server
    if _server is None:
        return None
    if _server.httpd is None:
        _server = None
    return _server

def start_server(session, port=None, ssl=None, json=False):
    """If 'json' is True, then the return value from a command will be a JSON object with the following
       name/value pairs:

       (name) json values
       (value) A list of json strings.  Typically a list of one string, but if the "command" executed is
            actually a series of commands separated by semicolons, then there will be a corresponding
            number of strings.  The contents of string varies from command to command, and should be
            documented in the doc string for the function that actually implements the commands.  Commands
            that don't (yet) implement JSON return values will have 'null' as their JSON value.

       (name) python values
       (value) A list of Python objects, with a list structure similar to that for 'json values'.  If the
            command's normal return value is None or can't be encoded into JSON, then the value will be
            'null'.

       (name) log messages
       (value) A JSON object, with names corresponding to log levels as given in chimerax.core.logger.Log.
            LEVEL_DESCRIPTS, and values that are lists of messages logged at that level during command
            execution.
    """

    global _server
    server = _get_server()
    if server is not None:
        session.logger.error("REST server is already running")
    else:
        from .server import RESTServer
        _server = RESTServer(session)
        # Run code will report port number
        _server.start(port, ssl, json)
from chimerax.core.commands import CmdDesc, IntArg, BoolArg
start_desc = CmdDesc(keyword=[("port", IntArg),
                              ("ssl", BoolArg),
                              ("json", BoolArg),
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

def stop_server(session, quiet=False):
    global _server
    server = _get_server()
    if server is None:
        if not quiet:
            session.logger.info("REST server is not running")
    else:
        server.terminate()
        _server = None
        if not quiet:
            session.logger.info("REST server stopped")
from chimerax.core.commands import CmdDesc, BoolArg
stop_desc = CmdDesc(synopsis="Stop REST server",
                    keyword=[("quiet", BoolArg)])
