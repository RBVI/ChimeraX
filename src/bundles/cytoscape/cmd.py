# vim: set expandtab shiftwidth=4 softtabstop=4:

_host = "localhost"
_port = 12345

import http.client, urllib.parse
#
# Public methods for interacting with Cytoscape
#

# Return all namespaces as a list
def get_namespaces(session):
    response = _send_command(session, None, None, None)
    return _get_string(response)

# Return all commands for a given namespace as a list
def get_commands(session, namespace):
    response = _send_command(session, namespace, None, None)
    return _get_string(response)

# Return all arguments for a given namespace and command as a list
def get_arguments(session, namespace, command):
    response = _send_command(session, namespace, command, None)
    return _get_string(response)

# Execute a Cytoscape command
def send_command(session, namespace, command, arguments):
    response = _send_command(session, namespace, command, arguments)
    return _get_string(response)

def _send_command(session, namespace, command, arguments):
    conn = http.client.HTTPConnection(_host, _port)
    url = '/v1/commands/'
    if namespace != None:
        url += namespace+'/'
        if command != None:
            url += urllib.parse.quote(command)+'/'
            if arguments != None:
                args = urllib.parse.urlencode(arguments)
                url += "?"+args
    conn.request('GET', url)
    print ("Sending "+url+" to Cytoscape")
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError("HTTP error return from Cytoscape: "+response.status)
    data = response.read()
    conn.close()
    return data.decode()

def _get_string(response):
    lines = response.split("\n")
    return lines

def _report_response(session, response):
    lines = response.split("\n")
    for line in lines:
        session.logger.info(line,is_html=True)

import shlex
def _parse_args(arguments):
    # Split on spaces, preserving quoted spaces
    arg_dict = {}
    arg_list = shlex.shlex(arguments, posix=True)
    arg_list.whitespace += ','
    arg_list.whitespace_split = True
    for arg in list(arg_list):
        (key,value) = arg.split("=")
        arg_dict[key] = value
    return arg_dict

def connect_cyto(session, port=None):
    global _host, _port
    if _host is not None:
        session.logger.error("Already connected to Cytoscape")
    else:
        if port != None:
            _port = int(port)
        _host = "localhost"
        session.logger.info("Cytoscape URL: http://%s:%d/"%(_host, _port))
    return
from chimerax.core.commands import CmdDesc, IntArg, BoolArg
connect_desc = CmdDesc(keyword=[("port", IntArg)],
                       synopsis="Connect to Cytoscape")

def send_cyto(session, namespace=None, command=None, arguments=None):
    global _host
    response = None
    if (_host == None):
        session.logger.error("Not connected to Cytoscape")
        return
    if (namespace == None):
        response = _send_command(session, None, None, None)
    elif (command == None):
        response = _send_command(session, namespace, None, None)
    elif (arguments == None):
        response = _send_command(session, namespace, command, None)
    else:
        args = _parse_args(arguments)
        response = _send_command(session, namespace, command, args)
    if response == None:
        return
    _report_response(session, response)

from chimerax.core.commands import CmdDesc, StringArg, RestOfLine
send_desc = CmdDesc(keyword=[("namespace", StringArg),
                             ("command", StringArg)],
                    optional=[("arguments", RestOfLine)],
                    synopsis="Connect to Cytoscape")
