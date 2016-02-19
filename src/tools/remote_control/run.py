#!/usr/bin/env python

# Python 2 test making an XMLRPC request to run a ChimeraX command

xmlrpc_port = 42184

from xmlrpclib import ServerProxy
s = ServerProxy(uri="http://127.0.0.1:%d/RPC2" % xmlrpc_port)

from sys import argv
status = s.run_command(argv[1])

print('Ran "%s", status %s' % (argv[1], status))
