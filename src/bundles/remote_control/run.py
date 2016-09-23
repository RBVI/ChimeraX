#!/usr/bin/env python

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Python 2 test making an XMLRPC request to run a ChimeraX command

xmlrpc_port = 42184

from xmlrpclib import ServerProxy
s = ServerProxy(uri="http://127.0.0.1:%d/RPC2" % xmlrpc_port)

from sys import argv
status = s.run_command(argv[1])

print('Ran "%s", status %s' % (argv[1], status))
