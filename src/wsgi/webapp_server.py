#!/usr/bin/env python

"""
webapp_server: Server side for Chimera 2 web application
========================================================

Communicate with WSGI app to process web app commands
and return results.

The ``Server`` class is an abstract base class that should
be subclassed to provide custom functionality.  The intended
usage pattern is::

	class MyServer(Server):
		def __init__(self, *args):
			Server.__init__(self)
			# Customization goes here
		def process_request(self, req):
			# Customization goes here

	s = MyServer()
	s.run()
"""

__all__ = [
	"Server",
]

class Server(object):
	"""Connect with WSGI app and process commands.

	This class establishes communications with WSGI app using
	`multiprocessing` `Connection` objects.  Commands are
	accepted on standard input and results are returned on
	standard output.

	Command line arguments are parsed and saved in the following
	attributes:

	session_file: name of file where session data may be found and/or saved
	"""

	def __init__(self):
		"""Constructor (extracts information from `sys.argv`)."""
		import sys, os.path
		session_dir = sys.argv[1]
		session_name = sys.argv[2]
		self.session_file = os.path.join(session_dir, session_file)
		from _multiprocessing import Connection
		self._inconn = Connection(0)
		self._outconn = Connection(1)
		self._log = None
		self._terminate = False

	def set_log(self, log):
		"""Set file-like object for logging events and errors."""
		self._log = log

	def terminate(self):
		"""Terminate event loop started by ``run()``."""
		self._terminate = True

	def run(self):
		"""Enter event loop reading and dispatching commands."""
		while not self._terminate:
			try:
				req = self._inconn.recv()
			except EOFError:
				req = None
			if not req:
				# Terminate if WSGI app is gone since we
				# will never get more input.
				break
			if self._log:
				print >> self._log, "run", req
				print >> self._log, "inconn", self.inconn
				print >> self._log, "outconn", self.outconn
			try:
				v = self.processRequest(*req)
			except:
				if self._log:
					import traceback
					traceback.print_exc(file=self._log)
					self._log.flush()
			if self._log:
				print >> self._log, "v", v
			self._outconn.send(v)
		self._terminate = False

	def process_request(self, req):
		"""Process web app request and return results.

		This method should be overridden in a subclass.
		Return value should be a 4-tuple of:

		status: HTTP return status, *string*
		ctype: MIME content type of return data, *string*
		headers: additional HTTP headers, *list* of 2-tuples of
		    ``(header_name, header_value)``
		data: application data, *string*

		encoded using the ``dumps`` method from ``pickle``.
		"""
		import cPickle as pickle
		return pickle.dumps(("500 Internal server error",
					"text/html", None, self.ERROR_MESSAGE))

	ERROR_MESSAGE = """<html>
<head><title>Chimera 2 WSGI Error</title></head>
<body><h1>Chimera 2 WSGI Error</h1>
<p><code>process_request</code> was not overridden in subclass.</p></body>
</html>
"""
