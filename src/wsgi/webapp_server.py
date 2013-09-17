#!/usr/bin/env python

"""
webapp_server: Server side for Chimera 2 web application
========================================================

Communicate with WSGI app to process web app requests
and return results.  Request and result formats are
described in `Wire Protocol <webapp.html#wire-protocol>`_.

The ``Server`` class is an abstract base class that should
be subclassed to provide custom functionality.  The intended
usage pattern is::

	s = Server()
	for tag, handler in tag_handler_list:
		s.register_handler(tag, handler)
	s.run()

Handlers are called with a single argument: the *value* field
of the request.  Return values from handlers should be a
dictionary with the same keys as the JSON return object
described in `Wire Protocol <webapp.html#wire-protocol>`_,
with the exception that the *id* field should _not_ be assigned.
"""

__all__ = [
	"Server",
]

# Make sure we use a protocol version that the WSGI script can handle
PICKLE_PROTOCOL = 2

class Server(object):
	"""Class for communicating with WSGI app and processing requests.

	This class establishes communications with WSGI app using
	`multiprocessing` `Connection` objects.  JSON requests are
	parsed and dispatched to registered handlers; return value
	from handlers are packaged into JSON and sent back to the
	WSGI app.  Requests are accepted on standard input and results
	are returned on standard output.

	Command line arguments are parsed and saved in the following
	attributes:

	session_file: name of file where session data may be found and/or saved
	"""

	def __init__(self):
		"""Constructor (extracts information from `sys.argv`).
		"""
		# Hack to make sure that multiprocessing only uses
		# pickle protocol 2 since the WSGI script is running
		# in Python 2 which does not handle protocol 3.
		import pickle
		pickle.HIGHEST_PROTOCOL = 2
		import sys
		self.session_file = sys.argv[1]
		from multiprocessing.connection import Connection
		self._inconn = Connection(0)
		self._outconn = Connection(1)
		self._log = None
		self._terminate = False
		self._handlers = dict()

	def set_log(self, log):
		"""Set file-like object for logging events and errors.
		"""
		self._log = log

	def register_handler(self, tag, handler):
		"""Register handler to process request and return result.

		For requests with the given *tag*, *handler* will be called
		with the corresponding *value*.  Multiple handlers may be
		registered for a single *tag* type.
		"""
		try:
			handler_list = self._handlers[tag]
		except KeyError:
			handler_list = list()
			self._handlers[tag] = handler_list
		handler_list.append(handler)

	def deregister_handler(self, tag, handler):
		"""Deregister request handler.
		"""
		try:
			handler_list = self._handlers[tag]
		except KeyError:
			pass
		else:
			try:
				handler_list.remove(handler)
			except ValueError:
				pass
			if not handler_list:
				del self._handlers[tag]

	def terminate(self):
		"""Terminate event loop started by ``run()``.

		This is usually called by one of the handlers to
		exit the currently active *run* event loop.
		"""
		self._terminate = True

	def run(self):
		"""Enter event loop to process requests.
		"""
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
				print("run", req, file=self._log)
				print("inconn", self._inconn, file=self._log)
				print("outconn", self._outconn, file=self._log)
				self._log.flush()
			try:
				v = self._process_request_batch(*req)
			except BaseException:
				if self._log:
					import traceback
					traceback.print_exc(file=self._log)
					self._log.flush()
				self._terminate = True
			else:
				if self._log:
					print("v", v, file=self._log)
					self._log.flush()
				self._outconn.send(v)
		self._terminate = False

	def _process_request_batch(self, req_data):
		# Return value should be a 4-tuple of:
		#
		# status: HTTP return status, *string*
		# ctype: MIME content type of return data, *string*
		# headers: additional HTTP headers, *list* of 2-tuples of
		#     ``(header_name, header_value)``
		# data: application data, *string*
		#
		# encoded using the ``dumps`` method from ``pickle``.

		import json
		try:
			req_list = json.loads(req_data)
		except ValueError:
			return self._decoding_failed()
		for req in req_list:
			if not isinstance(req, list) or len(req) != 3:
				return self._decoding_failed()
		reply_list = list()
		for req_id, req_tag, req_value in req_list:
			try:
				handler_list = self._handlers[req_tag]
			except KeyError:
				v = self._bad_tag(req_tag)
				v["id"] = req_id
				reply_list.append(v)
				continue
			for handler in handler_list:
				try:
					v = handler(req_value)
				except SystemExit as se:
					v = self._exited(se)
					self.terminate()
				except Exception:
					import traceback
					v = self._bad_handler(traceback.format_exc())
				v["id"] = req_id
				reply_list.append(v)

		try:
			reply = json.dumps(reply_list)
		except Exception:
			return self._encoding_failed()
		else:
			return self._success(reply)

	def _success(self, data):
		return ("200 OK", "application/json", None, data)

	def _decoding_failed(self):
		# Return value goes directly back to WSGI app
		return ("400 Bad Request", "text/plain", None,
						"Malformed request list")

	def _encoding_failed(self):
		# Return value goes directly back to WSGI app
		return ("500 JSON encoding failed", "text/plain", None,
						"JSON encoding failed")

	def _exited(self, exc):
		if exc.code == 0:
			return {
				"status": True,
				"stdout": "Server exited"
			}
		return {
			"status": False,
			"stderr": "Server died: %s" % exc.code
		}

	def _bad_tag(self, tag):
		# Return value is included in reply list
		return {
			"status": False,
			"stderr": "Unregistered tag: %s" % tag,
		}

	def _bad_handler(self, text):
		# Return value is included in reply list
		return {
			"status": False,
			"stderr": text,
		}
