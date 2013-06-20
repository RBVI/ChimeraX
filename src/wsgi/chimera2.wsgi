DefaultMsg = "Use HTML page to access Chimera 2 sessions"

#
# WSGI application code
#
Status_Okay = "200 OK"
Status_Redirect = "301 Redirect to HTML page"
Status_BadRequest = "400 Bad request"
Status_InternalServerError = "500 Internal server error"

ContentType_PlainText = "text/plain"
ContentType_HTML = "text/html"
ContentType_JSON = "application/json"

class WSGIError(Exception):
	"""WSGI exception class.

	Attribute ``wsgiData`` contains a 4-tuple of::

	HTTP status - *string*
	HTTP content type - *string*
	HTTP headers - *list* of key-value pairs
	page data - *string*
	"""
	def __init__(self, status, output, contentType=None, headers=None):
		Exception.__init__(self, status)
		if contentType is None:
			contentType = ContentType_PlainText
		self.wsgiData = (status, contentType, headers, output)

class App(object):
	"""WSGI application class.

	Maintains a session store and redirects web requests based on 
	the value of ``action`` parameters.
	"""
	def __init__(self):
		import os.path
		dbName = os.path.join(os.path.dirname(__file__),
						"sessions", "sessions.db")
		self.sessions = SessionStore(dbName)
		self.reqCount = 0

	def __del__(self):
		try:
			self.sessions.close()
		except:
			pass

	def __call__(self, environ, start_response):
		self.reqCount += 1
		import cgi
		fs = cgi.FieldStorage(fp=environ['wsgi.input'],
					environ=environ,
					keep_blank_values=True)
		try:
			status, ctype, hdrs, output = self.process(environ, fs)
			#status, ctype, output = self.showInput(environ, fs)
		except WSGIError, e:
			status, ctype, hdrs, output = e.wsgiData
		except:
			status = Status_InternalServerError
			ctype = ContentType_PlainText
			hdrs = None
			import traceback
			output = [ "<pre>\n", traceback.format_exc(), "</pre>" ]
		if isinstance(output, basestring):
			length = len(output)
			output = [ output ]
		else:
			length = sum([ len(o) for o in output ])
		headers = [ ("Content-Type", ctype),
				("Content-Length", str(length)) ]
		if hdrs:
			headers += hdrs
		start_response(status, headers)
		return output

	def process(self, environ, fs):
		"""Process a single web request."""
		# environ - dictionary such as os.environ
		# fs - instance of cgi.FieldStorage
		try:
			action = self._getField(fs, "action")
		except WSGIError:
			return self._default_action(environ)
		try:
			f = getattr(self, "_process_%s" % action)
		except AttributeError:
			raise WSGIError(Status_BadRequest,
					"unknown action: %s" % action)
		else:
			return f(environ, fs)

	def _getenv(self, environ, key, emptyOkay=False):
		if key not in environ:
			if emptyOkay:
				return ""
			raise WSGIError(Status_BadRequest,
				"Environment variable \"%s\" missing" % key)
		v = environ[key]
		if not emptyOkay and v == "":
			raise WSGIError(Status_BadRequest,
				"Environment variable \"%s\" undefined" % key)
		return v

	def _getField(self, fs, key, emptyOkay=False):
		if key not in fs:
			if emptyOkay:
				return ""
			raise WSGIError(Status_BadRequest,
				"Form field \"%s\" missing" % key)
		v = fs[key].value
		if not emptyOkay and v == "":
			raise WSGIError(Status_BadRequest,
				"Form field \"%s\" undefined" % key)
		return v

	def _default_action(self, environ):
		base = self._getenv(environ, "SCRIPT_URI").rsplit('/', 1)[0]
		frontend_url = base + "/chimera2_webapp.html"
		return (Status_Redirect,
				ContentType_PlainText,
				[ ( "Location", url ) ],
				DefaultMsg)
	
	#
	# These methods are named "_process_XXX" where XXX is the
	# name of the action.  They are dispatched from self.process()
	# and always receive the environment dictionary and field
	# storage instance as their arguments.  Their return values
	# should be 3-tuples of the form (status, content type, output).
	#
	
	#
	# These methods are named "_process_XXX" where XXX is the
	# name of the action.  They are dispatched from self.process()
	# and always receive the environment dictionary and field
	# storage instance as their arguments.  Their return values
	# should be 3-tuples of the form (status, content type, output).
	#

	def _process_list(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		import StringIO
		sf = StringIO.StringIO()
		sessionList = self.sessions.getSessionList(account)
		if not sessionList:
			print >> sf, "<p>There are no sessions for <i>%s</i>" % account
		else:
			print >> sf, "<h1>Sessions for <i>%s</i></h1>" % account
			print >> sf, "<table>"
			print >> sf, "<tr>"
			print >> sf, "<th>Name</th><th>Type</th><th>Last Access</th>"
			print >> sf, "</tr>"
			import time
			for s in sessionList:
				print >> sf, "<tr>"
				print >> sf, "<td>%s</th>" % s.name
				print >> sf, "<td>%s</th>" % s.type
				print >> sf, "<td>%s</th>" % s.accessTime()
				print >> sf, "</tr>"
			print >> sf, "</table>"
		output = sf.getvalue()
		return Status_Okay, ContentType_HTML, None, output

	def _process_jlist(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		sessionList = self.sessions.getSessionList(account)
		sList = [ { "name": s.name,
				"access": s.accessTime() }
				for s in sessionList ]
		import json
		output = json.dumps(sList)
		return Status_Okay, ContentType_JSON, None, output

	def _process_create(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		name = self._getField(fs, "session")
		password = self._getField(fs, "password", emptyOkay=True)
		with self.sessions.lock:
			sessionList = self.sessions.getSessionList(account)
			for s in sessionList:
				if s.name == name:
					raise WSGIError(Status_BadRequest,
						"Session \"%s\" already exists"
						% name)
			else:
				s = Session(name, password)
				sessionList.insert(0, s)
				self.sessions.updateSessionList(account,
								sessionList)
				output = "<p>Session <i>%s</i> created" % name
		return Status_Okay, ContentType_HTML, None, output

	def _process_delete(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		session = self._getField(fs, "session")
		password = self._getField(fs, "password", emptyOkay=True)
		with self.sessions.lock:
			s = self.sessions.findSession(account, session,
								password)
			if not s:
				raise WSGIError(Status_BadRequest,
					"No session named \"%s\"" % session)
			sessionList.remove(s)
			self.sessions.updateSessionList(account, sessionList)
		output = "<p>Session <i>%s</i> deleted" % session
		return Status_Okay, ContentType_HTML, None, output

	def _process_call(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		session = self._getField(fs, "session")
		password = self._getField(fs, "password", emptyOkay=True)
		s = self.sessions.findSession(account, session, password)
		if not s:
			raise WSGIError(Status_BadRequest,
					"No session named \"%s\"" % session)
		command = self._getField(fs, "command")
		return s.call(command)

	def _process_pwd(self, environ, fs):
		import os
		output = os.getcwd()
		return Status_Okay, ContentType_PlainText, None, output

	def _process_env(self, environ, fs):
		env = '\n'.join([ "<li>%s: %s</li>" % item
					for item in environ.iteritems() ])
		import sys
		path = '\n'.join([ "<li>%s</li>" % d for d in sys.path ])
		output = ( "<h2>Environment</h2><ul>" + env + "</ul>" +
			"<h2>Path</h2><ul>" + path + "</ul>" +
			"<h2>__file__</h2><ul><li>" + __file__ + "</li></ul>" +
			"<h2>__name__</h2><ul><li>" + __name__ + "</li></ul>")
		return Status_Okay, ContentType_HTML, None, output

	#
	# Rest of methods are for debugging
	#
	def showInput(self, environ, fs):
		import StringIO
		sf = StringIO.StringIO()
		print >> sf, "self:", self
		import os
		print >> sf, "id:", os.getuid(), os.getgid()
		import thread
		print >> sf, "thread:", thread.get_ident()
		print >> sf, "request #:", self.reqCount
		self._dumpFieldStorage(sf, fs)
		print >> sf, "extra path information:", environ["PATH_INFO"]
		print >> sf, "query string:", environ["QUERY_STRING"]
		print >> sf, "user:", environ["REMOTE_USER"]
		print >> sf, "host:", environ["REMOTE_ADDR"]
		self._dumpEnviron(sf, environ)
		output = sf.getvalue()
		return Status_Okay, ContentType_PlainText, output

	def _dumpFieldStorage(self, f, fs):
		print >> f, ""
		print >> f, "--- FieldStorage:"
		for k in fs.keys():
			print >> f, "%s: %s" % (k, fs[k].value)
		print >> f, "--- FieldStorage"
		print >> f, ""

	def _dumpEnviron(self, f, env):
		print >> f, ""
		print >> f, "--- Environment:"
		for item in env.iteritems():
			print >> f, "%s: %s" % item
		print >> f, "--- Environment"
		print >> f, ""

class Session(object):
	"""Session state class.

	Manages the state for a single session.  Public attributes are::

	``name`` - session name, *string*
	``password`` - session password, *string*
	``last_access`` - last access time, *int*
	"""
	def __init__(self, name, password):
		self._initRuntime()
		self._initState(name, password)

	def _initRuntime(self):
		self.pipe = None
		self.process = None
		import threading
		self.lock = threading.RLock()

	def _initState(self, name, password):
		self.name = name
		self.password = password
		self.updateAccess()
		self._sessionDir = os.path.join(os.path.dirname(__file__),
								"sessions")

	def __getstate__(self):
		return (self.name, self.password, self.lastAccess)

	def __setstate__(self, values):
		self._initRuntime()
		name, password, lastAccess = values
		self._initState(name, password)
		# We ignore saved access time because we just accessed it

	def __str__(self):
		return "%s <%s>" % (self.name, self.accessTime())

	def __del__(self):
		self.disconnect()

	def updateAccess(self):
		"""Set last access time for this session to current time."""
		import time
		self.lastAccess = time.time()

	def accessTime(self):
		"""Return last access time as a *string*."""
		import time
		return time.ctime(self.lastAccess)

	def call(self, *args):
		"""Pass ``args`` to session backend and return results."""
		self.lock.acquire()
		#print "%s.call(%s)" % (self.name, str(args))
		if self.process and not self.process.is_alive():
			try:
				# Make sure it's dead and gone
				self.process.join(1)
			except:
				pass
			self.process = None
			self.pipe = None
		if self.process is None:
			from multiprocessing import Pipe, Process
			toChild = Pipe()
			fromChild = Pipe()
			self.pipe = (fromChild[0], toChild[1])
			self.process = Process(target=backend,
						args=(self._sessionDir,
							self.name,
							toChild,
							fromChild))
			self.process.start()
			#print "%s - process %d started" % (self.name,
			#				self.process.pid)
			toChild[0].close()
			fromChild[1].close()
		self.pipe[1].send(args)
		output = self.pipe[0].recv()
		self.lock.release()
		import cPickle as pickle
		return pickle.loads(output)

	def disconnect(self):
		"""Terminate backend if present."""
		#print "disconnect", self.name
		#import traceback, sys
		#traceback.print_stack(file=sys.stdout)
		if not self.pipe:
			return
		self.pipe[1].send(None)
		self.pipe = None
		self.process.join(30)
		if self.process.is_alive():
			self.process.terminate()
			print "%s - process %d killed" % (self.name,
							self.process.pid)
		#else:
		#	print "%s - process %d finished" % (self.name,
		#					self.process.pid)
		self.process = None

class SessionStore(object):
	"""Session store class.

	A container class with dictionary semantics supporting
	persistence across process invocations.  The keys are account
	names and the values are lists of session instances.
	
	An in-memory cache is maintained for the persistent storage using
	an explicity write-back strategy: the cache is only flushed when
	``sync()`` is called.
	"""
	def __init__(self, dbName):
		self._cache = dict()
		import shelve
		self._store = shelve.open(dbName, "c")
		import threading
		self.lock = threading.RLock()

	def __getitem__(self, key):
		if self._cache.has_key(key):
			return self._cache[key]
		# If this raises KeyError, that's what we want anyway
		value = self._store[key]
		self._cache[key] = value
		return value

	def __delitem__(self, key):
		try:
			self._cache[key]
		except KeyError:
			# Maybe we haven't loaded it yet
			pass
		try:
			del self._store[key]
		except KeyError:
			# Maybe we haven't sync'ed it yet
			pass

	def __setitem__(self, key, value):
		self._cache[key] = value

	def get(self, key, defaultValue):
		"""Return session instance for ``key``."""
		try:
			return self[key]
		except KeyError:
			return defaultValue

	def setdefault(self, key, defaultValue):
		"""Return session instance for ``key``, creating if necessary."""
		try:
			return self[key]
		except KeyError:
			self[key] = defaultValue
			return defaultValue

	def keys(self):
		"""Return list of session keys (in the dictionary sense)."""
		return self._store.keys()

	def sync(self, key=None):
		"""Synchronize memory cache with persistent storage."""
		if key:
			self._store[key] = self._cache[key]
		else:
			for key in self._cache.iterkeys():
				self._store[key] = self._cache[key]

	def close(self):
		"""Clear memory
		self._cache = dict()
		self._store.close()

	#
	# Methods for manipulating session lists
	#
	def updateSessionList(self, account, sessionList):
		"""Update sessions associated with an account.
		
		If ``sessionList`` is None, the entire account entry
		is deleted.  Cache is synchronized with persistent
		storage after update."""
		with self.lock:
			if sessionList:
				self[account] = sessionList
			else:
				try:
					del self[account]
				except KeyError:
					print "no such account: %s" % account
			self.sync(account)

	def getSessionList(self, account):
		"""Return sessions associated with an account."""
		with self.lock:
			try:
				return self[account]
			except KeyError:
				return []

	def findSession(self, account, name, password):
		"""Find session matching ``name`` and ``password`` for ``account``."""
		with self.lock:
			for s in self.getSessionList(account):
				if s.name != name:
					continue
				if s.password != password:
					raise WSGIError(Status_BadRequest,
							"Password incorrect")
				return s
			else:
				return None

def backend(sessionDir, sessionName, toChild, fromChild):
	"""Invoke a backend process."""
	import os.path, sys, copy, os
	os.dup2(toChild[0].fileno(), 0)
	os.dup2(fromChild[1].fileno(), 1)
	toChild[0].close()
	toChild[1].close()
	fromChild[0].close()
	fromChild[1].close()
	env = copy.copy(os.environ)
	# Assume a directory layout of:
	# root (CHIMERA2)
	#  +--- bin (where programs live)
	#  +--- lib (where libraries and shared objects live)
	#  +--- webapp (where WSGI script lives)
	webapp_dir = os.path.dirname(__file__)
	chimera2_dir = os.path.dirname(webapp_dir)
	lib_dir = os.path.join(chimera2_dir, "lib")
	bin_dir = os.path.join(chimera2_dir, "bin")
	env["CHIMERA2"] = chimera2_dir
	try:
		ld_path = env["LD_LIBRARY_PATH"]
	except KeyError:
		env["LD_LIBRARY_PATH"] = lib_dir
	else:
		env["LD_LIBRARY_PATH"] = ld_path + ':' + lib_dir
	program = "webapp_backend"
	execle(os.path.join(bin_dir, program), program, sessionDir, sessionName)

#
# Main program - either WSGI app or command line
#
if __name__ != "__main__":
	# Must be a WSGI session
	application = App()
else:
	# Must be a command line session
	pool = SessionStore("test.db")

	# We don't bother with locking because we're running
	# single-threaded and cannot really test whether locking
	# is effective anyway
	def list(account=""):
		if account:
			sessionList = pool.getSessionList(account)
			if not sessionList:
				print "account \"%s\" has no sessions" % account
			else:
				print "account:", account
				for s in sessionList:
					print s
		else:
			for account in pool.keys():
				sessionList = pool.getSessionList(account)
				if not sessionList:
					continue
				print "account:", account
				for s in sessionList:
					print s

	def create(account, session, password=""):
		pool.lock.acquire()
		s = pool.findSession(account, session, password)
		if s:
			import sys
			print >> sys.stderr, "\"%s\" exists" % session
		else:
			sessionList = pool.getSessionList(account)
			sessionList.append(Session(session, password))
			pool.updateSessionList(account, sessionList)
			print "\"%s\" created" % session
		pool.lock.release()

	def delete(account, session, password=""):
		pool.lock.acquire()
		s = pool.findSession(account, session, password)
		if not s:
			import sys
			print >> sys.stderr, "\"%s\" does not exist" % session
		else:
			sessionList = pool.getSessionList(account)
			sessionList.remove(s)
			pool.updateSessionList(account, sessionList)
			print "\"%s\" removed" % session
		pool.lock.release()

	def call(account, session, *args):
		if len(args) > 0 and args[0].startswith("password="):
			password = args[0][9:]
			args = args[1:]
		else:
			password = ""
		s = pool.findSession(account, session, password)
		if not s:
			import sys
			print >> sys.stderr, "\"%s\" does not exist" % session
		else:
			print s.call(*args)

	def main():
		g = globals()
		while True:
			try:
				input = raw_input("-> ")
			except EOFError:
				input = None
			if not input:
				break
			fields = input.split()
			command = fields[0]
			try:
				f = g[command]
			except KeyError:
				import sys
				print >> sys.stderr, "%s: bad command" % command
			else:
				try:
					f(*fields[1:])
				except:
					import traceback
					traceback.print_exc()

	main()
	pool.close()
