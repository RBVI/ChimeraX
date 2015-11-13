DEFAULT_MESSAGE = "Use HTML page to access ChimeraX sessions"

#
# WSGI application code
#
STATUS_OKAY = "200 OK"
STATUS_REDIRECT = "301 Redirect to HTML page"
STATUS_BAD_REQUEST = "400 Bad request"
STATUS_SERVER_ERROR = "500 ChimeraX internal server error"

CONTENT_TYPE_PLAINTEXT = "text/plain"
CONTENT_TYPE_HTML = "text/html"
CONTENT_TYPE_JSON = "application/json"

def _debug_print(s, show_traceback=False):
	from time import ctime
	with file("/tmp/chimerax.log", "a") as f:
		print >> f, "wsgi", ctime(), s
		if show_traceback:
			import traceback
			traceback.print_exc(file=f)

class WSGIError(Exception):
	"""WSGI exception class.

	Attribute ``wsgi_data`` contains a 4-tuple of::

	HTTP status - *string*
	HTTP content type - *string*
	HTTP headers - *list* of key-value pairs
	page data - *string*
	"""
	def __init__(self, status, output, content_type=None, headers=None):
		status = "%s: %s" % (status, output)
		Exception.__init__(self, status)
		if content_type is None:
			content_type = CONTENT_TYPE_PLAINTEXT
		self.wsgi_data = (status, content_type, headers, output)

class App(object):
	"""WSGI application class.

	Maintains a session store and redirects web requests based on 
	the value of ``action`` parameters.
	"""
	def __init__(self):
		import os.path
		db_name = os.path.join(os.path.dirname(__file__),
						"sessions", "sessions.db")
		self.sessions = SessionStore(db_name)
		self.req_count = 0

	def __del__(self):
		try:
			self.sessions.close()
		except:
			pass

	def __call__(self, environ, start_response):
		self.req_count += 1
		import cgi
		fs = cgi.FieldStorage(fp=environ['wsgi.input'],
					environ=environ,
					keep_blank_values=True)
		try:
			status, ctype, hdrs, output = self.process(environ, fs)
			#status, ctype, hdrs, output = self.show_input(environ, fs)
		except WSGIError, e:
			status, ctype, hdrs, output = e.wsgi_data
		except:
			status = STATUS_SERVER_ERROR
			ctype = CONTENT_TYPE_PLAINTEXT
			hdrs = None
			import traceback
			output = [ "<pre>\n", traceback.format_exc(), "</pre>" ]
			_debug_print("trackback: %s" % output)
		if isinstance(output, basestring):
			length = len(output)
			output = [ output ]
		else:
			length = sum([ len(o) for o in output ])
		headers = [ ("Content-Type", ctype),
				("Content-Length", str(length)) ]
		if hdrs:
			headers += hdrs
		_debug_print("response status %s" % status)
		_debug_print("response headers %s" % headers)
		start_response(status, headers)
		return output

	def process(self, environ, fs):
		"""Process a single web request."""
		# environ - dictionary such as os.environ
		# fs - instance of cgi.FieldStorage
		try:
			action = self._get_field(fs, "action")
		except WSGIError:
			return self._default_action(environ)
		try:
			f = getattr(self, "_process_%s" % action)
		except AttributeError:
			raise WSGIError(STATUS_BAD_REQUEST,
					"unknown action: %s" % action)
		else:
			return f(environ, fs)

	def _getenv(self, environ, key, empty_okay=False):
		if key not in environ:
			if empty_okay:
				return ""
			raise WSGIError(STATUS_BAD_REQUEST,
				"Environment variable \"%s\" missing" % key)
		v = environ[key]
		if not empty_okay and v == "":
			raise WSGIError(STATUS_BAD_REQUEST,
				"Environment variable \"%s\" undefined" % key)
		return v

	def _get_field(self, fs, key, empty_okay=False):
		if key not in fs:
			if empty_okay:
				return ""
			raise WSGIError(STATUS_BAD_REQUEST,
				"Form field \"%s\" missing" % key)
		v = fs[key].value
		if not empty_okay and v == "":
			raise WSGIError(STATUS_BAD_REQUEST,
				"Form field \"%s\" undefined" % key)
		return v

	def _get_user(self, fs, environ):
		try:
			user = self._get_field(fs, "user")
		except WSGIError:
			user = self._getenv(environ, "REMOTE_USER")
		return user

	def _default_action(self, environ):
		uri = self._getenv(environ, "SCRIPT_URI")
		if uri.endswith('/') or uri.endswith(".wsgi"):
			base = uri.rsplit('/', 1)[0]
		else:
			base = uri
		frontend_url = base + "/www/index.html"
		return (STATUS_REDIRECT,
				CONTENT_TYPE_PLAINTEXT,
				[ ( "Location", frontend_url ) ],
				DEFAULT_MESSAGE)
	
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
		session_list = self.sessions.get_session_list(account)
		if not session_list:
			print >> sf, "<p>There are no sessions for <i>%s</i>" % account
		else:
			print >> sf, "<h1>Sessions for <i>%s</i></h1>" % account
			print >> sf, "<table>"
			print >> sf, "<tr>"
			print >> sf, "<th>Name</th><th>Last Access</th>"
			print >> sf, "</tr>"
			import time
			for s in session_list:
				print >> sf, "<tr>"
				print >> sf, "<td>%s</th>" % s.name
				print >> sf, "<td>%s</th>" % s.access_time()
				print >> sf, "</tr>"
			print >> sf, "</table>"
		output = sf.getvalue()
		return STATUS_OKAY, CONTENT_TYPE_HTML, None, output

	def _process_jlist(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		session_list = self.sessions.get_session_list(account)
		sessions = [
			account,
			[ { "name": s.name, "access": s.access_time() }
						for s in session_list ]
		]
		import json
		output = json.dumps(sessions)
		return STATUS_OKAY, CONTENT_TYPE_JSON, None, output

	def _process_create(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		name = self._get_field(fs, "session")
		password = self._get_field(fs, "password", empty_okay=True)
		with self.sessions.lock:
			session_list = self.sessions.get_session_list(account)
			for s in session_list:
				if s.name == name:
					raise WSGIError(STATUS_BAD_REQUEST,
						"Session \"%s\" already exists"
						% name)
			else:
				s = Session(name, password)
				session_list.insert(0, s)
				self.sessions.update_session_list(account,
								session_list)
				output = "<p>Session <i>%s</i> created" % name
		return STATUS_OKAY, CONTENT_TYPE_HTML, None, output

	def _process_delete(self, environ, fs):
		account = self._getenv(environ, "REMOTE_USER")
		session = self._get_field(fs, "session")
		password = self._get_field(fs, "password", empty_okay=True)
		with self.sessions.lock:
			s = self.sessions.find_session(account, session,
								password)
			if not s:
				raise WSGIError(STATUS_BAD_REQUEST,
					"No session named \"%s\"" % session)
			session_list = self.sessions.get_session_list(account)
			session_list.remove(s)
			self.sessions.update_session_list(account, session_list)
		output = "<p>Session <i>%s</i> deleted" % session
		return STATUS_OKAY, CONTENT_TYPE_HTML, None, output

	def _process_call(self, environ, fs):
		account = self._get_user(fs, environ)
		session = self._get_field(fs, "session")
		password = self._get_field(fs, "password", empty_okay=True)
		s = self.sessions.find_session(account, session, password)
		if not s:
			raise WSGIError(STATUS_BAD_REQUEST,
					"No session named \"%s\"" % session)
		command = self._get_field(fs, "command")
		return s.call(command)

	def _process_pwd(self, environ, fs):
		import os
		output = os.getcwd()
		return STATUS_OKAY, CONTENT_TYPE_PLAINTEXT, None, output

	def _process_env(self, environ, fs):
		env = '\n'.join([ "<li>%s: %s</li>" % item
				for item in sorted(environ.iteritems()) ])
		import sys
		path = '\n'.join([ "<li>%s</li>" % d for d in sys.path ])
		output = ( "<h2>Environment</h2><ul>" + env + "</ul>" +
			"<h2>Path</h2><ul>" + path + "</ul>" +
			"<h2>__file__</h2><ul><li>" + __file__ + "</li></ul>" +
			"<h2>__name__</h2><ul><li>" + __name__ + "</li></ul>")
		return STATUS_OKAY, CONTENT_TYPE_HTML, None, output

	#
	# Rest of methods are for debugging
	#
	def show_input(self, environ, fs):
		import StringIO
		sf = StringIO.StringIO()
		print >> sf, "self:", self
		import os
		print >> sf, "id:", os.getuid(), os.getgid()
		import thread
		print >> sf, "thread:", thread.get_ident()
		print >> sf, "request #:", self.req_count
		self._dump_fs(sf, fs)
		print >> sf, "extra path information:", environ["PATH_INFO"]
		print >> sf, "query string:", environ["QUERY_STRING"]
		print >> sf, "user:", environ["REMOTE_USER"]
		print >> sf, "host:", environ["REMOTE_ADDR"]
		self._dump_environ(sf, environ)
		output = sf.getvalue()
		return STATUS_OKAY, CONTENT_TYPE_PLAINTEXT, None, output

	def _dump_fs(self, f, fs):
		print >> f, ""
		print >> f, "--- FieldStorage:"
		for k in fs.keys():
			print >> f, "%s: %s" % (k, fs[k].value)
		print >> f, "--- FieldStorage"
		print >> f, ""

	def _dump_environ(self, f, env):
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
		self._init_runtime()
		self._init_state(name, password)

	def _init_runtime(self):
		self.pipe = None
		self.process = None
		import threading
		self.lock = threading.RLock()

	def _init_state(self, name, password):
		import os.path
		self.name = name
		self.password = password
		self.update_access()
		self._session_dir = os.path.join(os.path.dirname(__file__),
								"sessions")

	def __getstate__(self):
		return (self.name, self.password, self.last_access)

	def __setstate__(self, values):
		self._init_runtime()
		name, password, last_access = values
		self._init_state(name, password)
		# We ignore saved access time because we just accessed it

	def __str__(self):
		return "%s <%s>" % (self.name, self.access_time())

	def __del__(self):
		self.disconnect()

	def update_access(self):
		"""Set last access time for this session to current time."""
		import time
		self.last_access = time.time()

	def access_time(self):
		"""Return last access time as a *string*."""
		import time
		return time.ctime(self.last_access)

	def call(self, *args):
		"""Pass ``args`` to session backend and return results."""
		with self.lock:
			#print "%s.call(%s)" % (self.name, str(args))
			_debug_print("call %s" % str(args))
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
				import tempfile
				to_child = Pipe()
				from_child = Pipe()
				err_child = tempfile.TemporaryFile(bufsize=0)
				self.pipe = (from_child[0], to_child[1], err_child)
				self.process = Process(target=backend,
							args=(self._session_dir,
								self.name,
								to_child,
								from_child,
								err_child))
				self.process.start()
				#print "%s - process %d started" % (self.name,
				#			self.process.pid)
				to_child[0].close()
				from_child[1].close()
			self.pipe[1].send(args)
			try:
				output = self.pipe[0].recv()
			except EOFError:
				self.pipe[2].seek(0)
				output = self.pipe[2].read()
				_debug_print("exitcode: %d" % self.process.exitcode)
				_debug_print("error output: %s" % output)
		#_debug_print("output: %s" % str(output))
		status, content_type, headers, data = output
		results = (str(status), str(content_type), headers, str(data))
		#_debug_print("results: %s" % str(results))
		return results

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
	def __init__(self, db_name):
		self._cache = dict()
		import shelve
		self._store = shelve.open(db_name, "c")
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

	def get(self, key, default_value):
		"""Return session instance for ``key``."""
		try:
			return self[key]
		except KeyError:
			return default_value

	def setdefault(self, key, default_value):
		"""Return session instance for ``key``, creating if necessary."""
		try:
			return self[key]
		except KeyError:
			self[key] = default_value
			return default_value

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
		"""Clear memory cache"""
		self._cache = dict()
		self._store.close()

	#
	# Methods for manipulating session lists
	#
	def update_session_list(self, account, session_list):
		"""Update sessions associated with an account.
		
		If ``session_list`` is None, the entire account entry
		is deleted.  Cache is synchronized with persistent
		storage after update."""
		with self.lock:
			if session_list:
				self[account] = session_list
			else:
				try:
					del self[account]
				except KeyError:
					print "no such account: %s" % account
			self.sync(account)

	def get_session_list(self, account):
		"""Return sessions associated with an account."""
		with self.lock:
			try:
				return self[account]
			except KeyError:
				return []

	def find_session(self, account, name, password):
		"""Find session matching ``name`` and ``password`` for ``account``."""
		with self.lock:
			for s in self.get_session_list(account):
				if s.name != name:
					continue
				if s.password != password:
					raise WSGIError(STATUS_BAD_REQUEST,
							"Password incorrect")
				return s
			else:
				return None

def backend(session_dir, session_name, to_child, from_child, err_child):
	"""Invoke a backend process."""
	import sys, copy, os
	if 0:
		# enable to get core dumps from backend
		os.chdir("/tmp")
		import resource
		resource.setrlimit(resource.RLIMIT_CORE, (-1, -1))
	os.dup2(to_child[0].fileno(), 0)
	os.dup2(from_child[1].fileno(), 1)
	os.dup2(err_child.fileno(), 2)
	to_child[0].close()
	to_child[1].close()
	from_child[0].close()
	from_child[1].close()
	err_child.close()
	env = copy.copy(os.environ)
	# We can only find where the WSGI script lives.
	# Since there may be different WSGI scripts for different
	# developers, we assume that the installer script will
	# set up in the same directory a "chimerax" symlink
	# to the corresponding developer install tree.
	webapp_dir = os.path.dirname(__file__)
	chimerax_dir = os.path.join(webapp_dir, "chimerax")
	lib_dir = os.path.join(chimerax_dir, "lib")
	bin_dir = os.path.join(chimerax_dir, "bin")
	env["CHIMERA2"] = chimerax_dir
	env["LANG"] = "en_US.UTF-8"
	import pwd
	try:
		# if chimera user has lib directories, use those too
		ch_dir = pwd.getpwnam('chimera').pw_dir
		lib_dir = '%s:%s/lib64:%s/lib' % (lib_dir, ch_dir, ch_dir)
	except KeyError:
		pass
	try:
		ld_path = env["LD_LIBRARY_PATH"]
	except KeyError:
		env["LD_LIBRARY_PATH"] = lib_dir
	else:
		env["LD_LIBRARY_PATH"] = ld_path + ':' + lib_dir
	program = "python3"
	binary = os.path.join(bin_dir, program)
	script = os.path.join(webapp_dir, "webapp_backend.py")
	try:
		os.execle(binary, program, script,
				session_dir, session_name, env)
	except:
		_debug_print("exec failed: python %s script %s" % (binary, script), True)
		raise SystemExit(1)
	else:
		_debug_print("exec returned: python %s script %s" % (binary, script))
		raise SystemExit(1)

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
			session_list = pool.get_session_list(account)
			if not session_list:
				print "account \"%s\" has no sessions" % account
			else:
				print "account:", account
				for s in session_list:
					print s
		else:
			for account in pool.keys():
				session_list = pool.get_session_list(account)
				if not session_list:
					continue
				print "account:", account
				for s in session_list:
					print s

	def create(account, session, password=""):
		with pool.lock:
			s = pool.find_session(account, session, password)
			if s:
				import sys
				print >> sys.stderr, "\"%s\" exists" % session
			else:
				session_list = pool.get_session_list(account)
				session_list.append(Session(session, password))
				pool.update_session_list(account, session_list)
				print "\"%s\" created" % session

	def delete(account, session, password=""):
		with pool.lock:
			s = pool.find_session(account, session, password)
			if not s:
				import sys
				print >> sys.stderr, "\"%s\" does not exist" % session
			else:
				session_list = pool.get_session_list(account)
				session_list.remove(s)
				pool.update_session_list(account, session_list)
				print "\"%s\" removed" % session

	def call(account, session, *args):
		if len(args) > 0 and args[0].startswith("password="):
			password = args[0][9:]
			args = args[1:]
		else:
			password = ""
		s = pool.find_session(account, session, password)
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
