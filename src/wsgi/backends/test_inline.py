# vim: set expandtab shiftwidth=4 softtabstop=4:
Status_Okay = "200 OK"

ContentType_HTML = "text/html"
ContentType_JSON = "application/json"

from base import register

@register(__name__.split('.')[-1])
def backend(sessionDir, sessionName, toChild, fromChild):
	toChild[1].close()
	fromChild[0].close()
	Server(sessionDir, sessionName, toChild[0], fromChild[1]).run()

class Server(object):
	SavedAttributes = [
		"opened"
	]
	import os.path
	DefaultDir = os.path.dirname(__file__)

	def __init__(self, sessionDir, sessionName, inconn, outconn):
		#self.log = open("/var/tmp/conrad.out", "w")
		self.log = None
		self.sessionDir = sessionDir
		self.sessionName = sessionName
		self.inconn = inconn
		self.outconn = outconn
		self.seq = 0
		if self.log:
			print >> self.log, "inconn", inconn
			print >> self.log, "outconn", outconn
		self.restore()

	def _getDb(self, mode="c"):
		import os
		dbName = os.path.join(self.sessionDir, self.sessionName)
		import shelve
		return shelve.open(dbName, flag=mode)

	def save(self):
		db = self._getDb()
		for a in self.SavedAttributes:
			db[a] = getattr(self, a)
		db.close()

	def restore(self):
		import anydbm
		try:
			db = self._getDb(mode="r")
		except anydbm.error:
			# No previous state, initialize
			self.opened = None
		else:
			for k, v in db.iteritems():
				setattr(self, k, v)
			db.close()

	def run(self):
		while True:
			try:
				req = self.inconn.recv()
			except EOFError:
				req = None
			if not req:
				break
			if self.log:
				print >> self.log, "run", req
				print >> self.log, "inconn", self.inconn
				print >> self.log, "outconn", self.outconn
			try:
				v = self._processRequest(*req)
			except:
				if self.log:
					import traceback
					traceback.print_exc(file=self.log)
					self.log.flush()
			if self.log:
				print >> self.log, "v", v
			self.outconn.send(v)
		self.save()

	def _processRequest(self, req):
		if self.log:
			print >> self.log, "processRequest", self.seq
		self.seq += 1
		import os.path
		path = os.path.join(self.DefaultDir, req + ".json")
		if os.path.exists(path):
			if self.log:
				print >> self.log, "found", path
			with file(path) as f:
				output = f.read()
			ctype = ContentType_JSON
		else:
			if self.log:
				print >> self.log, "not found", path
			output = "%s is not available" % req
			ctype = ContentType_HTML
		import cPickle as pickle
		return pickle.dumps((Status_Okay, ctype, None, output))
