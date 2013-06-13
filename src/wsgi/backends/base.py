#
# Backend management code.  All backends are in "backends" subdirectory.
# All active backend modules are listed in the "__all__" variable in
# backends/__init__.py.  On import, backend module must register their type.
#
_backends = dict()

#def register(name, f):
#	_backends[name] = f

def register(name):
	print "register", name
	def _register(f, name=name):
		_backends[name] = f
		return f
	return _register

def list():
	return _backends.keys()

def find(name):
	return _backends[name]
