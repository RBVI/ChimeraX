# vim: set expandtab shiftwidth=4 softtabstop=4:
_version = (2, 0, 0, "")

def get_version_string():
	return "%d.%d.%d%s" % _version

def get_version():
	return _version

def get_major_version():
	return _version[0]

def get_minor_version():
	return _version[1]

def get_micro_version():
	return _version[2]

def get_nano_version():
	return _version[3]
