"""
formats -- default chimera2 file format support
"""

import os
from importlib import import_module

def initialize():
	"""scan formats directory for files that register file formats"""
	for filename in os.listdir(os.path.dirname(__file__)):
		if filename.startswith('_') or not filename.endswith('.py'):
			continue
		module_name = filename[:-3]
		module = import_module('.%s' % module_name, __package__)
		if hasattr(module, 'register'):
			module.register()
