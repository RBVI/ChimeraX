"""
Manage models
=============

Two

	OPEN_MODELS -- use as data type trigger, but created/deleted
		refers to models in the set of open models, rather than
		instances of an "OpenModels" type.
"""
from .trackchanges import track

_open_models = {}	# model id: model instance

# TODO: malloc-ish management of model ids, so they may be reused
from itertools import count as _count
_id_counter = _count(1)

@track.register_data_type
class Model:

	def __init__(self):
		self.id = next(_id_counter)
		self.name = "unknown"
		self.graphics = None
		track.created(Model, [self])

	def save(self):
		raise NotImplemented

	def restore(self):
		raise NotImplemented

	def export(self):
		raise NotImplemented

	def make_graphics(self):
		raise NotImplemented

	def close(self):
		if self.graphics:
			self.graphics.clear()
			self.graphics = None
		if self.id in _open_models:
			remove([self])
		track.deleted(Model, [self])

@track.register_data_type(after=[Model])
class OpenModels:
	pass

def add(models):
	"""Add models to set of open models"""
	for m in models:
		_open_models[m.id] = m
	track.created(OpenModels, models)

def remove(models):
	"""Remove models from set of open models"""
	for m in models:
		del _open_models[m.id]
	track.deleted(OpenModels, models)

_builtin_list = list
def list(types=None):
	"""Return a (restricted) list of open models

	:param types: a tuple of interesting Model subtypes
	"""
	if types is None:
		return _builtin_list(_open_models.values())
	if not isinstance(types, tuple):
		types = tuple(types)
	return [m for m in _open_models.values() if isinstance(m, types)]

def close(model_id):
	if model_id not in _open_models:
		raise ValueError("Unknown model '%s'" % model_id)
	m = _open_models[model_id]
	m.close()
