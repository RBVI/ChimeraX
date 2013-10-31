"""
Manage models

Exported triggers:

	OPEN_MODELS -- use as data type trigger, but created/deleted
		refers to models in the set of open models, rather than
		instances of an "OpenModels" type.
"""
from . import triggers
OPEN_MODELS = "open models"
triggers.add_trigger(OPEN_MODELS)

_open_models = {}

# TODO: malloc-ish management of model ids, so they may be reused
from itertools import count as _count
_id_counter = _count(1)

class Model:

	def __init__(self):
		self.id = next(_id_counter)
		self.name = "unknown"
		self.graphics = {}

def add(models):
	"""Add models to set of open models"""
	for m in models:
		_open_models[m.id] = m
	triggers.activate_trigger(OPEN_MODELS, { "created": models })

def remove(models):
	"""Remove models from set of open models"""
	triggers.activate_trigger(OPEN_MODELS, { "deleted": models })
	for m in models:
		del _open_models[m.id]

def list(types=None):
	"""Return a (restricted) list of open models

	:param types: a tuple of interesting Model subtypes
	"""
	if types is None:
		return list(_open_models.keys())
	if not isinstance(types, tuple):
		types = tuple(types)
	return [m for m in _open_models.values() if isinstance(m, types)]
