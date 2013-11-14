"""
trackchanges: Support for bulk tracking of data changes
=======================================================

This module provides a singleton of class Track, `track`,
that is used by tools and the GUI keep track of changes
to various data structures. 

::

	def computation():
		from chimera2.trackchanges import track
		track.block()
		...			# manipulate data
		track.release()		# propagate changes

"""
from . import triggerset

class Changes:

	__slots__ = ['created', 'modified', 'deleted', 'reasons', '_dirty']

	def __init__(self):
		self.created = set()
		self.modified = set()
		self.deleted = set()
		self.reasons = set()
		self._dirty = False

	def clear(self):
		self.created.clear()
		self.modified.clear()
		self.deleted.clear()
		self.reasons.clear()
		self._dirty = False

	def update(self):
		"""instances should only appear in one of the
		created, modified, or deleted sets"""
		if not self._dirty:
			return
		common = self.created.intersection(self.deleted)
		if common:
			# if created and deleted, ignore
			self.created.difference_update(common)
			self.modified.difference_update(common)
			self.deleted.difference_update(common)
		self.modified.difference_update(self.created)
		self.modified.difference_update(self.deleted)
		self._dirty = False

	def empty(self):
		return not any((self.created, self.modified, self.deleted))

class Track:

	def __init__(self):
		self._ts = triggerset.TriggerSet()
		self._blocked = 0
		self._changes = {}

	def register_data_type(self, data_type=None, usage_cb=None, after=(), before=()):
		"""Add data_type to those that are monitored for changes
		
		:param data_type: a consistent identifer for the data type
			to to monitored, often the class object.
		:param usage_cb: callback for when number of handlers changes
			from zero to non-zero or vice-versa.
		:param after: list of data types given to
			:py:method:`add_dependency'.

		Can be used as a decorator.
		"""
		if data_type is None:
			# act as a decorator
			def wrapper(data_type, usage_cb=usage_cb, after=after, before=before, self=self):
				return self.register_data_type(data_type, usage_cb, after, before)
			return wrapper
		self._changes[data_type] = Changes()
		self._ts.add_trigger(data_type, usage_cb, after)
		for dt in before:
			self._ts.add_dependency(dt, [data_type])
		return data_type	# needed when used as a decorator

	def add_dependency(self, data_type, after):
		"""Specify firing order dependency for 'data_type'.

		Specifies that 'data_type' should be monitored after all
		data types in the 'after' list.
		"""
		self._ts.add_dependency(data_type, after)

	def add_handler(self, data_type, func):
		assert(self._changes[data_type] is not None)
		data = self._changes[data_type]
		def wrapper(trigger, trigger_data, data=data, func=func):
			return func(data)
		return self._ts.add_handler(data_type, wrapper)

	def delete_handler(self, handler):
		self._ts.delete_handler(handler)

	def block(self):
		self._blocked += 1

	def release(self):
		self._blocked -= 1
		self._ts.block()
		for data_type, data in self._changes.items():
			data.update()
			if not data.empty():
				self._ts.activate_trigger(data_type, None)
		self._ts.release()
		for data in self._changes.values():
			data.clear()

	def _activate(self, data_type):
		data = self._changes[data_type]
		data.update()
		if not data.empty():
			self._ts.activate_trigger(data_type, None)
			data.clear()

	def created(self, data_type, instances):
		if data_type not in self._changes:
			raise ValueError("unknown data type")
		self._changes[data_type].created.update(instances)
		self._changes[data_type]._dirty = True
		if self._blocked:
			return
		self._activate(data_type)

	def modified(self, data_type, instances, reason):
		if data_type not in self._changes:
			raise ValueError("unknown data type")
		self._changes[data_type].modified.update(instances)
		self._changes[data_type].reasons.add(reason)
		self._changes[data_type]._dirty = True
		if self._blocked:
			return
		self._activate(data_type)

	def deleted(self, data_type, instances):
		if data_type not in self._changes:
			raise ValueError("unknown data type")
		self._changes[data_type].deleted.update(instances)
		self._changes[data_type]._dirty = True
		if self._blocked:
			return
		self._activate(data_type)

track = Track()
