"""
cli: application command line support
=====================================

This module provides a method for parsing text commands
and calling the functions that implement them.
First, commands are registered
with a description of the arguments they take,
and with a function that implements the command.
Later, a :py:class:`Command` instance is used to parse
a command line, and optionally execute the command.
Incomplete command lines are supported,
so possible completions can be suggested.

Text Commands
-------------

Synopsis::

	command_name r1 r2 [o1 [o2]] [k1 v1] [k2 v2]

Text commands are composed of a command name, which can be multiple words,
followed by required positional arguments, *rX*,
optional positional arguments, *oX*,
and keyword arguments with a value, *kX vX*.
Each argument has an associated Python argument name
(for keyword arguments it is the keyword),
so *rX*, *oX*, and *vX* are the type-checked values.
The names of the optional arguments are used to
let them be given as keyword arguments as well.
Multiple value arguments are separated by commas
and the commas may be followed by whitespace.
Depending on the type of an argument, *e.g.*, a color name,
whitespace can also appear within an argument value.
Argument values may be quoted with double quotes.
And in quoted strings, Python's string escape sequences are recognized,
*e.g.*, ``\\N{LATIN CAPITAL LETTER A WITH RING ABOVE}`` for the ångström sign,
\N{LATIN CAPITAL LETTER A WITH RING ABOVE}.

Words in the command text may be abbreviated
and are automatically completed to the first registered
command with the given prefix.  Likewise for keyword arguments.

Registering Commands
--------------------

To add a command, :py:func:`register` the command name,
a description of the arguments it takes,
and the function to call that implements the command.
Command registration can be partially delayed to avoid importing
the command description and function until needed.
See :py:func:`register` and :py:func:`defer_registration` for details.

The description is either an instance of the Command Information class,
:py:class:`CmdInfo`, or a tuple with the arguments to the initializer.
The CmdInfo initializer takes tuples describing the required, optional,
and keyword arguments.
Each tuple, contains tuples with the argument name and a type annotation
(see below).
Postconditions (see below) can be given too.

Type Annotations
----------------

There are many standard type notations and they should be reused
as much as possible:

+-------------------------------+---------------------------------------+
|  Type				|  Annotation				|
+===============================+=======================================+
+ :py:class:`bool`		| ``bool_arg``				|
+-------------------------------+---------------------------------------+
+ :py:class:`float`		| ``float_arg``				|
+-------------------------------+---------------------------------------+
+ :py:class:`int`		| ``int_arg``				|
+-------------------------------+---------------------------------------+
+ :py:class:`str`		| ``string_arg``			|
+-------------------------------+---------------------------------------+
+ tuple of 3 :py:class:`bool`	| ``bool3_arg``				|
+-------------------------------+---------------------------------------+
+ tuple of 3 :py:class:`float`	| ``float3_arg``			|
+-------------------------------+---------------------------------------+
+ tuple of 3 :py:class:`int`	| ``int3_arg``				|
+-------------------------------+---------------------------------------+
+ list of :py:class:`float`	| ``floats_arg``			|
+-------------------------------+---------------------------------------+
+ list of :py:class:`int`	| ``ints_arg``				|
+-------------------------------+---------------------------------------+

.. molecule_arg(s):
.. molecules_arg(s, min = 0):
.. atoms_arg(s):
.. model_arg(s):
.. models_arg(s):
.. model_id_arg(s):
.. specifier_arg(s):
.. openstate_arg(s):
.. volume_arg(v):
.. volumes_arg(v):
.. surfaces_arg(s):
.. surface_pieces_arg(spec):
.. multiscale_surface_pieces_arg(spec):
.. points_arg(a):

There is one special annotation: :py:obj:`rest_of_line` that consumes
the rest of the command line as a string.

Annotations are used to parse text and to support automatic completion.
Annotations can be extended with various specializers:

+-----------------------+-----------------------------------------------+
|  Specializer		|  Example					|
+=======================+===============================================+
+ :py:class:`Bounded`	| ``Bounded(float_arg, 0.0, 100.0)``		|
+-----------------------+-----------------------------------------------+
+ :py:class:`List_of`	| ``List_of(float_arg)``			|
+			| *a.k.a.*, ``floats_arg``			|
+-----------------------+-----------------------------------------------+
+ :py:class:`Set_of`	| ``Set_of(int_arg)``				|
+-----------------------+-----------------------------------------------+
+ :py:class:`Tuple_of`	| ``Tuple_of(float_arg, 3)``			|
+			| *a.k.a.*, ``float3_arg``			|
+-----------------------+-----------------------------------------------+
+ :py:class:`Or`	| ``Or(float_arg, string_arg)``	*discouraged*	|
+-----------------------+-----------------------------------------------+
+ :py:class:`Enum_of`	| enumerated values				|
+-----------------------+-----------------------------------------------+

Creating Your Own Type Annotation
---------------------------------

Annotations perform several functions:
(1) to convert text to a value of the appropriate type,
(2) to give reasonable error messages,
and (3) to provide possible completions for incomplete text.

See the :py:class:`Annotation` documentation for details.

Example
-------

Here is a simple example::

	import cli
	@register("echo", cli.CmdInfo(optional=(('text', cli.rest_of_line))))
	def echo(text=''):
		print(text)
	...
	command = cli.Command()
	command.parse_text(text, final=True)
	try:
		status = command.execute()
		if status:
			print(status)
	except cli.UserError as err:
		print(err, file=sys.stderr)

.. todo::

    Build data structure with introspected information and allow it to
    be supplemented separately for command functions with \*\*kw arguments.
    That way a command that is expanded at runtime could pick up new arguments
    (*e.g.*, the open command).

.. todo::

    Issues: autocompletion, minimum 2 letters? extensions?
    help URL? affect graphics flag?

"""

class UserError(ValueError):
	"""An exception provoked by the user's input.

	UserError(object) -> a UserError object

	This is in contrast to a error is a bug in the program.
	"""
	pass

import sys
from collections import OrderedDict

class Annotation:
	# TODO: Annotation is an ABC
	"""Base class for all annotations

	Each annotation should have the following attributes:

	.. py:attribute:: name

		Set to textual description of the annotation, including
		the leading article, *e.g.*, `"a truth value"`.

	.. py:attribute:: multiword

		Set to true if textual representation can have a space in it.
	"""
	multiword = False
	name = "** article name, e.g., _a_ _truth value_ **"

	@staticmethod
	def parse(text):
		"""Return text converted to appropriate type.
		
		:param text: command line text to parse 

		Abbreviations should be not accepted, instead they
		should be discovered via the possible completions.
		"""
		raise NotImplemented

	@staticmethod
	def completions(text):
		"""Return list of possible completions of the given text.
			
		:param text: Text to check for possible completions

		Note that if invalid completions are given, then parsing
		can go into an infinite loop when trying to automatically
		complete text.
		"""
		raise NotImplemented

class Aggregate(Annotation):
	"""Common class for collections of values.

	Aggregate(annotation, constructor, add_to, min_size=None, max_size=None, name=None) -> annotation
	
	:param annotation: annotation for values in the collection.
	:param constructor: function/type to create an empty collection.
	:param add_to: function to add an element to the collection,
	    typically an unbound method.  For immutable collections,
	    return a new collection.
	:param min_size: minimum size of collection, default `None`.
	:param max_size: maximum size of collection, default `None`.
	"""
	min_size = 0
	max_size = sys.maxsize

	def __init__(self, annotation, constructor, add_to, min_size=None, max_size=None, name=None):
		self.annotation = annotation
		self.constructor = constructor
		self.add_to = add_to
		if min_size is not None:
			self.min_size = min_size
		if max_size is not None:
			self.max_size = max_size
		if name is None:
			if ',' in annotation.name:
				name = "collection of %s" % annotation.name
			else:
				# discard article
				name = annotation.name.split(None, 1)[1]
				name = "%s(s)" % name
		self.name = name

	def parse(self, text):
		return self.annotation.parse(text)

	def completions(self, text):
		return self.annotation.completions(text)

def List_of(annotation, min_size=None, max_size=None):
	"""Annotation for lists of a single type
	
	List_of(annotation, min_size=None, max_size=None) -> annotation
	"""
	return Aggregate(annotation, list, list.append, min_size, max_size)

def Set_of(annotation, min_size=None, max_size=None):
	"""Annotation for sets of a single type
	
	Set_of(annotation, min_size=None, max_size=None) -> annotation
	"""
	return Aggregate(annotation, set, set.add, min_size, max_size)

def _tuple_append(t, value):
	return t + (value,)

def Tuple_of(annotation, size):
	"""Annotation for tuples of a single type
	
	Tuple_of(annotation, size) -> annotation
	"""
	return Aggregate(annotation, tuple, _tuple_append, size, size)

class bool_arg(Annotation):
	"""Annotation for boolean literals"""
	name = "a truth value"

	@staticmethod
	def parse(text):
		text = text.casefold()
		if text == "0" or text == "false":
			return False
		if text == "1" or text == "true":
			return True
		raise ValueError("Expected true or false (or 1 or 0)")

	@staticmethod
	def completions(text):
		result = []
		text = text.casefold()
		if "false".startswith(text):
			result.append("false")
		if "true".startswith(text):
			result.append("true")
		return result

class int_arg(Annotation):
	"""Annotation for integer literals"""
	name = "a whole number"

	@staticmethod
	def parse(text):
		try:
			return int(text)
		except ValueError:
			raise ValueError("Expected %s" % int_arg.name)

	@staticmethod
	def completions(text):
		int_chars = "+-0123456789"
		if not text:
			return [x for x in int_chars]
		return []

class float_arg(Annotation):
	"""Annotation for floating point literals"""
	name = "a floating point number"

	@staticmethod
	def parse(text):
		try:
			return float(text)
		except ValueError:
			raise ValueError("Expected %s" % float_arg.name)

	@staticmethod
	def completions(text):
		if not text:
			return [x for x in "+-0123456789"]

class string_arg(Annotation):
	"""Annotation for string literals"""
	name = "a text string"

	@staticmethod
	def parse(text):
		return text

	@staticmethod
	def completions(text):
		return []

class Bounded(Annotation):
	"""Support bounded numerical values

	Bounded(annotation, min=None, max=None, name=None) -> a Bounded object

	:param annotation: numerical annotation
	:param min: optional lower bound
	:param max: optional upper bound
	:param name: optional explicit name for annotation
	"""

	def __init__(self, annotation, min=None, max=None, name=None):
		self.anno = annotation
		self.min = min
		self.max = max
		if name is None:
			if min and max:
				name = "%s <= %s <= %s" % (min, annotation.name, max)
			elif min:
				name = "%s >= %s" % (annotation.name, min)
			elif max:
				name = "%s <= %s" % (annotation.name, max)
			else:
				name = annotation.name
		self.name = name

	def parse(self, text):
		value = self.anno.parse(text)
		if self.min is not None and value < self.min:
			raise ValueError("Must be greater than or equal to %s" % self.min)
		if self.max is not None and value > self.max:
			raise ValueError("Must be less than or equal to %s" % self.max)
		return value

	def completions(self, text):
		return self.anno.completions(text)

class Enum_of(Annotation):
	"""Support enumerated types

	Enum(values, ids=None, name=None) -> an Enum object

	:param values: sequence of values
	:param ids: optional sequence of identifiers
	:param name: optional explicit name for annotation

	If the *ids* are given, then there must be one for each
	and every value, otherwise the values are used as the identifiers.
	The identifiers must all be strings.
	"""
	def __init__(self, values, ids=None, name=None):
		if ids is not None:
			if len(values) != len(ids):
				raise ValueError("Must have an identifier for each and every value")
		self.values = values
		if ids is not None:
			assert(all([isinstance(x, str) for x in ids]))
			self.ids = ids
		else:
			assert(all([isinstance(x, str) for x in values]))
			self.ids = values
		self.values = values
		self.multiple = any([(' ' in i) for i in self.ids])
		if name is None:
			name = "one of '%s', or '%s'" % (
				"', '".join(self.ids[0:-1]), self.ids[-1])
		self.name = name

	def parse(self, text):
		text = text.casefold()
		for i, x in enumerate(self.ids):
			if x.casefold() == text:
				return self.values[i]
		raise ValueError("Invalid %s" % self.name)

	def completions(self, text):
		text = text.casefold()
		return [x for x in self.ids if x.casefold().startswith(text)]

class Or(Annotation):
	"""Support two or more alternative annotations
	
	Or(annotation, annotation [, annotation]*, name=None) -> annotation

	:param name: optional explicit name for annotation
	"""

	def __init__(self, *annotations, name=None):
		if len(annotations) < 2:
			raise ValueError("Need at two alternative annotations")
		self.annotations = annotations
		self.multiple = any([a.multiple for a in annotations])
		if name is None:
			name = "%s, or %s" % (
				", ".join(annotations[0:-1]), annotations[-1])
		self.name = name

	def parse(self, text):
		for anno in self.annotations:
			try:
				return anno.parse(text)
			except ValueError:
				pass
		names = [a.__name__ for a in self.annotations]
		msg = ', '.join(names[:-1])
		if len(names) > 2:
			msg += ', '
		msg += 'or ' + names[-1]
		raise ValueError("Excepted %s" % msg)

	def completions(self, text):
		"""completions are the union of alternative annotation completions"""
		completions = []
		for anno in self.annotations:
			completions += anno.completions(text)
		return completions

class rest_of_line(Annotation):
	name = "the rest of line"

	@staticmethod
	def parse(text):
		# convert \N{unicode name} to unicode, etc.
		return text.encode('utf-8').decode('unicode-escape')

	@staticmethod
	def completions(text):
		return []

bool3_arg = Tuple_of(bool_arg, 3)
ints_arg = List_of(int_arg)
int3_arg = Tuple_of(int_arg, 3)
floats_arg = List_of(float_arg)
float3_arg = Tuple_of(float_arg, 3)
positive_int_arg = Bounded(int_arg, min=1, name="natural number")
model_id_arg = positive_int_arg

class Postcondition:
	"""Base class for postconditions"""
	# TODO: Postcondition is an ABC

	def check(self, kw_args):
		"""Return true if function arguments are consistent"""
		raise NotImplemented

	def error_message(self):
		"""Appropriate error message if check fails."""
		raise NotImplemented

class SameSize(Postcondition):
	"""Postcondition check for same size arguments
	
	SameSize(name1, name2) -> a SameSize object

	:param name1: name of first argument to check
	:param name2: name of second argument to check
	"""

	__slots__ = ['name1', 'name2']

	def __init__(self, name1, name2):
		self.name1 = name1
		self.name2 = name2

	def check(self, kw_args):
		args1 = kw_args.get(self.name1, None)
		args2 = kw_args.get(self.name2, None)
		if args1 is None and args2 is None:
			return True
		try:
			return len(args1) == len(args2)
		except TypeError:
			return False

	def error_message(self):
		return "%s argument should be the same size as %s argument" % (self.name1, self.name2)

# _commands is a map of command name to command information.  Except when
# it is a multiword command name, then the preliminary words map to
# dictionaries that map to the command information.
# An OrderedDict is used so for autocompletion, the prefix of the first
# registered command with that prefix is used.
_commands = OrderedDict()

def _check_autocomplete(word, mapping, name):
	# this is a debugging aid for developers
	for key in mapping:
		if key.startswith(word) and key != word:
			raise ValueError("'%s' is a prefix of an existing command" % name)

class CmdInfo:
	"""Hold information about commands.

	:param required: required positional arguments tuple
	:param optional: optional positional arguments tuple
	:param keyword: keyword arguments tuple

	Each tuple contains tuples with the argument name and a type annotation.
	The command line parser uses the *optional* argument names to as
	keyword arguments.
	"""
	__slots__ = [
		'required', 'optional', 'keyword',
		'postconditions', 'function',
	]

	def __init__(self, required=(), optional=(), keyword=(),
			postconditions=()):
		self.required = OrderedDict(required)
		self.optional = OrderedDict(optional)
		self.keyword = dict(keyword)
		self.keyword.update(self.optional)
		self.postconditions = postconditions
		self.function = None

	def set_function(self, function):
		"""Set the function to call when the command matches.
		
		Double check that all function arguments, that do not
		have default values, are 'required'.
		"""
		if self.function:
			raise ValueError("Can not reuse CmdInfo instances")
		import inspect
		EMPTY = inspect.Parameter.empty
		signature = inspect.signature(function)
		for p in signature.parameters.values():
			if p.default != EMPTY or p.name in self.required:
				continue
			raise ValueError("Wrong function or '%s' argument must be required or have a default value" % p.name)

		self.function = function

	def copy(self):
		"""Return a copy suitable for use with another function."""
		import copy
		ci = copy.copy(self)
		ci.function = None
		return ci

class _Defer:
	# Enable function introspection to be deferred until needed
	#
	# _Defer(proxy_function, cmd_info) -> instance
	#
	# There are two uses: (1) the proxy function returns the actual
	# function that implements the command, or (2) the proxy function
	# register subcommands and returns None.  In the former case,
	# the proxy function will typically consist of an import statement,
	# followed by returning a function in the imported module.  In the
	# latter case, multiple subcommands are registered, and nothing is
	# returned.
	__slots__ = [ 'proxy', 'cmd_info' ]

	def __init__(self, proxy_function, cmd_info):
		self.proxy = proxy_function
		if isinstance(cmd_info, tuple):
			cmd_info = CmdInfo(*cmd_info)
		self.cmd_info = cmd_info

	def call(self):
		return self.proxy()

def delay_registration(name, proxy_function, cmd_info=None):
	"""delay registering a named command until needed

	If the command information is given, then the proxy function
	should return the actual function used to implement the command.
	Otherwise, the function should explicitly register commands.
	The proxy function may register subcommands.
	"""
	register(name, None, _Defer(proxy_function, cmd_info))

def register(name, cmd_info, function=None):
	"""register function that implements command
	
	:param name: the name of the command and may include spaces.
	:param cmd_info: information about the command, either an
	    instance of :py:class:`CmdInfo`, or the tuple with CmdInfo
	    parameters.
	:param function: the callback function.

	If the function is None, then it assumed that :py:func:`register`
	is being used as a decorator.

	To delay introspecting the function until it is actually used,
	register using the :py:func:`delay_registration` function.

	For autocompletion, the first command registered with a
	given prefix wins.  Registering a command that is a prefix
	of an existing command is an error since it breaks backwards
	compatibility.
	"""
	if function is None:
		# act as a decorator
		def wrapper(function, name=name, cmd_info=cmd_info):
			return register(name, cmd_info, function)
		return wrapper

	if isinstance(cmd_info, tuple):
		cmd_info = CmdInfo(*cmd_info)

	words = name.split()
	cmd_map = _commands
	for word in words[:-1]:
		what = cmd_map.get(word, None)
		if isinstance(what, dict):
			cmd_map = what
			continue
		deferred = isinstance(what, _Defer)
		if what is not None and not deferred:
			raise ValueError("Can't mix subcommands with no subcommands")
		if not deferred:
			_check_autocomplete(word, cmd_map, name)
		d = cmd_map[word] = OrderedDict()
		cmd_map = d
	word = words[-1]
	what = cmd_map.get(word, None)
	if isinstance(what, dict):
		raise ValueError("Command is part of multiword command")
	#if what is not None:
	#	pass # TODO: replacing, preserve extra keywords?
	_check_autocomplete(word, cmd_map, name)
	if isinstance(function, _Defer):
		# delay introspecting function
		cmd_info = function
	else:
		# introspect immediately to give errors
		cmd_info.set_function(function)
	cmd_map[word] = cmd_info
	return function		# needed when used as a decorator

def _lazy_introspect(cmd_map, word):
	deferred = cmd_map[word]
	function = deferred.call()
	if function is not None:
		cmd_info = deferred.cmd_info
		if cmd_info is None:
			raise RuntimeError("delayed registration forgot command information")
		cmd_info.set_function(function)
		cmd_map[word] = cmd_info
		return cmd_info
	# deferred function might have registered subcommands
	cmd_info = cmd_map[word]
	if isinstance(cmd_info, (dict, CmdInfo)):
		return cmd_info
	raise RuntimeError("delayed registration didn't register the command")

def add_keyword_arguments(name, kw_info):
	"""Make known additional keyword argument(s) for a command
	
	:param name: the name of the command
	:param kw_info: { keyword: annotation }
	"""
	words = name.split()
	cmd_map = _commands
	for word in words[:-1]:
		what = cmd_map.get(word, None)
		if isinstance(what, dict):
			cmd_map = what
			continue
		raise ValueError("'%s' is not a command" % name)
	word = words[-1]
	ci = cmd_map.get(word, None)
	if ci is None:
		raise ValueError("'%s' is not a command" % name)
	if isinstance(ci, _Defer):
		ci = _lazy_introspect(cmd_map, word)
	if isinstance(ci, dict):
		raise ValueError("'%s' is not the full command" % name)
	# TODO: fail if there are conflicts with existing keywords?
	ci.keyword.update(kw_info)
	# TODO: save appropriate kw_info, if reregistered?

import re
normal = re.compile(r"\S*")
normal_wo_comma = re.compile(r"[^,\s]*")
single = re.compile(r"'([^']|\')*'")
double = re.compile(r'"([^"]|\")*"')
whitespace = re.compile("\s+")

class Command:
	"""Keep track of partially typed command with possible completions
	
	:param text: the command text
	:param final: true if text is the complete command line (final version).
	"""
	def __init__(self, text='', final=False):
		self._reset()
		if text:
			self.parse_text(text, final)

	def _reset(self):
		self.current_text = ""
		self.amount_parsed = 0
		self.completion_prefix = ""
		self.completions = []
		self._error = "Missing command"
		self._ci = None
		self.command_name = None
		self._kwargs = {}
		self._error_checked = False

	def error_check(self):
		"""Error check results of calling parse_text

		Separate error checking logic from execute() so
		it may be done separately
		"""
		if self._error:
			raise UserError(self._error)
		for cond in self._ci.postconditions:
			if not cond():
				raise UserError(cond.message())
		self._error_checked = True

	def execute(self):
		"""If command is valid, execute it."""
		
		if not self._error_checked:
			self.error_check()
		try:
			return self._ci.function(**self._kwargs)
		except ValueError as err:
			# convert function's ValueErrors to UserErrors,
			# but not those of functions it calls
			import sys, traceback
			_, _, exc_traceback = sys.exc_info()
			if len(traceback.extract_tb(exc_traceback)) > 2:
				raise
			raise UserError(err)

	def _next_token(self, text, commas=False):
		# Return tuple of first argument in text and actual text used
		#
		# Arguments may be quoted, in which case the text between
		# the quotes is returned.  If there is no closing quote,
		# return rest of line for automatic completion purposes,
		# but set an error.
		m = whitespace.match(text)
		start = m.end() if m else 0
		if start == len(text):
			return '', text
		if text[start] == '"':
			m = double.match(text, start)
			if m:
				end = m.end()
				token = text[start + 1:end - 1]
			else:
				end = len(text)
				token = text[start + 1:end]
				self._error = "incomplete quoted text"
		elif text[start] == "'":
			m = single.match(text, start)
			if m:
				end = m.end()
				token = text[start + 1:end - 1]
			else:
				end = len(text)
				token = text[start + 1:end]
				self._error = "incomplete quoted text"
			# convert \N{unicode name} to unicode, etc.
			token = token.encode('utf-8').decode('unicode-escape')
		elif commas:
			if text[start] == ',':
				return ',', text[0:start + 1]
			m = normal_wo_comma.match(text, start)
			end = m.end()
			token = text[start:end]
		else:
			m = normal.match(text, start)
			end = m.end()
			token = text[start:end]
		return token, text[0:end]

	def _complete(self, chars, suffix):
		# insert completion taking into account quotes
		i = len(chars)
		c = chars[0]
		if c != '"' or chars[-1] != c:
			completion = chars + suffix
		else:
			completion = chars[:-1] + suffix + chars[-1]
		j = self.amount_parsed
		t = self.current_text 
		self.current_text = t[0:j] + completion + t[i + j:]
		return self.current_text[j:]

	def _parse_arg(self, annotation, text, final, commas=False):
		if annotation is rest_of_line:
			return text, ""

		multiword = annotation.multiword
		all_words = []
		all_chars = []
		count = 0
		while 1:
			count += 1
			if count > 100:
				raise RuntimeError("Invalid completions given by %s" % annotation)
			word, chars = self._next_token(text, commas)
			if not word:
				raise ValueError("Expected %s" % annotation.name)
			all_words.append(word)
			word = ' '.join(all_words)
			all_chars.append(chars)
			chars = ''.join(all_chars)
			try:
				value = annotation.parse(word)
				break
			except ValueError as err:
				completions = annotation.completions(word)
				if (final or len(text) > len(chars)) \
				and completions:
					c = completions[0][len(word):]
					if multiword:
						c = c.split(None, 1)[0]
					text = self._complete(chars, c)
					del all_words[-1]
					del all_chars[-1]
					continue
				self._error = err
				if multiword and not completions:
					# try shorter version
					del all_words[-1]
					word = ' '.join(all_words)
					completions = annotation.completions(word)
				self.completion_prefix = word
				self.completions = completions
				raise
		self.amount_parsed += len(chars)
		text = text[len(chars):]
		return value, text

	def _parse_aggregate(self, anno, text, final):
		# expect VALUE [, VALUE]*
		self._error = ""
		self.completion_prefix = ""
		self.completions = []
		values = anno.constructor()
		value, text = self._parse_arg(anno, text, final, True)
		x = anno.add_to(values, value)
		if x is not None:
			values = x
		while 1:
			word, chars = self._next_token(text, True)
			if word != ',':
				if len(values) < anno.min_size:
					if anno.min_size == anno.max_size:
						qual = "exactly"
					else:
						qual = "at least"
					raise ValueError("Need %s %d %s" % (qual, anno.min_size, anno.name))
				if len(values) > anno.max_size:
					if anno.min_size == anno.max_size:
						qual = "exactly"
					else:
						qual = "at most"
					raise ValueError("Need %s %d %s" % (qual, anno.max_size), anno.name)
				return values, text
			self.amount_parsed += len(chars)
			text = text[len(chars):]
			value, text = self._parse_arg(anno, text, final, True)
			x = anno.add_to(values, value)
			if x is not None:
				values = x

	def parse_text(self, text, final=False):
		"""Parse text into function and arguments
		
		:param text: The text to be parsed.
		:param final: True if last version of command text

		May be called multiple times.  There are a couple side effects:

		* The automatically completed text is put in self.current_text.
		* Possible completions are in self.completions.
		* The prefix of the completions is in self.completion_prefix.
		"""
		self._reset()	 # don't be smart, just start over

		# TODO: alias expansion

		# find command name
		self.current_text = text
		text = text[self.amount_parsed:]
		word_map = _commands
		while 1:
			word, chars = self._next_token(text)
			what = word_map.get(word, None)
			if what is None:
				self.completion_prefix = word
				self.completions = [
					x for x in word_map if x.startswith(word)]
				if word and (final or len(text) > len(chars)) \
				and self.completions:
					# If final version of text, or if there
					# is following text, make best guess,
					# and retry
					c = self.completions[0]
					text = self._complete(chars, c[len(word):])
					continue
				if word:
					self._error = "Unknown command"
				return
			self.amount_parsed += len(chars)
			text = text[len(chars):]
			if isinstance(what, _Defer):
				what = _lazy_introspect(word_map, word)
			if isinstance(what, dict):
				# word is part of multiword command name
				word_map = what
				self._error = "Incomplete command: %s" % self.current_text[0:self.amount_parsed]
				continue
			assert(isinstance(what, CmdInfo))
			self._ci = what
			self.command_name =  self.current_text[:self.amount_parsed]
			break

		# process positional arguments
		positional = self._ci.required.copy()
		positional.update(self._ci.optional)
		self.completion_prefix = ''
		self.completions = []
		for name, anno in positional.items():
			if name in self._ci.optional:
				self._error = ""
			else:
				self._error = "Missing required argument %s" % name
			try:
				if isinstance(anno, Aggregate):
					value, text = self._parse_aggregate(anno, text, final)
				else:
					value, text = self._parse_arg(anno, text, final)
				self._kwargs[name] = value
			except ValueError as err:
				if name in self._ci.required:
					self._error = "Invalid '%s' argument: %s" % (name, err)
					return
				# optional and wrong type, try as keyword
				break
		self._error = ""

		# process keyword arguments
		while 1:
			word, chars = self._next_token(text)
			if not word:
				# count extra whitespace as parsed
				self.amount_parsed += len(chars)
				break

			if word not in self._ci.keyword:
				self.completion_prefix = word
				self.completions = [
					x for x in word_map if x.startswith(word)]
				if (final or len(text) > len(chars)) \
				and self.completions:
					# If final version of text, or if there
					# is following text, make best guess,
					# and retry
					c = self.completions[0]
					text = self._complete(chars, c[len(word):])
					continue
				self._error = "Expected keyword, got '%s'" % word
				return
			self.amount_parsed += len(chars)
			text = text[len(chars):]

			name = word
			anno = self._ci.keyword[name]
			try:
				if isinstance(anno, Aggregate):
					value, text = self._parse_aggregate(anno, text, final)
				else:
					value, text = self._parse_arg(anno, text, final)
				self._kwargs[name] = value
			except ValueError as err:
				self._error = "Invalid '%s' argument: %s" % (name, err)
				return

if __name__ == '__main__':

	test1_info = CmdInfo(
		required=[('a', int_arg), ('b', float_arg)], 
		keyword=[('color', string_arg)]
	)
	@register('test1', test1_info)
	def test1(a: int, b: float, color=None):
		print('test1 a: %s %s' % (type(a), a))
		print('test1 b: %s %s' % (type(b), b))
		print('test1 color: %s %s' % (type(color), color))

	test2_info = CmdInfo(
		#required=[('a', string_arg)],
		#optional=[('text', rest_of_line)],
		keyword=[('color', string_arg), ('radius', float_arg)]
	)
	@register('test2', test2_info)
	def test2(a: str='', text='', color=None, radius: float=0):
		#print('test2 a: %s %s' % (type(a), a))
		#print('test2 text: %s %s' % (type(text), text))
		print('test2 color: %s %s' % (type(color), color))
		print('test2 radius: %s %s' % (type(radius), radius))

	register('mw test1', test1_info.copy(), test1)
	register('mw test2', test2_info.copy(), test2)

	test3_info = CmdInfo(
		required=[('name', string_arg)],
		optional=[('value', float_arg)]
	)
	@register('test3', test3_info)
	def test3(name: str, value=None):
		print('test3 name: %s %s' % (type(name), name))
		print('test3 value: %s %s' % (type(value), value))

	test4_info = CmdInfo(
		optional=[('draw', float_arg)]
	)
	@register('test4', test4_info)
	def test4(draw: bool=None):
		print('test4 draw: %s %s' % (type(draw), draw))

	test5_info = CmdInfo(
		optional=[('ints', floats_arg)]
	)
	@register('test5', test5_info)
	def test5(ints=None):
		print('test5 ints: %s %s' % (type(ints), ints))

	test6_info = CmdInfo(
		required=[('center', float3_arg)]
	)
	@register('test6', test6_info)
	def test6(center):
		print('test6 center:', center)

	test7_info = CmdInfo(
		optional=[('center', float3_arg)]
	)
	@register('test7', test7_info)
	def test7(center=None):
		print('test7 center:', center)

	test8_info = CmdInfo(
		optional=[
			('always', bool_arg),
			('target', string_arg),
			('names', List_of(string_arg)),
			]
	)
	@register('test8', test8_info)
	def test8(always=True, target="all", names=[None]):
		print('test8 always, target, names:', always, target, names)

	tests = [
		(True,	'test1 color red 12 3.5'),
		(True,	'test1 12 color red 3.5'),
		(True,	'test1 12 3.5 color red'),
		(True,	'test1 12 3.5 color'),
		(True,	'te'),
		(True,	'test2 color red radius 3.5 foo'),
		(True,	'test2 color red radius 3.5'),
		(True,	'test2 color red radius xyzzy'),
		(True,	'test2 color red radius'),
		(True,	'test2 color light gray'),
		(True,	'test2 color "light gray"'),
		(True,	'test2 c'),
		(True,	'test3 radius'),
		(True,	'test3 radius 12.3'),
		(True,	'test4'),
		(True,	'test4 draw'),
		(True,	'test5'),
		(True,	'test5 ints 5'),
		(True,	'test5 ints 5 ints 6'),
		(True,	'test5 ints 5, 6, 7, 8, 9'),
		(True,	'mw test1 color red 12 3.5'),
		(True,	'mw test1 color red 12 3.5'),
		(True,	'mw test2 color red radius 3.5 foo'),
		(False,	'mw te'),
		(True,	'mw '),
		(False,	'mw'),
		(True,	'te 12 3.5 co red'),
		(True,	'm te 12 3.5 col red'),
		(True,	'test6 3.4, 5.6, 7.8'),
		(True,	'test6 3.4 abc 7.8'),
		(True,	'test7 center 3.4, 5.6, 7.8'),
		(True,	'test7 center 3.4, 5.6'),
		(True,  'test8 always false'),
		(True,  'test8 always true target tool'),
		(True,  'test8 always true tool'),
		(True,  'test8 always tool'),
		(True,  'test8 TRUE tool xyzzy, plugh '),
	]
	cmd = Command()
	for t in tests:
		final, text = t
		try:
			print("\nTEST: '%s'" % text)
			cmd.parse_text(text, final=final)
			if cmd.current_text != text:
				print(cmd.current_text)
			#print(cmd.current_text, cmd._kwargs)
			p = cmd.completions
			if p:
				print('completions:', p)
			cmd.execute()
			print('SUCCESS')
		except UserError as err:
			print('FAIL:', err)
