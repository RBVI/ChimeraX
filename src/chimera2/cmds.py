"""
cmds: Unified application command line support
==============================================

Text commands are composed of a command name, which can be several
separate words, followed by positional and keyword arguments.  Keyword
arguments are followed by a value, unless the keyword represents a boolean
value, in which case the presence of the keyword indicates that it is
true.  Any part of the command line may be quoted with either single or
double quotes.  And keyword and positional arguments may be interspersed.

Words in the command text are autocompleted to the first registered
command with the given prefix.  Likewise for keyword arguments.

Registering Commands
--------------------

To add a command, :py:func:`register` it.  Later, when processing
the command line, create a :py:class:`Command` object with the
text, optionally update the text as more of the command line is
available to find out possible command and keyword completions, then
:py:func:`~Command.execute` it.

The registered command functions are introspected to figure out the
positional and keyword arguments, and argument annotations are used
to convert arguments to the correct type.  Unlike Python, where any
function argument can be given as a keyword argument, only function
arguments with default values are considered keyword arguments, and
the argument name is used as the keyword in text commands.  There are
three exceptions: (1) :py:class:`bool`ean arguments with a default value,
where the presence of the argument name sets it to :py:const:`True`, (2)
optional positional arguments (see below), and (3) aggregate arguments
(see below).  Arguments that start with an underscore are considered
private and are not exposed.

Function introspection can be deferred so that command names can
registered before the code that implements the commands is imported.
See the :py:func:`register` function for details.

Function Argument Annotations
-----------------------------

Argument notations should be a Python :py:func:`callable`, often a
Python type, that converts text to the right type.  There are three
special annotations, :py:class:`Optional`, :py:class:`Aggregate`,
and :py:class:`Or`.  An argument with an Optional annotation is treated
as a positional argument and must have a default value.  An Aggregate
annotation turns an argument a keyword argument that can be repeated
so the values are aggregated together.  Prebuilt aggregates include
:py:func:`List_of`, :py:func:`Set_of`, and :py:func:`Tuple_of`.  The
Or annotation allows for any one of several alternative annotations.

If an annotation function has a :py:class:`bool` argument without a
default value, then it needs to be annotated with the :py:class:`Bool`
annotation since any text given to the bool constructor, besides the
empty string, evaluates to True.

Annotations are also used to support autocompletion.  If the annotation
has a 'completions' method, then it is called to return a :py:class:`list`
of possible completions.  When autocompleting takes place, the first
value is used.

Example
-------

Here is a simple example::

	import cmds
	def echo(*args):
		print(*args)
	cmds.register("echo", echo)
	....

	command = cmds.Command()
	command.parse_text(text)
	try:
		print(command.execute())
	except cmds.UserError as e:
		print(e, file=sys.stderr)

.. todo::

    Build data structure with introspected information and allow it to
    be supplemented separately for command functions with \*\*kw arguments.
    That way a command that is expanded at runtime could pick up new arguments
    (e.g., the open command).

.. todo::

    Issues: autocompletion, minimum 2 letters? extensions?  Delaying
    introspect for extensions, help URL? affect graphics flag?

.. todo::

    Maybe let typed argument conversion determine how much text to consume
    (so arguments could have spaces in them)
"""

__all__ = [
	'UserError',
	'Aggregate', 'List_of', 'Set_of', 'Tuple_of',
	'Array_of',
	'Optional',
	'Or',
	'Bool',
	'register',
	#'add_keyword_arguments',
	'Defer',
	'Command',
]

class UserError(ValueError):
	"""An exception provoked by the user's input.

	As opposed to one that is a bug in the program.
	"""
	pass

from collections import OrderedDict

class Optional:
	"""Hook for optional positional arguments
	
	Optional(annotation) -> annotation
	"""

	def __init__(self, annotation):
		self.anno = annotation

	def __call__(self, text):
		return self.anno(text)

	def completions(self, text):
		if hasattr(self.anno, "completions"):
			return self.anno.completions(text)
		return []

class Aggregate:
	"""Hook for aggregating arguments

	Aggregate(annotation, constructor, add_to) -> annotation
	
	:param annotation: annotation for values in the collection
	:param constructor: function/type to create an empty collection
	:param add_to: function to add an element to the collection,
	typically an unbound method
	"""

	def __init__(self, annotation, constructor, add_to):
		self.anno = annotation
		self.constructor = constructor
		self.add_to = add_to

	def __call__(self, text):
		return self.anno(text)

	def completions(self, text):
		if hasattr(self.anno, "completions"):
			return self.anno.completions(text)
		return []

def List_of(annotation):
	"""Annotation for lists of a single type
	
	List_of(annotation) -> annotation
	"""
	return Aggregate(annotation, list, list.append)

def Set_of(annotation):
	"""Annotation for sets of a single type
	
	Set_of(annotation) -> annotation
	"""
	return Aggregate(annotation, set, set.add)

def Tuple_of(annotation):
	"""Annotation for sets of a single type
	
	Tuple_of(annotation) -> annotation
	"""
	import operator
	return Aggregate(annotation, tuple, operator.add)

class Array_of:
	"""Hook for n-element annotations"""

	def __init__(self, annotation, min, max=None):
		self.anno = annotation
		self.min_len = min
		if max is None:
			max = min
		self.max_len = max

	def __call__(self, text):
		return self.anno(text)

	def completions(self, text):
		if hasattr(self.anno, "completions"):
			return self.anno.completions(text)

class Or:
	"""Support two or more alternative annotations
	
	Or(annotation, annotation [, annotation]*) -> annotation
	"""

	def __init__(self, *annotations):
		if len(annotations) < 2:
			raise ValueError("need at two alternative annotations")
		self.annotations = annotations

	def __call__(self, text):
		for anno in self.annotations:
			try:
				return anno(text)
			except ValueError:
				pass
		names = [a.__name__ for a in self.annotations]
		msg = ', '.join(names[:-1])
		if len(names) > 2:
			msg += ', '
		msg += 'or ' + names[-1]
		raise ValueError("excepted a %s" % msg)

	def completions(self, text):
		"""completions are the union of alternative annotation completions"""
		completions = []
		for anno in self.annotations:
			if hasattr(anno, "completions"):
				completions += anno.completions(text)
		return completions

class _Bool:

	def __call__(self, text):
		text = text.casefold()
		if text == "0" or "false".startswith(text):
			return False
		if text == "1" or "true".startswith(text):
			return True
		raise ValueError("invalid boolean literal")

	def completions(self, text):
		result = []
		text = text.casefold()
		if "false".startswith(text):
			result.append("false")
		if "true".startswith(text):
			result.append("true")
		return result
Bool = _Bool() #: Annotation for boolean literals

class _FunctionInfo:
	# cache information about functions

	__slots__ = [
		'function', 'annotations', 'keywords',
		'positionals', 'optionals', 'aggregates', 'booleans',
		'var_positional', 'var_keyword',
	]

	def __init__(self, function, signature):
		self.function = function
		self.annotations = {}
		self.keywords = set()
		self.positionals = []
		self.optionals = set()
		self.booleans = set()
		self.var_positional = False
		self.var_keyword = False

		import sys
		import inspect
		Param = inspect.Parameter
		for p in signature.parameters.values():
			if p.name[0] == '_':
				# private argument
				if (p.kind == Param.POSITIONAL_ONLY
				or (p.kind == Param.POSITIONAL_OR_KEYWORD
						and p.default == Param.empty)):
					raise ValueError("can not handle private positional arguments")
				continue
			if p.annotation == Param.empty:
				self.annotations[p.name] = None
			else:
				self.annotations[p.name] = p.annotation
			if p.kind == Param.POSITIONAL_OR_KEYWORD:
				if p.default == Param.empty:
					self.positionals.append(p.name)
				elif isinstance(p.annotation, Optional):
					self.positionals.append(p.name)
					self.optionals.add(p.name)
					if len(self.keywords) != 0:
						print("warning: Optional argument %s must be before other keyword arguments in %s" % (p.name, function), file=sys.stderr)
				elif isinstance(p.annotation, type) and issubclass(p.annotation, bool) and p.default != Param.empty:
					self.booleans.add(p.name)
					self.keywords.add(p.name)
					# TODO: relax this restriction
					if p.default:
						print("warning: boolean argument %s needs to default to non-True in %s: %s" % (p.name, function, p.default), file=sys.stderr)
				else:
					self.keywords.add(p.name)
			elif p.kind == Param.KEYWORD_ONLY:
				self.keywords.add(p.name)
				if isinstance(p.annotation, type) and issubclass(p.annotation, bool) and p.default != Param.empty:
					self.booleans.add(p.name)
					# TODO: relax this restriction
					if not p.default:
						print("warning: boolean argument %s needs to default to non-True in %s" % (p.name, function), file=sys.stderr)
			elif p.kind == Param.POSITIONAL_ONLY:
				if isinstance(p.annotation, Aggregate):
					raise ValueError("Aggregate values depend on repeated keywords")
				self.positionals.append(p.name)
			elif p.kind == Param.VAR_POSITIONAL:
				self.var_positional = True
			elif p.kind == Param.VAR_KEYWORD:
				self.var_keyword = True
		return
		print("""%s function info:
			positional: %s
			optional: %s
			aggregate: %s
			boolean: %s
			keyword: %s
		""" % (function,
			self.positionals,
			self.optionals,
			self.aggregates,
			self.booleans,
			self.keywords,
			))

# _commands is a map of command name to command function.  Except when
# it is a multiword command name, then the preliminary words map to
# dictionaries that map to the function.  And instead of the function,
# it has precomputed information about the function.
# An OrderedDict is used so for autocompletion, the prefix of the first
# registered command with that prefix is used.
_commands = OrderedDict()

def _check_autocomplete(word, mapping, name):
	# this is a debugging aid for developers
	for key in mapping:
		if key.startswith(word):
			raise ValueError("'%s' is a prefix of an existing command" % name)

class Defer:
	"""Enable function introspection to be deferred until needed

	Defer(proxy_function) -> instance
	
	There are two uses: (1) the proxy function returns the actual
	function that implements the command, or (2) the proxy function
	register subcommands and returns None.  In the former case,
	the proxy function will typically consist of an import statement,
	followed by returning a function in the imported module.  In the
	latter case, multiple subcommands are registered, and nothing is
	returned.
	"""
	__slots__ = [ 'proxy' ]

	def __init__(self, proxy_function):
		self.proxy = proxy_function

	def __call__(self):
		return self.proxy()

def register(name, function=None):
	"""register function that implements command
	
	:param name: the name of the command and may include spaces.
	:param function: the callback function.

	There are two ways to defer introspecting the function until
	it is actually used: (1) put the logic in an instance of the
	:py:class:`Defer` class, which, when called, returns the function
	for the command, or (2) give a string with a fully qualified
	name, *i.e.*, function.__module__ + '.' + function.__qualname__,
	which is imported.

	If the function is None, then it assumed that :py:func:`register`
	is being used as a decorator.

	For autocompletion, the first command registered with a
	given prefix wins.  Registering a command that is a prefix
	of an existing command is an error since it breaks backwards
	compatibility.
	"""
	if function is None: # act as a decorator
		import functools
		return functools.partial(register, name)
	words = name.split()
	cmd_map = _commands
	for word in words[:-1]:
		what = cmd_map.get(word, None)
		if isinstance(what, dict):
			cmd_map = what
			continue
		deferred = isinstance(what, Defer)
		if what is not None and not deferred:
			raise ValueError("command extends previous command")
		if not deferred:
			_check_autocomplete(word, cmd_map, name)
		d = cmd_map[word] = OrderedDict()
		cmd_map = d
	word = words[-1]
	what = cmd_map.get(word, None)
	if isinstance(what, dict):
		raise ValueError("command is part of multiword command")
	#if what is not None:
	#	# TODO: replacing, preserve extra keywords
	_check_autocomplete(word, cmd_map, name)
	if isinstance(function, (str, Defer)):
		# delay introspecting function
		fi = function
	else:
		# introspect immediately to give errors
		import inspect
		signature = inspect.signature(function)
		fi = _FunctionInfo(function, signature)
	cmd_map[word] = fi

def _not_implemented(*args):
	raise RuntimeError("command is not implemented or missing")

def _lazy_introspect(cmd_map, word):
	deferred = cmd_map[word]
	if isinstance(deferred, Defer):
		function = deferred()
		if function is None:
			# deferred function should have registered subcommands
			fi = cmd_map[word]
			if isinstance(fi, dict):
				return fi
			function = _not_implemented
	elif isinstance(deferred, str):
		module_name, function_name = deferred.rsplit('.', 1)
		try:
			import importlib
			module = importlib.import_module(module_name)
			function = getattr(module, function_name)
		except (ImportError, KeyError):
			function = _not_implemented
	else:
		raise RuntimeError("unknown deferred method")
	import inspect
	signature = inspect.signature(function)
	fi = _FunctionInfo(function, signature)
	cmd_map[word] = fi
	return fi

def add_keyword_arguments(name, info):
	"""TODO: Make known additional keyword argument(s) for a command
	
	:param name: the name of the command
	:param info: { keyword: annotation }
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
	fi = cmd_map.get(word, None)
	if fi is None:
		raise ValueError("'%s' is not a command" % name)
	if isinstance(fi, (str, Defer)):
		fi = _lazy_introspect(cmd_map, word)
	if isinstance(fi, dict):
		raise ValueError("'%s' is not the full command" % name)
	if not fi.var_keyword:
		raise ValueError("'%s' does not take a variable number of keywords" % name)
	# TODO: fail if there are conflicts with existing keywords
	# fi.keywords.add(kw)
	# TODO: save appropriate annotation somehow

import re
normal = re.compile(r"(\.|\S)*")
single = re.compile(r"'([^']|\')*'")
double = re.compile(r'"([^"]|\")*"')
whitespace = re.compile("\s+")

class Command:
	"""Keep track of partially typed command with possible completions
	
	:param text: the command text
	:param autocomplete: true if command names and keyword arguments
	should be automatically completed.
	"""

	def __init__(self, text="", autocomplete=True):
		self._reset()
		if text:
			self.parse_text(text, autocomplete)

	def _reset(self):
		self.current_text = ""
		self.amount_parsed = 0
		self.completion_prefix = ""
		self.completions = []
		self._error = "Missing command"
		self._word_map = _commands
		self._fi = None
		self._args = []
		self._kwargs = {}
		self._in_kwarg = ""
		self._in_array = None	# (arg_name, values)
		self._in_str = False

	def error_check(self):
		"""Error check results of calling parse_text


		# separate error checking logic from execute() so
		# it may be done separately
		"""
		if self._error:
			raise UserError(self._error)
		if (not self._fi.var_positional
		and len(self._args) > len(self._fi.positionals)):
			raise UserError("too many arguments")
		needed_args = len(self._fi.positionals) - len(self._fi.optionals) - len(self._args)
		if needed_args > 1:
			raise UserError("missing arguments")
		if needed_args == 1:
			raise UserError("missing argument")

	def execute(self, error_check=True):
		"""If command is valid, execute it"""
		if error_check:
			self.error_check()
		try:
			return self._fi.function(*self._args, **self._kwargs)
		except ValueError as e:
			# convert function's ValueErrors to UserErrors,
			# but not those of functions it calls
			import sys, traceback
			_, _, exc_traceback = sys.exc_info()
			if len(traceback.extract_tb(exc_traceback)) > 2:
				raise
			raise UserError(str(e))

	def _convert(self, arg_name, word, annotation):
		# Convert argument word from string to type
		# given a parameter annotation.  And update
		# possible completions if conversion fails.
		if annotation is None or not callable(annotation):
			return word
		try:
			return annotation(word)
		except ValueError as e:
			self._error = "Bad '%s' argument value: %s" % (arg_name, e)
			self.completion_prefix = ""
			if hasattr(annotation, 'completions'):
				self.completions = annotation.completions(word)
			else:
				self.completions = []
			raise

	def _next_token(self, text):
		# Return tuple of first argument in text and actual text used
		#
		# Arguments may be quoted, in which case the text between
		# the quotes is returned.  If there is no closing quote,
		# return rest of line for autocompletion purposes, but
		# take note.
		m = whitespace.match(text)
		start = m.end() if m else 0
		if start == len(text):
			return "", text[0:start]
		if text[0] == '"':
			m = double.match(text, start)
			if m:
				end = m.end()
				token = text[start + 1:end - 1]
			else:
				end = len(text)
				token = text[start + 1:end]
				self._in_str = True
				self._error = "incomplete quoted text"
		elif text[0] == "'":
			m = single.match(text, start)
			if m:
				end = m.end()
				token = text[start + 1:end - 1]
			else:
				end = len(text)
				token = text[start + 1:end]
				self._in_str = True
				self._error = "incomplete quoted text"
		else:
			m = normal.match(text, start)
			end = m.end()
			token = text[start:end]
		# convert \N{unicode name} to unicode, etc.
		token = token.encode('utf-8').decode('unicode-escape')
		return token, text[0:end]

	def _complete(self, chars, suffix):
		# insert completion taking into account quotes
		i = len(chars)
		c = chars[0]
		if c not in "\"'" or chars[-1] != c:
			completion = chars + suffix
		else:
			completion = chars[:-1] + suffix + chars[-1]
		j = self.amount_parsed
		t = self.current_text 
		self.current_text = t[0:j] + completion + t[i + j:]
		return self.current_text[j:]

	def parse_text(self, text, autocomplete=True):
		"""Parse text into function and arguments
		
		:param text: The text to be parsed.
		:param autocomplete: True if function and arguments that
		are a prefix of an actual should be completed. (TODO:
		should only unambigous prefixes be completed?)

		May be called multiple times.  There are a couple side effects:

		* The autocompleted text is put in self.current_text.
		* Possible completions are in self.completions.
		"""
		if not text.startswith(self.current_text[0:self.amount_parsed]):
			self._reset()
		elif self._in_str:
			self._reset()
		elif (0 < self.amount_parsed < len(text)
		and not self.current_text[-1].isspace()
		and not text[self.amount_parsed].isspace()):
			# easier to start over
			self._reset()

		self.current_text = text
		text = text[self.amount_parsed:]
		while 1:
			word, chars = self._next_token(text)
			if not word:
				# no more words
				self.amount_parsed += len(chars)
				#if self.current_text and not self._in_kwarg:
				#	if self.current_text[-1].isspace():
				#		fmt = '%s'
				#		self.completion_prefix = ""
				#	else:
				#		fmt = ' %s'
				#		self.completion_prefix = ' '
				#	self.completions = [fmt % w for w in self._word_map]
				#else:
				#	self.completion_prefix = ""
				#	self.completions = []
				break
			if self._in_array:
				arg_name, values = self._in_array 
				anno = self._fi.annotations[arg_name]
				try:
					value = self._convert(arg_name, word, anno)
				except ValueError:
					if len(values) >= anno.min_len:
						self._error = ""
						self.completion_prefix = ""
						self.completions = []
						continue
					break
				self.amount_parsed += len(chars)
				text = text[len(chars):]
				values.append(value)
				if len(values) >= anno.min_len:
					self._error = ""
					self.completion_prefix = ""
					self.completions = []
				if len(values) == anno.max_len:
					self._in_array = None
				continue
			if self._in_kwarg:
				anno = self._fi.annotations[self._in_kwarg]
				try:
					value = self._convert(self._in_kwarg, word, anno)
				except ValueError:
					break
				self.amount_parsed += len(chars)
				text = text[len(chars):]
				if not isinstance(anno, Aggregate):
					self._kwargs[self._in_kwarg] = value
				else:
					if self._in_kwarg not in self._kwargs:
						self._kwargs[self._in_kwarg] = anno.constructor()
					x = anno.add_to(self._kwargs[self._in_kwarg], value)
					if x is not None:
						self._kwargs[self._in_kwarg] = x
				self._in_kwarg = ""
				self._error = ""
				self.completion_prefix = ""
				self.completions = []
				continue
			if self._fi:
				what = word if word in self._word_map else None
			else:
				what = self._word_map.get(word, None)
			if what is not None:
				self.amount_parsed += len(chars)
				text = text[len(chars):]
				if self._fi:
					# word is an argument keyword
					if word in self._fi.booleans:
						self._kwargs[word] = True
						continue
					anno = self._fi.annotations[word]
					if isinstance(anno, Array_of):
						values = self._kwargs[word] = []
						self._in_array = word, values
						self._error = "Expected more values for %s keyword" % word
						continue
					self._in_kwarg = word
					self._error = "Expected value for %s keyword" % word
					self.completion_prefix = word
					self.completions = [kw for kw in self._word_map if kw.startswith(word)]
					continue
				if isinstance(what, (str, Defer)):
					what = _lazy_introspect(self._word_map, word)
				if isinstance(what, dict):
					# word is part of multiword command name
					self._word_map = what
					self._error = "Incomplete command: %s" % self.current_text[0:self.amount_parsed]
					continue
				# word was last word in command name
				self._fi = what
				self._word_map = self._fi.keywords
				self._error = ""
				self.completion_prefix = ""
				self.completions = []
				continue
			if not self._fi:
				# haven't typed in full command name yet
				self._error = "Unknown command"
				self.completion_prefix = word
				self.completions = [cmd for cmd in self._word_map if cmd.startswith(word)]
				if autocomplete and self.completions:
					c = self.completions[0]
					text = self._complete(chars, c[len(word):])
					continue
				break
			# assume positional argument
			# but might be the start of a keyword argument
			self.completion_prefix = word
			self.completions = [kw for kw in self._word_map if kw.startswith(word)]
			name = None
			if len(self._args) < len(self._fi.positionals):
				name = self._fi.positionals[len(self._args)]
				anno = self._fi.annotations[name]
				if isinstance(anno, Array_of):
					values = []
					if name in self._fi.optionals:
						self._kwargs[name] = values
					else:
						self._args.append(values)
					self._in_array = name, values
					self._error = "Expected more values for %s keyword" % word
					continue
				try:
					value = self._convert(name, word, anno)
				except ValueError:
					break
			elif autocomplete and self.completions:
				c = self.completions[0]
				text = self._complete(chars, c[len(word):])
				continue
			else:
				value = word
			self.amount_parsed += len(chars)
			text = text[len(chars):]
			if name in self._fi.optionals:
				self._kwargs[name] = value
			else:
				self._args.append(value)

if __name__ == '__main__':

	@register('test1')
	def test1(a: int, b: float, color=None):
		print('test1 a: %s %s' % (type(a), a))
		print('test1 b: %s %s' % (type(b), b))
		print('test1 color: %s %s' % (type(color), color))

	@register('test2')
	def test2(a: str, *args, color=None, center: float=0):
		print('test2 a: %s %s' % (type(a), a))
		print('test2 args: %s %s' % (type(args), args))
		print('test2 color: %s %s' % (type(color), color))
		print('test2 center: %s %s' % (type(center), center))

	register('mw test1', test1)
	register('mw test2', test2)

	@register('test3')
	def test3(name: str, value: Optional(float)=None):
		print('test3 name: %s %s' % (type(name), name))
		print('test3 value: %s %s' % (type(value), value))

	@register('test4')
	def test4(draw: bool=None):
		print('test4 draw: %s %s' % (type(draw), draw))

	@register('test5')
	def test5(ints: List_of(int)=None):
		print('test5 ints: %s %s' % (type(ints), ints))

	@register('test6')
	def test6(center: Array_of(float, 3)):
		print('test6 center:', center)

	@register('test7')
	def test7(center: Array_of(float, 3)=None):
		print('test7 center:', center)

	tests = [
		(True,	'test1 color red 12 3.5'),
		(True,	'test1 12 color red 3.5'),
		(True,	'test1 12 3.5 color red'),
		(True,	'test1 color red xyzzy 3.5'),
		(True,	'test1 color red'),
		(True,	'test1 color'),
		(True,	'te'),
		(True,	'test2 color red center 3.5 foo'),
		(True,	'test2 color red center 3.5'),
		(True,	'test2 color red center xyzzy'),
		(True,	'test2 color red center'),
		(True,	'test2 c'),
		(True,	'test3 radius'),
		(True,	'test3 radius 12.3'),
		(True,	'test4'),
		(True,	'test4 draw'),
		(True,	'test5'),
		(True,	'test5 ints 5'),
		(True,	'test5 ints 5 ints 6'),
		(True,	'mw test1 color red 12 3.5'),
		(True,	'mw test1 color red 12 3.5'),
		(True,	'mw test2 color red center 3.5 foo'),
		(True,	'mw te'),
		(True,	'mw '),
		(True,	'mw'),
		(True,	'te co red 12 3.5'),
		(True,	'm te col red 12 3.5'),
		(True,	'test6 3.4 5.6 7.8'),
		(True,	'test6 3.4 abc 7.8'),
		(True,	'test7 center 3.4 5.6 7.8'),
		(True,	'test7 center 3.4 5.6'),
	]
	cmd = Command()
	for t in tests:
		autocomplete, text = t
		try:
			print("\nTEST: '%s'" % text)
			cmd.parse_text(text, autocomplete=autocomplete)
			if cmd.current_text != text:
				print(cmd.current_text)
			#print(cmd._args, cmd._kwargs)
			p = cmd.completions
			if p:
				print('completions:', p)
			cmd.execute()
			print('SUCCESS')
		except UserError as e:
			print('FAIL:', e)
