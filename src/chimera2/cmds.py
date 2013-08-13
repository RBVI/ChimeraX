"""
cmds: Support for application command lines
===========================================

To add a command, :py:func:`register` it.  Later, when processing
the command line, create a :py:class:`Command` object with the text,
optionally update the text as more of the command line is available
to find out possible command and keyword completions, then
:py:func:`Command.execute` it.

The registered command functions are introspected to figure out the
positional and keyword arguments.  Unlike Python, where any function
argument can be given as a keyword argument, only function arguments
with default values are considered keyword arguments.  The argument name
is used as the keyword in text commands.  Argument annotations are used
to convert arguments to the correct type.  Keyword arguments that start
with an underscore are ignored.

Argument values may be quoted with either single or double quotes.
Keyword and positional arguments may be interspersed.

.. todo:

    Add type registry that, in additions to syntax checking, can give
    autocompletions suggestions, e.g., for enumerated values or filenames.

.. todo::

    Expand quoting code to support escaped quotes:
    ``'ab'\\'"cd"`` should be ``ab'cd``

.. todo::

    Build data structure with introspected information and allow it to
    be supplemented separately for command functions with \*\*kw arguments.
    That way a command that is expanded at runtime could pick up new arguments
    (e.g., the open command).

.. todo::

    Maybe let typed argument conversion determine how much text to consume
    (so arguments could have spaces in them)
"""

__all__ = [
	'register',
	'Command',
	'UserError',
]

import inspect
Param = inspect.Parameter	# shortcut to Parameter constants
from collections import OrderedDict
try:
	from .orderedset import OrderedSet
except SystemError:
	# for debugging
	from orderedset import OrderedSet

class UserError(ValueError):
	"""Use for cases where user provided input causes an error."""
	pass

class _FunctionInfo:
	# cache information about functions

	__slots__ = [
		'function', 'signature', 'keyword_arguments',
		'positional_arguments', 'var_positional'
	]

	def __init__(self, function, signature):
		self.function = function
		self.signature = signature
		self.keyword_arguments = OrderedSet()
		self.positional_arguments = []
		self.var_positional = False

		for p in signature.parameters.values():
			if p.name[0] == '_':
				# private argument
				if (p.kind == Param.POSITIONAL_ONLY
				or (p.kind == Param.POSITIONAL_OR_KEYWORD
						and p.default == Param.empty)):
					raise ValueError("can not handle private positional arguments")
				continue
			if p.kind == Param.POSITIONAL_OR_KEYWORD:
				if p.default == Param.empty:
					self.positional_arguments.append(p.name)
				else:
					self.keyword_arguments.add(p.name)
			elif p.kind == Param.KEYWORD_ONLY:
				self.keyword_arguments.add(p.name)
			elif p.kind == Param.POSITIONAL_ONLY:
				self.positional_arguments.append(p.name)
			elif p.kind == Param.VAR_POSITIONAL:
				self.var_positional = True
			elif p.kind == Param.VAR_KEYWORD:
				# ignore unknown unnamed keyword arguments
				pass

# _commands is a map of command name to command function.  Except when
# it is a multiword command name, then the preliminary words map to
# dictionaries that map to the function.  And instead of the function,
# it has precomputed information about the function.
_commands = OrderedDict()

def register(name, function):
	"""register function that implements command
	
	:param name: the name of the command
	:param function: the callback function

	Command names may have spaces in them.

	For autocompletion, the first command registered with a given
	prefix wins.
	"""
	words = name.split()
	cmd_map = _commands
	for word in words[:-1]:
		what = cmd_map.get(word, None)
		if isinstance(what, dict):
			cmd_map = what
			continue
		if what is not None:
			raise ValueError("command extends previous command")
		d = cmd_map[word] = OrderedDict()
		cmd_map = d
	word = words[-1]
	what = cmd_map.get(word, None)
	if isinstance(what, dict):
		raise ValueError("command is part of multiword command")
	signature = inspect.signature(function)
	fi = _FunctionInfo(function, signature)
	cmd_map[word] = fi

class Command:
	"""Keep track of partially typed command with possible completions
	
	:param text: the command text
	:param autocomplete: true if command names and keyword arguments
	should be automatically completed (only recommended for
	non-interactive use).
	"""

	def __init__(self, text="", autocomplete=False):
		self._reset()
		if text:
			self.parse_text(text, autocomplete)

	def _reset(self):
		self.current_text = ""
		self.amount_parsed = 0
		self.completion_prefix = ""
		self.completions = []
		self._in_kwarg = ""
		self._error = "no command"
		self._word_map = _commands
		self._fi = None
		self._args = []
		self._kwargs = {}

	def execute(self):
		if self._error:
			raise UserError(self._error)
		if (not self._fi.var_positional
		and len(self._args) > len(self._fi.positional_arguments)):
			raise UserError("too many arguments")
		if len(self._args) < len(self._fi.positional_arguments):
			raise UserError("missing arguments")
		return self._fi.function(*self._args, **self._kwargs)

	def parse_text(self, text, autocomplete=False):
		if not text.startswith(self.current_text[0:self.amount_parsed]):
			self._reset()
		if (self.amount_parsed > 0
		and not self.current_text[-1].isspace()
		and not text[self.amount_parsed].isspace()):
			# easier to start over
			self._reset()

		self.current_text = text
		text = text[self.amount_parsed:]
		while text:
			word, rest = first_arg(text)
			if self._in_kwarg:
				p = self._fi.signature.parameters[self._in_kwarg]
				if p.annotation is not Param.empty:
					try:
						word = convert(word, p.annotation)
					except ValueError as e:
						self._error = "Bad '%s' argument: %s" % (p.name, e)
						self.completion_prefix = ""
						self.completions = []
						break
				self.amount_parsed += len(text) - len(rest)
				text = rest
				self._kwargs[self._in_kwarg] = word
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
				self.amount_parsed += len(text) - len(rest)
				text = rest
				if self._fi:
					self._in_kwarg = word
					self._error = "expect argument for %s keyword" % word
					self.completion_prefix = word
					self.completions = [kw for kw in self._word_map if kw.startswith(word)]
					continue
				if isinstance(what, dict):
					self._word_map = what
					self._error = "incomplete command: %s" % self.current_text[0:self.amount_parsed]
					if not rest:
						if self.current_text[-1].isspace():
							fmt = '%s'
							self.completion_prefix = ""
						else:
							fmt = ' %s'
							self.completion_prefix = ' '
						self.completions = [fmt % cmd for cmd in self._word_map.keys()]
					else:
						self.completion_prefix = ""
						self.completions = []
					continue
				self._fi = what
				self._word_map = self._fi.keyword_arguments
				self._error = ""
				self.completion_prefix = ""
				self.completions = []
				continue
			if not self._fi:
				# haven't typed in full command name yet
				self._error = "unknown command"
				self.completion_prefix = word
				self.completions = [cmd for cmd in self._word_map if cmd.startswith(word)]
				if autocomplete and self.completions:
					c = self.completions[0]
					i = len(word)
					text = c + text[i:]
					j = self.amount_parsed
					t = self.current_text 
					self.current_text = t[0:j] + c + t[i + j:]
					continue
				break
			# assume positional argument
			# but might be the start of a keyword argument
			self.completion_prefix = word
			self.completions = [kw for kw in self._word_map if kw.startswith(word)]
			if autocomplete and self.completions:
				c = self.completions[0]
				i = len(word)
				text = c + text[i:]
				j = self.amount_parsed
				t = self.current_text 
				self.current_text = t[0:j] + c + t[i + j:]
				continue
			if len(self._args) < len(self._fi.positional_arguments):
				name = self._fi.positional_arguments[len(self._args)]
				p = self._fi.signature.parameters[name]
				if p.annotation is not Param.empty:
					try:
						word = convert(word, p.annotation)
					except ValueError as e:
						self._error = "Bad '%s' argument: %s" % (p.name, e)
						self.completion_prefix = ""
						self.completions = []
						break
			self.amount_parsed += len(text) - len(rest)
			text = rest
			self._args.append(word)

def convert(arg, annotation):
	"""Convert argument to type given a parameter annotation

	Right now, assume that the annotation is a type that can
	be used as a constructor that takes a string argument, e.g., int.
	Might be more elaborate in the future.
	"""
	if callable(annotation):
		return annotation(arg)
	return arg

def first_arg(text):
	"""Return tuple of first argument in text and rest of text

	Arguments may be quoted, in which case the text between the quotes
	is returned.
	"""
	if text and text[0] in ["'", '"']:
		i = text.find(text[0], 1)
		if i == -1:
			raise UserError("Unmatched quote in argument")
		return text[1:i], text[i + 1:].lstrip()
	tmp = text.split(None, 1)
	if not tmp:
		return "", ""
	if len(tmp) == 1:
		return tmp[0], ""
	return tmp

if __name__ == '__main__':

	def testfunc(a: int, b: float, color=None):
		print('a: %s %s' % (type(a), a))
		print('b: %s %s' % (type(b), b))
		print('color: %s %s' % (type(color), color))
	register('test', testfunc)

	def tempfunc(a: str, *args, color=None, center: float=0):
		pass
	register('temp', tempfunc)

	register('mw test', testfunc)
	register('mw temp', tempfunc)

	tests = [
		'test color red 12 3.5',
		'test 12 color red 3.5',
		'test 12 3.5 color red',
		'test color red xyzzy 3.5',
		'test color red',
		'test color',
		'te',
		'temp color red center 3.5 foo',
		'temp color red center 3.5',
		'temp color red center xyzzy',
		'temp color red center',
		'temp c',
		'tem',
		'mw test color red 12 3.5',
		'mw temp color red center 3.5 foo',
		'mw te',
		'mw ',
		'mw',
	]
	autotests = [
		'te co red 12 3.5',
		'm te col red 12 3.5',
	]
	for t in tests:
		try:
			print("\nTEST: '%s'" % t)
			c = Command(t)
			p = c.completions
			if p:
				print('completions:', p)
			c.execute()
		except UserError as e:
			print(e)
	for t in autotests:
		try:
			print("\nAUTOCOMPLETE TEST: '%s'" % t)
			c = Command(t, autocomplete=True)
			print(c.current_text)
			p = c.completions
			if p:
				print('completions:', p)
			c.execute()
		except UserError as e:
			print(e)
