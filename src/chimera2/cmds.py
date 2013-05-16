"""
cmds: Support for application command lines
===========================================

To add a command, first :py:func:`register` it and then call
:py:func:`process_command` to parse the text and invoke the
appropriate function.

The registered command functions are introspected to figure out the
positional and keyword arguments.  Unlike Python, where any function
argument can be given as a keyword argument, only function arguments with
default values are considered keyword arguments.  The function argument
name is used as the keyword in text commands.  Argument annotations are
used to convert arguments to the correct type.  Keyword arguments that
start with an underscore are ignored.

Unlike Python and like typical UNIX shell commands, in a textual
command, the keyword arguments are given first.  Keyword arguments
can be abbreviated to their unique prefixes of at least 3 letters.
Argument values may be quoted with either single or double quotes.

.. todo::

    Add mechanism to supply alternative argument conversion functions.

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
	'process_command',
	'UserError',
]

# _commands is a map of comamnd name to command function
_commands = {}

def register(name, function):
	"""register function that implements command
	
	:param name: the name of the command
	:param funtion: the callback function
	"""
	_commands[name] = function

class UserError(ValueError):
	"""Use for cases where user provided input causes an error."""
	pass

def process_command(text):
	"""Parse command text and execute command.
	
	:py:exc:`UserError` is thrown if there is a parsing error."""

	cmd_name, arg_text = first_arg(text)
	cmd_func = _commands.get(cmd_name, None)
	if not cmd_func:
		raise UserError('unknown command: %s' % cmd_name)
	try:
		return exec_with_args(cmd_func, arg_text)
	except UserError as exc:
		raise UserError('bad command: %s: %s' % (cmd_name, exc))

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

def keyword_match(keyword, keywords, unique=True, case_independent=False):
	"""match keyword in list of keywords allowing prefix matching

	If unique is False, then return first match from list.
	If case_independent is True, then assume that all keywords are
	in lowercase, and lowercase the given keyword to find a match.
	"""
	if case_independent:
		keyword = keyword.casefold()
	if not unique:
		for i, k in enumerate(keywords):
			if k.startswith(keyword):
				return i
	else:
		if len(keyword) >= 3:
			matches = [k for k in keywords if k.startswith(keyword)]
		else:
			matches = [k for k in keywords if k == keyword]
		if not matches:
			return None
		# check for an unique match
		if len(matches) == 1:
			return matches[0]
		# check for an exact match
		for i, k in enumerate(keywords):
			if k == keyword:
				return i
		raise UserError("Keyword '%s' matches multiple known"
			" keywords: %s" % (keyword, ' '.join(matches)))
	return None

def exec_with_args(function, text):
	"""Call function with parsed argument text

	Parse the text to supply the positional and keyword arguments
	needed by the given function.  Argument types are determined by
	introspeciton.  See this module's documentation for the rules.
	"""

	keyword_arguments = []
	positional_arguments = []
	var_positional = False
	import inspect
	Param = inspect.Parameter
	sig = inspect.signature(function)
	for p in sig.parameters.values():
		if p.name[0] == '_':
			# private argument
			if (p.kind == Param.POSITIONAL_ONLY
			or (p.kind == Param.POSITIONAL_OR_KEYWORD
						and p.default == P.empty)):
				raise ValueError("can not handle private positional arguments")
			continue
		if p.kind == Param.POSITIONAL_OR_KEYWORD:
			if p.default == Param.empty:
				positional_arguments.append(p.name)
			else:
				keyword_arguments.append(p.name)
		elif p.kind == Param.KEYWORD_ONLY:
			keyword_arguments.append(p.name)
		elif p.kind == Param.POSITIONAL_ONLY:
			positional_arguments.append(p.name)
		elif p.kind == Param.VAR_POSITIONAL:
			var_positional = True
		elif p.kind == Param.VAR_KEYWORD:
			# ignore unknown unnamed keyword arguments
			pass

	args = []
	kwd_args = {}
	try_keyword = len(keyword_arguments) != 0
	while text:
		arg, text = first_arg(text)
		keyword = None
		if try_keyword:
			keyword = keyword_match(arg, keyword_arguments)
		if keyword is None:
			try_keyword = False
			if len(args) < len(positional_arguments):
				name = positional_arguments[len(args)]
				p = sig.parameters[name]
				if p.annotation is not Param.empty:
					try:
						arg = convert(arg, p.annotation)
					except Exception as e:
						raise UserError("Bad '%s' argument: %s" % (p.name, e))
			args.append(arg)
			continue

		if not text:
			raise UserError("missing value for '%s' argument" % keyword)
		kw_arg, text = first_arg(text)
		p = sig.parameters[keyword]
		if p.annotation is not Param.empty:
			try:
				kw_arg = convert(kw_arg, p.annotation)
			except Exception as e:
				raise UserError("Bad keyword '%s' argument: %s" % (p.name, e))
		kwd_args[keyword] = kw_arg

	if not var_positional and len(args) > len(positional_arguments):
		raise UserError("too many arguments")
	if len(args) < len(positional_arguments):
		raise UserError("missing arguments")

	return function(*args, **kwd_args)

if __name__ == '__main__':

	def testfunc(a: int, b: float, color=None):
		print('a: %s %s' % (type(a), a))
		print('b: %s %s' % (type(b), b))
		print('color: %s %s' % (type(color), color))

	register('test', testfunc)
	try:
		process_command('test color red 12 3.5')
		process_command('test color red xyzzy 3.5')
	except UserError as e:
		print(e)
