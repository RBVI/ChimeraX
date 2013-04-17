"""
cmds: Support for application command lines
===========================================

To add a command, first register() it
and then process_text() it to parse the text and call the appropriate function.
The register()ed command functions are introspected to
figure out their positional and keyword arguments.
Keyword arguments can be abbreviated to their unique prefixes.
Keywords that start with an underscore are ignored.
Argument values may be quoted.

.. todo::

    add mechanism to supply optional argument type information (default is
    string) and register a parser for that type.
    In Python3 function arguments can be annotated.

.. todo::

    expand quoting code to support escaped quotes:
    ``'ab'\\'"cd"`` should be ``ab'cd``

.. todo::

    build data structure with introspected information and allow it to
    be supplemented separately for command functions with \*\*kw arguments.
    That way a command that is expanded at runtime could pick up new arguments
    (e.g., the open command).

"""

# _commands is a map of comamnd_name to command_function
_commands = {}

def register(command_name, command_function):
	"""register function that implements command"""
	_commands[command_name] = command_function

class UserError(ValueError):
	"""Use for cases where user provided input causes an error"""
	pass

def process_command(text):
	"""Parse command text and execute command."""

	cmd_name, arg_text = first_arg(text)
	cmd_func = _commands.get(cmd_name, None)
	if not cmd_func:
		raise UserError('unknown command: %s' % cmd_name)
	try:
		return exec_with_args(cmd_func, arg_text)
	except UserError as exc:
		raise UserError('bad command: %s: %s' % (cmd_name, exc))

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
		keyword = keyword.lower()
	if not unique:
		for i, k in enumerate(keywords):
			if k.startswith(keyword):
				return i
	else:
		matches = [(i, k) for i, k in enumerate(keywords) if k.startswith(keyword)]
		if len(matches) == 1:
			return matches[0][0]
		for i, k in enumerate(keywords):
			if k == keyword:
				return i
		raise UserError("Keyword '%s' matches multiple known"
			" keywords: %s" % (
				keyword, " ".join([m[1] for m in matches])))
	raise UserError("Keyword '%s' doesn't match any known keywords" % keyword)

def exec_with_args(function, text):
	"""Call function with parsed argument text

	Parse the text to supply the positional and keyword arguments
	needed by the given function.  Those arguments are determined
	by introspecting the function.  Keyword arguments can be
	abbreviated to their unique prefixes.  Keywords that start
	with an underscore are ignored.  Argument values may be quoted.

	TODO: extend with argument typechecking and let typed argument
	parsing determine how much text to consume (so arguments could
	have spaces in them)
	"""

	import inspect
	arg_spec = inspect.getargspec(function)
	if arg_spec.varargs or arg_spec.keywords:
		raise ValueError("can not handle functions variable number of arguments")
	if arg_spec.defaults:
		num_positional = len(arg_spec.args) - len(arg_spec.defaults)
		keyword_arguments = [k for k in arg_spec.args[num_positional:]
						if not k.startswith('_')]
	else:
		num_positional = len(arg_spec.args)
		keyword_arguments = []
	if hasattr(function, '__self__') and function.__self__ is not None:
		num_positional -= 1	# ignore bound self argument

	args = []
	kwd_args = {}
	while text:
		arg, text = first_arg(text)
		if len(args) < num_positional:
			args.append(arg)
			continue
		if not keyword_arguments:
			raise UserError("too many arguments")
		keyword = keyword_match(arg, keyword_arguments)
		if not text:
			raise UserError("missing value for %s argument" % keyword)
		kw_arg, text = first_arg(text)
		kwd_args[keyword] = kw_arg
	if len(args) < num_positional:
		raise UserError("missing arguments")

	return function(*args, **kwd_args)
