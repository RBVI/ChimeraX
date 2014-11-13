# vim: set expandtab shiftwidth=4 softtabstop=4:
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

In addition to registering command functions,
there is a separate mechanism to support textual `Command Aliases`_.

Text Commands
-------------

Synopsis::

    command_name rv1 rv2 [ov1 [ov2]] [kn1 kv1] [kn2 kv2]

Text commands are composed of a command name, which can be multiple words,
followed by required positional arguments, *rvX*,
optional positional arguments, *ovX*,
and keyword arguments with a value, *knX kvX*.
Each argument has an associated Python argument name
(for keyword arguments it is the keyword, *knX*).
*rvX*, *ovX*, and *kvX* are the type-checked values.
The names of the optional arguments are used to
let them be given as keyword arguments as well.
Multiple value arguments are separated by commas
and the commas may be followed by whitespace.
Depending on the type of an argument, *e.g.*, a color name,
whitespace can also appear within an argument value.
Argument values may be quoted with double quotes.
And in quoted text, Python's textual escape sequences are recognized,
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
See :py:func:`register` and :py:func:`delay_registration` for details.

The description is either an instance of the Command Description class,
:py:class:`CmdDesc`, or a tuple with the arguments to the initializer.
The CmdDesc initializer takes tuples describing the required, optional,
and keyword arguments.
Each tuple contains tuples with the argument name and a type annotation
(see below).
Postconditions (see below) can be given too.

Command Functions
-----------------

The command function arguments are expected to start with a ``session``
argument.  The rest of the arguments are assembled as keyword arguments,
as built from the command line and the command description.
The initial ``session`` argument to a command function
is not part of the command description.

Type Annotations
----------------

There are many standard type notations and they should be reused
as much as possible:

+-------------------------------+---------------------------------------+
|  Type                         |  Annotation                           |
+===============================+=======================================+
+ :py:class:`bool`              | ``BoolArg``                           |
+-------------------------------+---------------------------------------+
+ :py:class:`float`             | ``FloatArg``                          |
+-------------------------------+---------------------------------------+
+ :py:class:`int`               | ``IntArg``                            |
+-------------------------------+---------------------------------------+
+ :py:class:`str`               | ``StringArg``                         |
+-------------------------------+---------------------------------------+
+ tuple of 3 :py:class:`bool`   | ``Bool3Arg``                          |
+-------------------------------+---------------------------------------+
+ tuple of 3 :py:class:`float`  | ``Float3Arg``                         |
+-------------------------------+---------------------------------------+
+ tuple of 3 :py:class:`int`    | ``Int3Arg``                           |
+-------------------------------+---------------------------------------+
+ list of :py:class:`float`     | ``FloatsArg``                         |
+-------------------------------+---------------------------------------+
+ list of :py:class:`int`       | ``IntsArg``                           |
+-------------------------------+---------------------------------------+

.. MoleculeArg(s):
.. MoleculesArg(s, min = 0):
.. AtomsArg(s):
.. ModelArg(s):
.. ModelsArg(s):
.. SpecifierArg(s):
.. OpenstateArg(s):
.. VolumeArg(v):
.. VolumesArg(v):
.. SurfacesArg(s):
.. SurfacePiecesArg(spec):
.. MultiscaleSurfacePiecesArg(spec):
.. PointsArg(a):

There is one special annotation: :py:obj:`RestOfLine` that consumes
the rest of the command line as text.

Annotations are used to parse text and to support automatic completion.
Annotations can be extended with various specializers:

+-----------------------+-----------------------------------------------+
|  Specializer          |  Example                                      |
+=======================+===============================================+
+ :py:class:`Bounded`   | ``Bounded(FloatArg, 0.0, 100.0)``             |
+-----------------------+-----------------------------------------------+
+ :py:class:`ListOf`    | ``ListOf(FloatArg)``                          |
+                       | *a.k.a.*, ``FloatsArg``                       |
+-----------------------+-----------------------------------------------+
+ :py:class:`SetOf`     | ``SetOf(IntArg)``                             |
+-----------------------+-----------------------------------------------+
+ :py:class:`TupleOf`   | ``TupleOf(FloatArg, 3)``                      |
+                       | *a.k.a.*, ``Float3Arg``                       |
+-----------------------+-----------------------------------------------+
+ :py:class:`Or`        | ``Or(FloatArg, StringArg)``   *discouraged*   |
+-----------------------+-----------------------------------------------+
+ :py:class:`EnumOf`    | enumerated values                             |
+-----------------------+-----------------------------------------------+

Creating Your Own Type Annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annotations perform several functions:
(1) to convert text to a value of the appropriate type,
(2) to give reasonable error messages,
and (3) to provide possible completions for incomplete text.

See the :py:class:`Annotation` documentation for details.

Example
-------

Here is a simple example::

    import cli
    @register("echo", cli.CmdDesc(optional=[('text', cli.RestOfLine)]))
    def echo(session, text=''):
        print(text)
    ...
    command = cli.Command(session)
    command.parse_text(text, final=True)
    try:
        status = command.execute()
        if status:
            print(status)
    except cli.UserError as err:
        print(err, file=sys.stderr)


Command Aliases
---------------

    Normally, command aliases are made with the alias command, but
    they can also be explicitly register with :py:func:`alias` and
    removed with :py:func:`unalias`.

    An alias definition uses **$n** to refer to passed in arguments.
    $1 may appear more than once.  $$ is $.

    To register a multiword alias, quote the command name.


.. todo::

    Issues: autocompletion, minimum 2 letters? extensions?
    help URL? affect graphics flag?

"""

import abc
import re
import sys
from collections import OrderedDict

_debugging = False
_normal_token = re.compile(r"[^,;\s]*")
_single = re.compile(r"'([^']|\')*'")
_double = re.compile(r'"([^"]|\")*"')
_whitespace = re.compile("\s+")


class UserError(ValueError):
    """An exception provoked by the user's input.

    UserError(object) -> a UserError object

    This is in contrast to a error is a bug in the program.
    """
    pass


class Annotation(metaclass=abc.ABCMeta):
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
    def parse(text, session):
        """Return text converted to appropriate type.

        :param text: command line text to parse
        :param session: for session-dependent data types

        Abbreviations should be not accepted, instead they
        should be discovered via the possible completions.
        """
        raise NotImplemented

    @staticmethod
    def completions(text, session):
        """Return list of possible completions of the given text.

        :param text: Text to check for possible completions
        :param session: for session-dependent data types
        :returns: list of possible completions

        It is better to return an empty list (i.e., no known possible
        completions), because if invalid completions are given, then
        parsing can go into an infinite loop when trying to automatically
        complete text.
        """
        raise NotImplemented


class Aggregate(Annotation):
    """Common class for collections of values.

    Aggregate(annotation, constructor, add_to, min_size=None, max_size=None,
            name=None) -> annotation

    :param annotation: annotation for values in the collection.
    :param min_size: minimum size of collection, default `None`.
    :param max_size: maximum size of collection, default `None`.
    :param name: optionally override name in error messages.

    This class is typically used via :py:func:`ListOf`, :py:func:`SetOf`,
    or :py:func:`TupleOf`.
    The comma separator for aggregate values is handled by the
    :py:class:`Command` class, so parsing and completions are
    delegated to the underlying annotation.

    Subclasses need to set the constructor attribute and replace
    the add_to method.
    """
    min_size = 0
    max_size = sys.maxsize
    constructor = None

    def __init__(self, annotation, min_size=None,
                 max_size=None, name=None):
        if not issubclass(annotation, Annotation):
            raise ValueError("need an annotation, not %s" % annotation)
        self.annotation = annotation
        self.multiword = annotation.multiword
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

    def add_to(self, container, element):
        """Add to add an element to the container

        :param container: the container to add elements to
        :param element: the element to add to the container
        :returns: None for mutable containers, or a new
            container if immutable.
        """
        raise NotImplemented

    def parse(self, text, session):
        return self.annotation.parse(text, session)

    def completions(self, text, session):
        return self.annotation.completions(text, session)


class ListOf(Aggregate):
    """Annotation for lists of a single type

    ListOf(annotation, min_size=None, max_size=None) -> annotation
    """
    constructor = list

    def add_to(self, container, value):
        container.append(value)


class SetOf(Aggregate):
    """Annotation for sets of a single type

    SetOf(annotation, min_size=None, max_size=None) -> annotation
    """
    constructor = set

    def add_to(self, container, value):
        container.add(value)


# TupleOf function acts like Annotation subclass
class TupleOf(Aggregate):
    """Annotation for tuples of a single type

    TupleOf(annotation, size) -> annotation
    """
    constructor = tuple

    def __init__(self, annotation, size, name=None):
        return Aggregate.__init__(self, annotation, size, size, name=name)

    def add_to(self, container, value):
        return container + (value,)


class BoolArg(Annotation):
    """Annotation for boolean literals"""
    name = "a truth value"

    @staticmethod
    def parse(text, session):
        text = text.casefold()
        if text == "0" or text == "false":
            return False
        if text == "1" or text == "true":
            return True
        raise ValueError("Expected true or false (or 1 or 0)")

    @staticmethod
    def completions(text, session):
        result = []
        folded = text.casefold()
        if "false".startswith(folded):
            result.append(text + "false"[len(text):])
        if "true".startswith(folded):
            result.append(text + "true"[len(text):])
        return result


class IntArg(Annotation):
    """Annotation for integer literals"""
    name = "a whole number"

    @staticmethod
    def parse(text, session):
        try:
            return int(text)
        except ValueError:
            raise ValueError("Expected %s" % IntArg.name)

    @staticmethod
    def completions(text, session):
        chars = "-+0123456789"
        if not text:
            return [x for x in chars]
        if text[-1] in chars:
            tmp = [text + x for x in chars[2:]]
            if text[-1] in chars[2:]:
                return [text] + tmp
            return tmp
        return []


class FloatArg(Annotation):
    """Annotation for floating point literals"""
    name = "a floating point number"

    @staticmethod
    def parse(text, session):
        try:
            return float(text)
        except ValueError:
            raise ValueError("Expected %s" % FloatArg.name)

    @staticmethod
    def completions(text, session):
        chars = "-+0123456789."
        if not text:
            return [x for x in chars]
        # TODO: be more elaborate
        return []


class StringArg(Annotation):
    """Annotation for text (a word or quoted)"""
    name = "a text string"

    @staticmethod
    def parse(text, session):
        return text

    @staticmethod
    def completions(text, session):
        # strings can be extended arbitrarily, so don't make suggestions
        return []


class Bounded(Annotation):
    """Support bounded numerical values

    Bounded(annotation, min=None, max=None, name=None) -> an Annotation

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

    def parse(self, text, session):
        value = self.anno.parse(text, session)
        if self.min is not None and value < self.min:
            raise ValueError("Must be greater than or equal to %s" % self.min)
        if self.max is not None and value > self.max:
            raise ValueError("Must be less than or equal to %s" % self.max)
        return value

    def completions(self, text, session):
        #if not text:
            return self.anno.completions(text, session)
        # better to return that there are no completions,
        # rather than return completions that are out of bounds
        #return []


class EnumOf(Annotation):
    """Support enumerated types

    EnumOf(values, ids=None, name=None) -> an Annotation

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
                raise ValueError("Must have an identifier for "
                                 "each and every value")
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
            name = "one of '%s', or '%s'" % ("', '".join(self.ids[0:-1]),
                                             self.ids[-1])
        self.name = name

    def parse(self, text, session):
        folded = text.casefold()
        for i, x in enumerate(self.ids):
            if x.casefold() == folded:
                return self.values[i]
        raise ValueError("Invalid %s" % self.name)

    def completions(self, text, session):
        folded = text.casefold()
        return [text + x[len(text):] for x in self.ids
                if x.casefold().startswith(folded)]


class Or(Annotation):
    """Support two or more alternative annotations

    Or(annotation, annotation [, annotation]*, name=None) -> an Annotation

    :param name: optional explicit name for annotation
    """

    def __init__(self, *annotations, name=None):
        if len(annotations) < 2:
            raise ValueError("Need at two alternative annotations")
        self.annotations = annotations
        self.multiple = any([a.multiple for a in annotations])
        if name is None:
            name = "%s, or %s" % (", ".join(annotations[0:-1]),
                                  annotations[-1])
        self.name = name

    def parse(self, text, session):
        for anno in self.annotations:
            try:
                return anno.parse(text, session)
            except ValueError:
                pass
        names = [a.__name__ for a in self.annotations]
        msg = ', '.join(names[:-1])
        if len(names) > 2:
            msg += ', '
        msg += 'or ' + names[-1]
        raise ValueError("Excepted %s" % msg)

    def completions(self, text, session):
        """completions are the union of alternative annotation completions"""
        completions = []
        for anno in self.annotations:
            completions += anno.completions(text, session)
        return completions


_escape_table = {
    "'": "'",
    '"': '"',
    'a': '\a',
    'b': '\b',
    'f': '\f',
    'n': '\n',
    'r': '\r',
    't': '\t',
    'v': '\v',
}

def unescape(text):
    """Replace backslash escape sequences with actual character
    
    :param text: the input text
    :returns: the processed text

    Follows Python's :ref:`string literal <python:stringescapeseq>` syntax."""
    # standard Python backslashes including \N{unicode name}
    start = 0
    while start < len(text):
        index = text.find('\\', start)
        if index == -1:
            break
        if index == len(text) - 1:
            break
        escaped = text[index + 1]
        if escaped in _escape_table:
            text = text[:index] + _escape_table[escaped] + text[index + 2:]
            start = index + 1
        elif escaped == 'o':
            try:
                char = chr(int(text[index + 2: index + 5], 8))
                text = text[:index] + char + text[index + 5:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'x':
            try:
                char = chr(int(text[index + 2: index + 4], 16))
                text = text[:index] + char + text[index + 4:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'u':
            try:
                char = chr(int(text[index + 2: index + 6], 16))
                text = text[:index] + char + text[index + 6:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'U':
            try:
                char = chr(int(text[index + 2: index + 10], 16))
                text = text[:index] + char + text[index + 10:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'N':
            if len(text) < index + 2 or text[index + 2] != '{':
                start = index + 1
                continue
            end = text.find('}', index + 3)
            if end > 0:
                import unicodedata
                name = text[index + 3:end]
                print('name', name)
                try:
                    char = unicodedata.lookup(name)
                    text = text[:index] + char + text[end + 1:]
                except KeyError:
                    pass
            start = index + 1
        else:
            # leave backslash in text like Python
            start = index + 1
    return text


class RestOfLine(Annotation):
    name = "the rest of line"

    @staticmethod
    def parse(text, session):
        return unescape(text)

    @staticmethod
    def completions(text, session):
        return []

class WholeRestOfLine(Annotation):
    name = "the rest of line"

    @staticmethod
    def parse(text, session):
        return unescape(text)

    @staticmethod
    def completions(text, session):
        return []

Bool3Arg = TupleOf(BoolArg, 3)
IntsArg = ListOf(IntArg)
Int3Arg = TupleOf(IntArg, 3)
FloatsArg = ListOf(FloatArg)
Float3Arg = TupleOf(FloatArg, 3)
PositiveIntArg = Bounded(IntArg, min=1, name="natural number")
ModelIdArg = PositiveIntArg


class Postcondition(metaclass=abc.ABCMeta):
    """Base class for postconditions"""

    def check(self, kw_args):
        """Assert arguments match postcondition

        :param kw_args: dictionary of arguments that will be passed
            to command callback function.
        :returns: True if arguments are consistent
        """
        raise NotImplemented

    def error_message(self):
        """Appropriate error message if check fails.
        
        :returns: error message
        """
        raise NotImplemented


class Limited(Postcondition):
    """Bounded numerical values postcondition

    Limited(name, min=None, max=None) -> Postcondition

    :param name: name of argument to check
    :param min: optional inclusive lower bound
    :param max: optional inclusive upper bound

    If possible, use the Bounded annotation because the location of
    the error is the beginning of the argument, not the end of the line.
    """

    __slots__ = ['name', 'min', 'max']

    def __init__(self, name, min=None, max=None):
        self.name = name
        self.min = min
        self.max = max

    def check(self, kw_args):
        arg = kw_args.get(self.name, None)
        if arg is None:
            # argument not present, must be optional
            return True
        if self.min and self.max:
            return self.min <= arg <= self.max
        elif self.min:
            return arg >= self.min
        elif self.max:
            return arg <= self.max
        else:
            return True

    def error_message(self):
        message = "Invalid argument %r: " % self.name
        if self.min and self.max:
            return message + "Must be greater than or equal to %s and less than or equal to %s" % (self.min, self.max)
        elif self.min:
            return message + "Must be greater than or equal to %s" % self.min
        elif self.max:
            return message + "Must be less than or equal to %s" % self.max

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
        return "%s argument should be the same size as %s argument" % (
            self.name1, self.name2)

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


class CmdDesc:
    """Describe command arguments.

    :param required: required positional arguments tuple
    :param optional: optional positional arguments tuple
    :param keyword: keyword arguments tuple
    :param help: placeholder for help information (e.g., URL)

    .. data: function

        function that implements command

    Each tuple contains tuples with the argument name and a type annotation.
    The command line parser uses the *optional* argument names to as
    keyword arguments.
    """
    __slots__ = [
        '_required', '_optional', '_keyword',
        '_postconditions', '_function',
        'help',
    ]

    def __init__(self, required=(), optional=(), keyword=(),
                 postconditions=(), help=None):
        self._required = OrderedDict(required)
        self._optional = OrderedDict(optional)
        self._keyword = OrderedDict(keyword)
        self._keyword.update(self._optional)
        self._postconditions = postconditions
        self.help = help
        self._function = None

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        """Set the function to call when the command matches.

        Double check that all function arguments, that do not
        have default values, are 'required'.
        """
        if self._function:
            raise ValueError("Can not reuse CmdDesc instances")
        import inspect
        Empty = inspect.Parameter.empty
        Positional = inspect.Parameter.VAR_POSITIONAL
        signature = inspect.signature(function)
        params = list(signature.parameters.values())
        if len(params) < 1 or params[0].name != "session":
            raise ValueError("Missing initial 'session' argument")
        for p in params[1:]:
            if (p.default != Empty or p.name in self._required
                    or p.kind == Positional):
                continue
            raise ValueError("Wrong function or '%s' argument must be "
                             "required or have a default value" % p.name)

        self._function = function

    def copy(self):
        """Return a copy suitable for use with another function."""
        import copy
        ci = copy.copy(self)
        ci._function = None
        return ci


class _Defer:
    # Enable function introspection to be deferred until needed
    __slots__ = ['proxy']

    def __init__(self, proxy_function, cmd_desc):
        self.proxy = proxy_function

    def call(self):
        return self.proxy()


def delay_registration(name, proxy_function):
    """delay registering a named command until needed

    :param proxy_function: the function to call if command is used

    The proxy function should explicitly reregister the command or
    register subcommands and return nothing.

    Example::

        from chimera.core import cli

        def lazy_reg():
            import module
            cli.register('cmd subcmd1', module.subcmd1_desc, module.subcmd1)
            cli.register('cmd subcmd2', module.subcmd2_desc, module.subcmd2)

        cli.delay_registration('cmd', lazy_reg)
    """
    register(name, None, _Defer(proxy_function))


def register(name, cmd_desc=(), function=None):
    """register function that implements command

    :param name: the name of the command and may include spaces.
    :param cmd_desc: information about the command, either an
        instance of :py:class:`CmdDesc`, or the tuple with CmdDesc
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
        def wrapper(function, name=name, cmd_desc=cmd_desc):
            return register(name, cmd_desc, function)
        return wrapper

    if isinstance(cmd_desc, tuple):
        cmd_desc = CmdDesc(*cmd_desc)

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
    #    pass # TODO: replacing, preserve extra keywords?
    _check_autocomplete(word, cmd_map, name)
    if isinstance(function, _Defer):
        # delay introspecting function
        cmd_desc = function
    else:
        # introspect immediately to give errors
        cmd_desc.function = function
    #if word in cmd_map:
    #    print(word, type(function), type(cmd_map[word].function))
    #if (isinstance(function, _Alias) and word in cmd_map
    #        and isinstance(cmd_map[word].function, _Alias)):
    #    raise UserError("Can not alias existing command")
    cmd_map[word] = cmd_desc
    return function     # needed when used as a decorator


def _unregister(name):
    # used internally by unalias
    # none of the exceptions below should happen
    words = name.split()
    cmd_map = _commands
    for word in words[:-1]:
        what = cmd_map.get(word, None)
        if isinstance(what, dict):
            cmd_map = what
            continue
        raise RuntimeError("unregistering unknown multiword command")
    word = words[-1]
    what = cmd_map.get(word, None)
    if what is None:
        raise RuntimeError("unregistering unknown command")
    if isinstance(what, dict):
        raise RuntimeError("unregistering beginning of multiword command")
    del cmd_map[word]

def _lazy_register(cmd_map, word):
    deferred = cmd_map[word]
    try:
        deferred.call()
    except:
        raise RuntimeError("delayed registration failed")
    # deferred function might have registered subcommands
    cmd_desc = cmd_map[word]
    if isinstance(cmd_desc, (dict, CmdDesc)):
        return cmd_desc
    raise RuntimeError("delayed registration didn't register the command")


def add_keyword_arguments(name, kw_info):
    """Make known additional keyword argument(s) for a command

    :param name: the name of the command
    :param kw_info: { keyword: annotation }
    """
    cmd = Command(None, name, final=True)
    if cmd.multiple or not cmd._ci:
        raise ValueError("'%s' is not a command" % name)
    # TODO: fail if there are conflicts with existing keywords?
    cmd._ci._keyword.update(kw_info)
    # TODO: save appropriate kw_info, if reregistered?


class Command:
    """Keep track of partially typed command with possible completions

    :param text: the command text
    :param final: true if text is the complete command line (final version).

    .. data: current_text

        The expanded version of the command.

    .. data: amount_parsed

        Amount of current text that has been successfully parsed.

    .. data: completions

        Possible command completions.  The first one will be used
        if the command is executed.

    .. data: completion_prefix

        Partial word used for command completions.

    .. data: multiple

        True if command is multiple commands (semicolon separated)
    """
    def __init__(self, session, text='', final=False):
        import weakref
        self._session = None if session is None else weakref.proxy(session)
        self._reset()
        if text:
            self.parse_text(text, final)

    def _reset(self):
        self.current_text = ""
        self.amount_parsed = 0
        self.completion_prefix = ""
        self.completions = []
        self.multiple = False
        self._error = "Missing command"
        self._ci = None
        self.command_name = None
        self._kwargs = {}
        self._error_checked = False

    def error_check(self):
        """Error check results of calling parse_text

        :raises UserError: if parsing error is found

        Separate error checking logic from execute() so
        it may be done separately
        """
        if self._error:
            raise UserError(self._error)
        for cond in self._ci._postconditions:
            if not cond.check(self._kwargs):
                raise UserError(cond.error_message())
        self._error_checked = True

    def execute(self):
        """If command is valid, execute it with given session."""

        session = self._session  # resolve back reference
        if not self._error_checked:
            self.error_check()
        try:
            if not isinstance(self._ci.function, _Alias):
                return self._ci.function(session, **self._kwargs)
            arg_names = [k for k in self._kwargs.keys() if isinstance(k, int)]
            arg_names.sort()
            args = [self._kwargs[k] for k in arg_names]
            if 'optional' in self._kwargs:
                optional = self._kwargs['optional']
            else:
                optional = ''
            return self._ci.function(session, *args, optional=optional)
        except ValueError as err:
            if isinstance(self._ci.function, _Alias):
                # propagate expanded alias
                cmd = self._ci.function.cmd
                self.current_text = cmd.current_text
                self.amount_parsed = cmd.amount_parsed
                self._error = cmd._error
            # convert function's ValueErrors to UserErrors,
            # but not those of functions it calls
            import traceback
            _, _, exc_traceback = sys.exc_info()
            if len(traceback.extract_tb(exc_traceback)) > 2:
                raise
            raise UserError(err)
        except UserError as err:
            if isinstance(self._ci.function, _Alias):
                # propagate expanded alias
                cmd = self._ci.function.cmd
                self.current_text = cmd.current_text
                self.amount_parsed = cmd.amount_parsed
                self._error = cmd._error
            raise

    def _next_token(self, text):
        # Return tuple of first argument in text and actual text used
        #
        # Arguments may be quoted, in which case the text between
        # the quotes is returned.  If there is no closing quote,
        # return rest of line for automatic completion purposes,
        # but set an error.
        m = _whitespace.match(text)
        start = m.end() if m else 0
        if start == len(text):
            return '', text
        if text[start] == '"':
            m = _double.match(text, start)
            if m:
                end = m.end()
                token = text[start + 1:end - 1]
                if len(text) > end and not text[end].isspace():
                    self._error = "quoted text must end with whitespace"
                    raise UserError(self._error)
            else:
                end = len(text)
                token = text[start + 1:end]
                self._error = "incomplete quoted text"
            token = unescape(token)
        elif text[start] == "'":
            m = _single.match(text, start)
            if m:
                end = m.end()
                token = text[start + 1:end - 1]
                if len(text) > end and not text[end].isspace():
                    self._error = "quoted text must end with whitespace"
                    raise UserError(self._error)
            else:
                end = len(text)
                token = text[start + 1:end]
                self._error = "incomplete quoted text"
            token = unescape(token)
        elif text[start] in ',;':
            return text[start], text[0:start + 1]
        else:
            m = _normal_token.match(text, start)
            end = m.end()
            token = text[start:end]
        return token, text[0:end]

    def _upto_semicolon(self, text):
        # return text up to next semicolon, taking into account tokens
        start = 0
        size = len(text)
        while start < size:
            m = _whitespace.match(text, start)
            if m:
                start = m.end()
                if start == size:
                    break
            if text[start] == '"':
                m = _double.match(text, start)
                if m:
                    start = m.end()
                else:
                    start = size
                    self._error = "incomplete quoted text"
                    break
            elif text[start] == "'":
                m = _single.match(text, start)
                if m:
                    start = m.end()
                else:
                    start = size
                    self._error = "incomplete quoted text"
                    break
            elif text[start] == ',':
                start += 1
            elif text[start] == ';':
                break
            else:
                m = _normal_token.match(text, start)
                start = m.end()
        return text[:start], text[start:]

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

    def _parse_arg(self, annotation, text, final):
        if annotation is WholeRestOfLine:
            self.amount_parsed += len(text)
            return text.lstrip(), ""

        if annotation is RestOfLine:
            text, rest = self._upto_semicolon(text)
            self.amount_parsed += len(text)
            return text.lstrip(), rest

        session = self._session  # resolve back reference
        multiword = annotation.multiword
        all_words = []
        all_chars = []
        count = 0
        while 1:
            count += 1
            if count > 1024:
                # TODO: change test to see if still consuming characters
                # 1024 is greater than the number of words
                # in a multiword argument (think atomspecs)
                raise RuntimeError("Invalid completions given by %s"
                                   % annotation.name)
            word, chars = self._next_token(text)
            if _debugging:
                print('arg _next_token(%r) -> %r %r' % (text, word, chars))
            if not word:
                raise ValueError("Expected %s" % annotation.name)
            all_words.append(word)
            word = ' '.join(all_words)
            all_chars.append(chars)
            #chars = ''.join(all_chars) need short version in exception
            try:
                value = annotation.parse(word, session)
                break
            except ValueError as err:
                completions = annotation.completions(word, session)
                if completions and (final or len(text) > len(chars)):
                    c = completions[0]
                    if not c.startswith(word) or len(c) <= len(word):
                        raise RuntimeError("Invalid completions given by %s"
                                           % annotation.name)
                    c = c[len(word):]
                    if multiword:
                        if c[0].isspace():
                            text = text[len(chars):]
                            continue
                        c = c.split(None, 1)[0]
                    chars = ''.join(all_chars)
                    text = self._complete(chars, c)
                    del all_words[-1]
                    del all_chars[-1]
                    chars = ''.join(all_chars)
                    text = text[len(chars):]
                    continue
                self._error = err
                if multiword and not completions:
                    # try shorter version
                    del all_words[-1]
                    word = ' '.join(all_words)
                    completions = annotation.completions(word, session)
                self.completion_prefix = word
                self.completions = completions
                raise
        chars = ''.join(all_chars)
        self.amount_parsed += len(chars)
        text = self.current_text[self.amount_parsed:]
        return value, text

    def _parse_aggregate(self, anno, text, final):
        # expect VALUE [, VALUE]*
        self._error = ""
        self.completion_prefix = ""
        self.completions = []
        values = anno.constructor()
        value, text = self._parse_arg(anno, text, final)
        x = anno.add_to(values, value)
        if x is not None:
            values = x
        while 1:
            word, chars = self._next_token(text)
            if _debugging:
                print('agg _next_token(%r) -> %r %r' % (text, word, chars))
            if word != ',':
                if len(values) < anno.min_size:
                    if anno.min_size == anno.max_size:
                        qual = "exactly"
                    else:
                        qual = "at least"
                    raise ValueError("Need %s %d %s" % (qual, anno.min_size,
                                                        anno.name))
                if len(values) > anno.max_size:
                    if anno.min_size == anno.max_size:
                        qual = "exactly"
                    else:
                        qual = "at most"
                    raise ValueError("Need %s %d %s" % (qual, anno.max_size,
                                     anno.name))
                return values, text
            self.amount_parsed += len(chars)
            text = text[len(chars):]
            value, text = self._parse_arg(anno, text, final)
            x = anno.add_to(values, value)
            if x is not None:
                values = x

    def _find_command_name(self, final):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, sets self._ci
        start = self.amount_parsed
        text = self.current_text[start:]
        m = _whitespace.match(text, start)
        if m:
            start = m.end()
        word_map = _commands
        while 1:
            word, chars = self._next_token(text)
            if _debugging:
                print('cmd _next_token(%r) -> %r %r' % (text, word, chars))
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
                what = _lazy_register(word_map, word)
            if isinstance(what, dict):
                # word is part of multiword command name
                word_map = what
                self._error = ("Incomplete command: %s"
                               % self.current_text[start:self.amount_parsed])
                continue
            assert(isinstance(what, CmdDesc))
            self._ci = what
            self.command_name = self.current_text[start:self.amount_parsed]
            return

    def _process_positional_arguments(self):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, updates self._kwargs
        text = self.current_text[self.amount_parsed:]
        positional = self._ci._required.copy()
        positional.update(self._ci._optional)
        self.completion_prefix = ''
        self.completions = []
        for name, anno in positional.items():
            if name in self._ci._optional:
                self._error = ""
                tmp = text.split(None, 1)
                if not tmp:
                    break
                if tmp[0] in self._ci._keyword:
                    # matches keyword,
                    # so switch to keyword processing
                    break
            else:
                self._error = "Missing required argument %s" % name
            try:
                if isinstance(anno, Aggregate):
                    value, text = self._parse_aggregate(anno, text, False)
                else:
                    value, text = self._parse_arg(anno, text, False)
                self._kwargs[name] = value
            except ValueError as err:
                if name in self._ci._required:
                    self._error = "Invalid argument %r: %s" % (name, err)
                    return
                # optional and wrong type, try as keyword
                break
        self._error = ""

    def _process_keyword_arguments(self, final):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, updates self._kwargs
        text = self.current_text[self.amount_parsed:]
        while 1:
            word, chars = self._next_token(text)
            if _debugging:
                print('key _next_token(%r) -> %r %r' % (text, word, chars))
            if not word:
                # count extra whitespace as parsed
                self.amount_parsed += len(chars)
                break

            if word not in self._ci._keyword:
                self.completion_prefix = word
                self.completions = [x for x in self._ci._keyword
                                    if x.startswith(word)]
                if (final or len(text) > len(chars)) and self.completions:
                    # If final version of text, or if there
                    # is following text, make best guess,
                    # and retry
                    c = self.completions[0]
                    text = self._complete(chars, c[len(word):])
                    self.completions = []
                    continue
                if len(self._ci._keyword) > 0:
                    self._error = "Expected keyword, got '%s'" % word
                else:
                    self._error = "Too many arguments"
                return
            self.amount_parsed += len(chars)
            text = text[len(chars):]

            name = word
            anno = self._ci._keyword[name]
            try:
                if isinstance(anno, Aggregate):
                    value, text = self._parse_aggregate(anno, text, final)
                else:
                    value, text = self._parse_arg(anno, text, final)
                self._kwargs[name] = value
            except ValueError as err:
                self._error = "Invalid  argument %r: %s" % (name, err)
                return

    def parse_text(self, text, final=False):
        """Parse text into function and arguments

        :param text: The text to be parsed.
        :param final: True if last version of command text

        May be called multiple times.  There are a couple side effects:

        * The automatically completed text is put in self.current_text.
        * Possible completions are in self.completions.
        * The prefix of the completions is in self.completion_prefix.
        """
        self._reset()   # don't be smart, just start over

        # find command name
        self.current_text = text

        #while 1:
        if 1:
            self._find_command_name(final)
            if not self._ci:
                return
            self._process_positional_arguments()
            if self._error:
                return
            self._process_keyword_arguments(final)
            if self._error:
                return
            if self.amount_parsed == len(self.current_text):
                return
            #self.multiple = 1


def command_function(name):
    """Return callable for given command name

    :param name: the name of the command
    :returns: the callable that implements the command
    """
    cmd = Command(None, name, final=True)
    if cmd.multiple or not cmd._ci:
        raise ValueError("'%s' is not a (single) command" % name)
    return cmd._ci.function


def command_help(name):
    """Return help for given command name

    :param name: the name of the command
    :returns: the help object registered with the command
    """
    cmd = Command(None, name, final=True)
    if cmd.multiple or not cmd._ci:
        raise ValueError("'%s' is not a (single) command" % name)
    return cmd._ci.help


def command_usage(name):
    """Return usage string for given command name

    :param name: the name of the command
    :returns: a usage string for the command
    """
    cmd = Command(None, name, final=True)
    if cmd.multiple or not cmd._ci:
        raise ValueError("'%s' is not a (single) command" % name)
    usage = cmd.command_name
    ci = cmd._ci
    for arg in ci._required:
        usage += ' %s' % arg
    num_opt = 0
    for arg in ci._optional:
        usage += ' [%s' % arg
        num_opt += 1
    usage += ']' * num_opt
    for arg in ci._keyword:
        type = ci._keyword[arg].name
        usage += ' [%s _%s_]' % (arg, type)
    return usage


def command_html_usage(name):
    """Return usage string in HTML for given command name

    :param name: the name of the command
    :returns: a HTML usage string for the command
    """
    cmd = Command(None, name, final=True)
    if cmd.multiple or not cmd._ci:
        raise ValueError("'%s' is not a (single) command" % name)
    from html import escape
    usage = '<b>%s</b>' % escape(cmd.command_name)
    ci = cmd._ci
    for arg in ci._required:
        type = ci._required[arg].name
        usage += ' <span title="%s"><i>%s</i></span>' % (escape(type),
                                                         escape(arg))
    num_opt = 0
    for arg in ci._optional:
        num_opt += 1
        type = ci._optional[arg].name
        usage += ' [<span title="%s"><i>%s</i></span>' % (escape(type),
                                                          escape(arg))
    usage += ']' * num_opt
    for arg in ci._keyword:
        type = ci._keyword[arg].name
        usage += ' [<b>%s</b> <i>%s</i>]' % (escape(arg), escape(type))
    return usage


_cmd_aliases = OrderedDict()


class _Alias:
    """Internal alias command implementation"""

    def __init__(self, text):
        text = text.lstrip()
        self.original_text = text
        self.num_args = 0
        self.parts = []  # list of strings and integer argument numbers
        self.cmd = None
        self.optional_rest_of_line = False

        not_dollar = re.compile(r"[^$]*")
        number = re.compile(r"\d*")

        start = 0
        while True:
            m = not_dollar.match(text, start)
            end = m.end()
            if end > start:
                self.parts.append(text[start:end])
                start = end
            if start == len(text):
                break
            start += 1  # skip over $
            if start < len(text) and text[start] == '$':
                self.parts.append('$')   # $$
                start += 1
                continue
            if start < len(text) and text[start] == '*':
                self.optional_rest_of_line = True
                self.parts.append(-1)
                start += 1
                continue
            m = number.match(text, start)
            end = m.end()
            if end == start:
                # not followed by a number
                self.parts.append('$')
                continue
            i = int(text[start:end])
            if i > self.num_args:
                self.num_args = i
            self.parts.append(i - 1)     # convert to a 0-based index
            start = end

    def desc(self):
        required = [((i + 1), StringArg) for i in range(self.num_args)]
        if not self.optional_rest_of_line:
            return CmdDesc(required=required)
        return CmdDesc(required=required, optional=[('optional', RestOfLine)])

    def __call__(self, session, *args, optional=''):
        assert(len(args) >= self.num_args)
        # substitute args for positional arguments
        text = ''
        for part in self.parts:
            if isinstance(part, str):
                text += part
                continue
            # part is an argument index
            if part < 0:
                text += optional
            else:
                text += args[part]
        self.cmd = Command(session, text, final=True)
        return self.cmd.execute()


@register('alias', CmdDesc(optional=[('name', StringArg),
                                     ('text', WholeRestOfLine)]))
def alias(session, name='', text=''):
    """Create command alias

    :param name: optional name of the alias
    :param text: optional text of the alias

    If the alias name is not given, then a text list of all the aliases is
    returned.  If alias text is not given, the the text of the named alias
    is returned.  If both arguments are given, then a new alias is made.
    """
    if not name:
        # list aliases
        names = ', '.join(list(_cmd_aliases.keys()))
        if names:
            return 'Aliases: %s' % names
        return 'No aliases.'
    if not text:
        if name not in _cmd_aliases:
            return 'No alias named %r found.' % name
        return 'Aliased %r to %r' % (name, _cmd_aliases[name].original_text)
    cmd = _Alias(text)
    try:
        register(name, cmd.desc(), cmd)
    except:
        raise


@register('~alias', CmdDesc(required=[('name', StringArg)]))
def unalias(session, name):
    """Remove command alias

    :param name: name of the alias
    """
    # remove command alias
    try:
        del _cmd_aliases[name]
    except KeyError:
        raise UserError('No alias named %r exists' % name)
    _unregister(name)

if __name__ == '__main__':

    class ColorArg(Annotation):
        multiword = True
        name = 'a color'

        Builtin_Colors = {
            "light gray": (211, 211, 211),
            "red": (255, 0, 0)
        }

        @staticmethod
        def parse(text, session):
            text = text.casefold()
            try:
                color = ColorArg.Builtin_Colors[text]
            except KeyError:
                raise ValueError("Invalid color name")
            return ([x / 255.0 for x in color])

        @staticmethod
        def completions(text, session):
            text = text.casefold()
            names = [n for n in ColorArg.Builtin_Colors if n.startswith(text)]
            return names

    test1_desc = CmdDesc(
        required=[('a', IntArg), ('b', FloatArg)],
        keyword=[('color', ColorArg)]
    )

    @register('test1', test1_desc)
    def test1(session, a, b, color=None):
        print('test1 a: %s %s' % (type(a), a))
        print('test1 b: %s %s' % (type(b), b))
        print('test1 color: %s %s' % (type(color), color))

    test2_desc = CmdDesc(
        #required=[('a', StringArg)],
        #optional=[('text', RestOfLine)],
        keyword=[('color', ColorArg), ('radius', FloatArg)]
    )

    @register('test2', test2_desc)
    def test2(session, a='a', text='', color=None, radius=0):
        #print('test2 a: %s %s' % (type(a), a))
        #print('test2 text: %s %s' % (type(text), text))
        print('test2 color: %s %s' % (type(color), color))
        print('test2 radius: %s %s' % (type(radius), radius))

    register('mw test1', test1_desc.copy(), test1)
    register('mw test2', test2_desc.copy(), test2)

    test3_desc = CmdDesc(
        required=[('name', StringArg)],
        optional=[('value', FloatArg)]
    )

    @register('test3', test3_desc)
    def test3(session, name, value=None):
        print('test3 name: %s %s' % (type(name), name))
        print('test3 value: %s %s' % (type(value), value))

    test4_desc = CmdDesc(
        optional=[('draw', PositiveIntArg)]
    )

    @register('test4', test4_desc)
    def test4(session, draw=None):
        print('test4 draw: %s %s' % (type(draw), draw))

    test4b_desc = CmdDesc(
        optional=[('draw', IntArg)],
        postconditions=[Limited('draw', min=1)]
    )

    @register('test4b', test4b_desc)
    def test4b(session, draw=None):
        print('test4b draw: %s %s' % (type(draw), draw))

    test5_desc = CmdDesc(
        optional=[('ints', IntsArg)]
    )

    @register('test5', test5_desc)
    def test5(session, ints=None):
        print('test5 ints: %s %s' % (type(ints), ints))

    test6_desc = CmdDesc(
        required=[('center', Float3Arg)]
    )

    @register('test6', test6_desc)
    def test6(session, center):
        print('test6 center:', center)

    test7_desc = CmdDesc(
        optional=[('center', Float3Arg)]
    )

    @register('test7', test7_desc)
    def test7(session, center=None):
        print('test7 center:', center)

    test8_desc = CmdDesc(
        optional=[
            ('always', BoolArg),
            ('target', StringArg),
            ('names', ListOf(StringArg)),
        ],
    )

    @register('test8', test8_desc)
    def test8(session, always=True, target="all", names=[None]):
        print('test8 always, target, names:', always, target, names)

    test9_desc = CmdDesc(
        optional=(
            ("target", StringArg),
            ("names", ListOf(StringArg))
        ),
        keyword=(("full", BoolArg),)
    )

    @register('test9', test9_desc)
    def test9(session, target="all", names=[None], full=False):
        print('test9 full, target, names: %r, %r, %r' % (full, target, names))

    test10_desc = CmdDesc(
        required=(
            ("colors", ListOf(ColorArg)),
            ("offsets", ListOf(FloatArg)),
        ),
        postconditions=(SameSize('colors', 'offsets'),)
    )

    @register('test10', test10_desc)
    def test10(session, colors=[], offsets=[]):
        print('test10 colors, offsets:', colors, offsets)

    if len(sys.argv) > 1:
        _debugging = 'd' in sys.argv[1]
        @register('exit')
        def exit(session):
            raise SystemExit(0)
        @register('echo', CmdDesc(optional=[('text', RestOfLine)]))
        def echo(session, text=''):
            return text
        @register('usage', CmdDesc(required=[('name', RestOfLine)]))
        def usage(session, name):
            print(command_usage(name))
        @register('html_usage', CmdDesc(required=[('name', RestOfLine)]))
        def html_usage(session, name):
            print(command_html_usage(name))
        prompt = 'cmd> '
        cmd = Command(None)
        while True:
            try:
                text = input(prompt)
                cmd.parse_text(text, final=True)
                result = cmd.execute()
                if result is not None:
                    print(result)
            except EOFError:
                raise SystemExit(0)
            except UserError as err:
                print(cmd.current_text)
                rest = cmd.current_text[cmd.amount_parsed:]
                spaces = len(rest) - len(rest.lstrip())
                error_at = cmd.amount_parsed + spaces
                print("%s^" % ('.' * error_at))
                print(err)

    tests = [   # (fail, final, command)
        (True, True, 'test1 color red 12 3.5'),
        (True, True, 'test1 12 color red 3.5'),
        (False, True, 'test1 12 3.5 color red'),
        (True, True, 'test1 12 3.5 color'),
        (True, True, 'te'),
        (True, True, 'test2 color red radius 3.5 foo'),
        (False, True, 'test2 color red radius 3.5'),
        (True, True, 'test2 color red radius xyzzy'),
        (True, True, 'test2 color red radius'),
        (False, True, 'test2 color "light gray"'),
        (False, True, 'test2 color light gray'),
        (False, True, 'test2 color li gr'),
        (False, True, 'test2 co li gr rad 11'),
        (True, True, 'test2 c'),
        (False, True, 'test3 radius'),
        (False, True, 'test3 radius 12.3'),
        (False, True, 'test4'),
        (True, True, 'test4 draw'),
        (False, True, 'test4 draw 3'),
        (True, False, 'test4 draw -34'),
        (True, False, 'test4b draw -34'),
        (False, True, 'test5'),
        (False, True, 'test5 ints 55'),
        (False, True, 'test5 ints 5 ints 6'),
        (False, True, 'test5 ints 5, 6, -7, 8, 9'),
        (True, True, 'mw test1 color red 12 3.5'),
        (True, True, 'mw test1 color red 12 3.5'),
        (True, True, 'mw test2 color red radius 3.5 foo'),
        (True, False, 'mw te'),
        (True, True, 'mw '),
        (True, False, 'mw'),
        (False, True, 'te 12 3.5 co red'),
        (False, True, 'm te 12 3.5 col red'),
        (False, True, 'test6 3.4, 5.6, 7.8'),
        (True, True, 'test6 3.4 abc 7.8'),
        (False, True, 'test7 center 3.4, 5.6, 7.8'),
        (True, True, 'test7 center 3.4, 5.6'),
        (False, True, 'test8 always false'),
        (False, True, 'test8 always  Tr target tool'),
        (True, True, 'test8 always true tool'),
        (True, True, 'test8 always tool'),
        (False, True, 'test8 TRUE tool xyzzy, plugh '),
        (False, True, 'test9 full true'),
        (True, True, 'test9 names a,b,c d'),
        (True, True, 'test10'),
        (True, False, 'test10 red'),
        (False, True, 'test10 red 0.5'),
        (True, False, 'test10 red, light gray'),
        (False, False, 'test10 light  gray, red 0.33, 0.67'),
        (True, False, 'test10 light  gray, red 0.33'),
        (True, False, 'test10 li  gr, red'),
        (True, False, 'test10 "red"10.3'),
    ]
    # TODO: delayed registration tests
    successes = 0
    failures = 0
    cmd = Command(None)
    for t in tests:
        fail, final, text = t
        try:
            print("\nTEST: '%s'" % text)
            cmd.parse_text(text, final=final)
            print(cmd.current_text)
            #print(cmd.current_text, cmd._kwargs)
            result = cmd.execute()
            if result:
                print(result)
            if fail:
                # expected failure and it didn't
                failures += 1
                print('FAIL')
            else:
                successes += 1
                print('SUCCESS')
        except UserError as err:
            rest = cmd.current_text[cmd.amount_parsed:]
            spaces = len(rest) - len(rest.lstrip())
            error_at = cmd.amount_parsed + spaces
            #parsed = cmd.current_text[:cmd.amount_parsed]
            print("%s^" % ('.' * error_at))
            if fail:
                successes += 1
                print('SUCCESS:', err)
            else:
                failures += 1
                print('FAIL:', err)
            p = cmd.completions
            if p:
                print('completions:', p)
    print(successes, 'success(es),', failures, 'failure(s)')
    if failures:
        raise SystemExit(1)
    raise SystemExit(0)
