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
so possible commands can be suggested.

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

Words in the command name may be truncated
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

Annotations are used to parse text and convert it to the appropriate type.
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
and (2) to give reasonable error messages.

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
"""

import abc
from keyword import iskeyword
import re
import sys
from collections import OrderedDict

_debugging = False
_normal_token = re.compile(r"[^;\s]*")
_single_quote = re.compile(r"'(.|\')*?'(\s|$)")
_double_quote = re.compile(r'"(.|\")*?"(\s|$)')
_whitespace = re.compile("\s*")


class UserError(ValueError):
    """An exception provoked by the user's input.

    UserError(object) -> a UserError object

    This is in contrast to a error is a bug in the program.
    """
    pass


class AnnotationError(UserError):
    """Error, with optional offset, in annotation"""

    def __init__(self, message, offset=None):
        ValueError.__init__(self, message)
        self.offset = offset


class Annotation(metaclass=abc.ABCMeta):
    """Base class for all annotations

    Each annotation should have the following attributes:

    .. py:attribute:: name

        Set to textual description of the annotation, including
        the leading article, *e.g.*, `"a truth value"`.
    """
    name = "** article name, e.g., _a_ _truth value_ **"
    help = None  #: placeholder for help information (e.g., URL)

    @staticmethod
    def parse(text, session):
        """Convert text to appropriate type.

        :param text: command line text to parse
        :param session: for session-dependent data types
        :returns: 3-tuple with the converted value, consumed text
            (possibly altered with expanded truncations), and the
            remaining unconsumed text
        :raises ValueError: if unable to convert text

        The leading space in text must already be removed.
        It is up to the particular annotation to support truncatations.
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

    This class is typically used via :py:class:`ListOf`, :py:class:`SetOf`,
    or :py:class:`TupleOf`.
    The comma separator for aggregate values is handled by the
    :py:class:`Command` class, so parsing is delegated to the
    underlying annotation.

    Subclasses need to set the constructor attribute and replace
    the add_to method.
    """
    min_size = 0
    max_size = sys.maxsize
    constructor = None
    separator = ','

    def __init__(self, annotation, min_size=None,
                 max_size=None, name=None):
        if (not issubclass(annotation, Annotation)
                and not isinstance(annotation, Annotation)):
            raise ValueError("need an annotation, not %s" % annotation)
        self.annotation = annotation
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
        result = self.constructor()
        used = ''
        while 1:
            i = text.find(self.separator)
            if i == -1:
                # no separator found
                try:
                    value, consumed, rest = self.annotation.parse(text,
                                                                  session)
                except AnnotationError as err:
                    if err.offset is None:
                        err.offset = len(used)
                    else:
                        err.offset += len(used)
                    raise
                tmp = self.add_to(result, value)
                if tmp:
                    result = tmp
                used += consumed
                break
            examine = text[:i]
            try:
                value, consumed, rest = self.annotation.parse(examine, session)
            except AnnotationError as err:
                if err.offset is None:
                    err.offset = len(used)
                else:
                    err.offset += len(used)
                raise
            tmp = self.add_to(result, value)
            if tmp:
                result = tmp
            used += consumed
            if len(rest) > 0:
                # later separator found, but didn't consume all of the text,
                # so probably a different argument, unless the rest is
                # all whitespace
                m = _whitespace.match(rest)
                if m.end() == len(rest):
                    used += rest
                    rest = ''
                if len(rest) > 0:
                    rest += text[i:]
                    break
            used += self.separator
            text = text[i + 1:]
            m = _whitespace.match(text)
            i = m.end()
            if i:
                used += text[:i]
                text = text[i:]
        if len(result) < self.min_size:
            if self.min_size == self.max_size:
                qual = "exactly"
            else:
                qual = "at least"
            raise AnnotationError("Need %s %d %s" % (qual, self.min_size,
                                                     self.name), len(used))
        if len(result) > self.max_size:
            if self.min_size == self.max_size:
                qual = "exactly"
            else:
                qual = "at most"
            raise AnnotationError("Need %s %d %s" % (qual, self.max_size,
                                                     self.name), len(used))
        return result, used, rest


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


class DottedTupleOf(Aggregate):
    """Annotation for dot-separated lists of a single type

    DottedListOf(annotation, min_size=None, max_size=None) -> annotation
    """
    separator = '.'
    constructor = tuple

    def __init__(self, annotation, min_size=None,
                 max_size=None, name=None):
        Aggregate.__init__(self, annotation, min_size, max_size, name)
        if name is None:
            if ',' in annotation.name:
                name = "dotted list of %s" % annotation.name
            else:
                # discard article
                name = annotation.name.split(None, 1)[1]
                name = "dotted %s(s)" % name
            self.name = name

    def add_to(self, container, value):
        return container + (value,)


class BoolArg(Annotation):
    """Annotation for boolean literals"""
    name = "a truth value"

    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        token = token.casefold()
        if token == "0" or "false".startswith(token):
            return False, "false", rest
        if token == "1" or "true".startswith(token):
            return True, "true", rest
        raise AnnotationError("Expected true or false (or 1 or 0)")


class IntArg(Annotation):
    """Annotation for integer literals"""
    name = "a whole number"

    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            return int(token), text, rest
        except ValueError:
            raise AnnotationError("Expected %s" % IntArg.name)


class FloatArg(Annotation):
    """Annotation for floating point literals"""
    name = "a floating point number"

    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        try:
            return float(token), text, rest
        except ValueError:
            raise AnnotationError("Expected %s" % FloatArg.name)


class StringArg(Annotation):
    """Annotation for text (a word or quoted)"""
    name = "a text string"

    @staticmethod
    def parse(text, session):
        token, text, rest = next_token(text)
        return token, text, rest


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
        value, new_text, rest = self.anno.parse(text, session)
        if self.min is not None and value < self.min:
            raise AnnotationError("Must be greater than or equal to %s"
                                  % self.min, len(text) - len(rest))
        if self.max is not None and value > self.max:
            raise AnnotationError("Must be less than or equal to %s"
                                  % self.max, len(text) - len(rest))
        return value, new_text, rest


class EnumOf(Annotation):
    """Support enumerated types

    EnumOf(values, ids=None, name=None) -> an Annotation

    :param values: sequence of values
    :param ids: optional sequence of identifiers
    :param name: optional explicit name for annotation

    .. data: allow_truncated

        (Defaults to True.)  If true, then recognize truncated ids.

    If the *ids* are given, then there must be one for each
    and every value, otherwise the values are used as the identifiers.
    The identifiers must all be strings.
    """

    allow_truncated = True

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
        if name is None:
            name = "one of '%s', or '%s'" % ("', '".join(self.ids[0:-1]),
                                             self.ids[-1])
        self.name = name

    def parse(self, text, session):
        token, text, rest = next_token(text)
        folded = token.casefold()
        for i, ident in enumerate(self.ids):
            if self.allow_truncated:
                if ident.casefold().startswith(folded):
                    return self.values[i], ident, rest
            else:
                if ident.casefold() == folded:
                    return self.values[i], ident, rest
        raise AnnotationError("Invalid %s" % self.name)


class Or(Annotation):
    """Support two or more alternative annotations

    Or(annotation, annotation [, annotation]*, name=None) -> an Annotation

    :param name: optional explicit name for annotation
    """

    def __init__(self, *annotations, name=None):
        if len(annotations) < 2:
            raise ValueError("Need at two alternative annotations")
        self.annotations = annotations
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
        raise AnnotationError("Excepted %s" % msg)


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
    """Replace backslash escape sequences with actual character.

    :param text: the input text
    :returns: the processed text

    Follows Python's :ref:`string literal <python:literals>` syntax
    for escape sequences."""
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


def next_token(text):
    """
    Extract next token from given text.

    :param text: text to parse without leading whitespace
    :returns: a 3-tuple of first argument in text, the actual text used,
              and the rest of the text.


    Tokens may be quoted, in which case the text between
    the quotes is returned.
    """
    assert(text and not text[0].isspace())
    # m = _whitespace.match(text)
    # start = m.end()
    # if start == len(text):
    #     return '', text, ''
    start = 0
    if text[start] == '"':
        m = _double_quote.match(text, start)
        if m:
            end = m.end()
            if text[end - 1].isspace():
                end -= 1
            token = text[start + 1:end - 1]
        else:
            end = len(text)
            token = text[start + 1:end]
            raise AnnotationError("incomplete quoted text")
        token = unescape(token)
    # elif text[start] == "'":
    #     m = _single_quote.match(text, start)
    #     if m:
    #         end = m.end()
    #         if text[end - 1].isspace():
    #             end -= 1
    #         token = text[start + 1:end - 1]
    #     else:
    #         end = len(text)
    #         token = text[start + 1:end]
    #         raise AnnotationError("incomplete quoted text")
    #     token = unescape(token)
    elif text[start] == ';':
        return ';', ';', text[start + 1:]
    else:
        m = _normal_token.match(text, start)
        end = m.end()
        token = text[start:end]
    return token, text[:end], text[end:]
_next_token = next_token  # backward compatibility


def _upto_semicolon(text):
    # return text up to next semicolon, taking into account tokens
    start = 0
    size = len(text)
    while start < size:
        m = _whitespace.match(text, start)
        start = m.end()
        if start == size:
            break
        if text[start] == '"':
            m = _double_quote.match(text, start)
            if m:
                start = m.end()
            else:
                start = size
                raise AnnotationError("incomplete quoted text")
                break
        # elif text[start] == "'":
        #     m = _single_quote.match(text, start)
        #     if m:
        #         start = m.end()
        #     else:
        #         start = size
        #         raise AnnotationError("incomplete quoted text")
        #         break
        elif text[start] == ';':
            break
        else:
            m = _normal_token.match(text, start)
            start = m.end()
    return text[:start], text[start:]


class RestOfLine(Annotation):
    name = "the rest of line"

    @staticmethod
    def parse(text, session):
        m = _whitespace.match(text)
        start = m.end()
        leading = text[:start]
        text, rest = _upto_semicolon(text[start:])
        return unescape(text), leading + text, rest


class WholeRestOfLine(Annotation):
    name = "the rest of line"

    @staticmethod
    def parse(text, session):
        return unescape(text), text, ''

Bool3Arg = TupleOf(BoolArg, 3)
IntsArg = ListOf(IntArg)
Int3Arg = TupleOf(IntArg, 3)
FloatsArg = ListOf(FloatArg)
Float3Arg = TupleOf(FloatArg, 3)
PositiveIntArg = Bounded(IntArg, min=1, name="a natural number")
ModelIdArg = DottedTupleOf(PositiveIntArg, name="a model id")


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
            return message + ("Must be greater than or equal to %s and less"
                              " than or equal to %s" % (self.min, self.max))
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
    # This is a primary debugging aid for developers,
    # but it prevents existing abbreviated commands from changing
    # what command they correspond to.
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
        empty = inspect.Parameter.empty
        var_positional = inspect.Parameter.VAR_POSITIONAL
        var_keyword = inspect.Parameter.VAR_KEYWORD
        signature = inspect.signature(function)
        params = list(signature.parameters.values())
        if len(params) < 1 or params[0].name != "session":
            raise ValueError("Missing initial 'session' argument")
        for p in params[1:]:
            if (p.default != empty or p.name in self._required
                    or p.kind in (var_positional, var_keyword)):
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

    def __init__(self, proxy_function):
        self.proxy = proxy_function

    def call(self):
        return self.proxy()


def delay_registration(name, proxy_function, logger=None):
    """delay registering a named command until needed

    :param proxy_function: the function to call if command is used
    :param logger: optional logger

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
    register(name, None, _Defer(proxy_function), logger=logger)


# keep track of commands that have been overridden by an alias
_aliased_commands = {}


def register(name, cmd_desc=(), function=None, logger=None):
    """register function that implements command

    :param name: the name of the command and may include spaces.
    :param cmd_desc: information about the command, either an
        instance of :py:class:`CmdDesc`, or the tuple with CmdDesc
        parameters.
    :param function: the callback function.
    :param logger: optional logger

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
            return register(name, cmd_desc, function, logger=logger)
        return wrapper

    if isinstance(cmd_desc, tuple):
        cmd_desc = CmdDesc(*cmd_desc)

    words = name.split()
    name = ' '.join(words)  # canonicalize
    cmd_map = _commands
    for word in words[:-1]:
        what = cmd_map.get(word, None)
        if isinstance(what, _Defer):
            what = _lazy_register(cmd_map, word)
        if isinstance(what, dict):
            cmd_map = what
            continue
        if what is not None:
            raise ValueError("Can't mix subcommands with no subcommands")
        _check_autocomplete(word, cmd_map, name)
        d = cmd_map[word] = OrderedDict()
        cmd_map = d
    word = words[-1]
    what = cmd_map.get(word, None)
    if isinstance(what, dict):
        raise ValueError("Command is part of multiword command")
    try:
        _check_autocomplete(word, cmd_map, name)
    except ValueError:
        if not isinstance(function, _Alias):
            raise
        if logger is not None:
            logger.warn("alias %r hides existing command" % name)
    if isinstance(function, _Defer):
        cmd_desc = function
    else:
        cmd_desc.function = function
    if what is None:
        cmd_map[word] = cmd_desc
    else:
        # command already registered
        if isinstance(function, _Alias):
            if not isinstance(cmd_map[word].function, _Alias):
                # only save nonaliased version of command
                _aliased_commands[name] = cmd_map[word]
            cmd_map[word] = cmd_desc
        elif isinstance(what.function, _Alias):
            # command is aliased, but new one isn't, so replaced saved version
            _aliased_commands[name] = cmd_desc
        else:
            if logger is not None:
                logger.warn("command %r is replacing existing command" % name)
            cmd_map[word] = cmd_desc
    return function     # needed when used as a decorator


def deregister(name):
    """Remove existing command

    If the command was an alias, the previous version is restored"""
    # none of the exceptions below should happen
    words = name.split()
    name = ' '.join(words)  # canonicalize
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
    hidden_cmd = _aliased_commands.get(name, None)
    if hidden_cmd:
        what = _aliased_commands[name]
        del _aliased_commands[name]
    else:
        del cmd_map[word]

    # remove any subcommand aliases
    size = len(name)
    aliases = [a for a in _cmd_aliases
               if a.startswith(name) and len(name) > size and a[size] == ' ']
    for a in aliases:
        del _cmd_aliases[a]
        if a in _aliased_commands:
            del _aliased_commands[a]

    # allow commands to be reregistered with same description
    def clear_cmd_desc(d):
        if isinstance(d, CmdDesc):
            d._function = None
            return
        if not isinstance(d, dict):
            return
        for v in d.values():
            clear_cmd_desc(v)
    clear_cmd_desc(what)


def _lazy_register(cmd_map, word):
    deferred = cmd_map[word]
    del cmd_map[word]   # prevent recursion
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
    cmd.current_text = name
    cmd._find_command_name(True)
    if not cmd._ci or cmd.amount_parsed != len(cmd.current_text):
        raise ValueError("'%s' is not a command name" % name)
    # TODO: fail if there are conflicts with existing keywords?
    cmd._ci._keyword.update(kw_info)
    # TODO: save appropriate kw_info, if reregistered?


class Command:
    """Keep track of (partially) typed command with possible completions

    :param text: the command text
    :param final: true if text is the complete command line (final version).

    .. data: current_text

        The expanded version of the command.

    .. data: amount_parsed

        Amount of current text that has been successfully parsed.

    .. data: completions

        Possible command or keyword completions if given an incomplete command.
        The first one will be used if the command is executed.

    .. data: completion_prefix

        Partial word used for command completions.
    """
    def __init__(self, session, text='', final=False, _used_aliases=None):
        import weakref
        if session is None:
            class FakeSession:
                pass
            session = FakeSession()
        self._session = weakref.ref(session)
        self._reset()
        if text:
            self.parse_text(text, final, _used_aliases)

    def _reset(self):
        self.current_text = ""
        self.amount_parsed = 0
        self.completion_prefix = ""
        self.completions = []
        self._multiple = []
        self._error = ""
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
        for (cmd_name, ci, kwargs) in self._multiple:
            for cond in ci._postconditions:
                if not cond.check(kwargs):
                    raise UserError(cond.error_message())
        self._error_checked = True

    def execute(self, _used_aliases=None):
        """If command is valid, execute it with given session."""

        session = self._session()  # resolve back reference
        if not self._error_checked:
            self.error_check()
        results = []
        for (cmd_name, ci, kwargs) in self._multiple:
            try:
                if not isinstance(ci.function, _Alias):
                    results.append(ci.function(session, **kwargs))
                    continue
                arg_names = [k for k in kwargs.keys() if isinstance(k, int)]
                arg_names.sort()
                args = [kwargs[k] for k in arg_names]
                if 'optional' in kwargs:
                    optional = kwargs['optional']
                else:
                    optional = ''
                if _used_aliases is None:
                    _used_aliases = {cmd_name}
                results.append(ci.function(session, *args, optional=optional,
                               used_aliases=_used_aliases))
                continue
            except UserError as err:
                if isinstance(ci.function, _Alias):
                    # propagate expanded alias
                    cmd = ci.function.cmd
                    self.current_text = cmd.current_text
                    self.amount_parsed = cmd.amount_parsed
                    self._error = cmd._error
                raise
            except ValueError as err:
                if isinstance(ci.function, _Alias):
                    # propagate expanded alias
                    cmd = ci.function.cmd
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
        return results

    def _replace(self, chars, replacement):
        # insert replacement taking into account quotes
        i = len(chars)
        c = chars[0]
        if c != '"' or chars[-1] != c:
            completion = replacement
        else:
            completion = c + replacement + c
        j = self.amount_parsed
        t = self.current_text
        self.current_text = t[0:j] + completion + t[i + j:]
        return len(completion)

    def _parse_arg(self, annotation, text, session, final):
        m = _whitespace.match(text)
        start = m.end()
        leading, text = text[:start], text[start:]
        self.amount_parsed += start
        self.current_text += leading
        value, replacement, rest = annotation.parse(text, session)
        if len(rest) > 0:
            text = text[:-len(rest)]
        self.amount_parsed += self._replace(text, replacement)
        return value, rest

    def _find_command_name(self, final, used_aliases=None):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, sets self._ci
        self._error = "Missing command"
        word_map = _commands
        start = self.amount_parsed
        while 1:
            m = _whitespace.match(self.current_text, self.amount_parsed)
            self.amount_parsed = m.end()
            text = self.current_text[self.amount_parsed:]
            if not text:
                return
            word, chars, text = next_token(text)
            if _debugging:
                print('cmd next_token(%r) -> %r %r %r' % (text, word, chars,
                                                          text))
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
                    self._replace(chars, c)
                    text = self.current_text[self.amount_parsed:]
                    continue
                if word:
                    self._error = "Unknown command"
                return
            self.amount_parsed += len(chars)
            if isinstance(what, _Defer):
                what = _lazy_register(word_map, word)
            if isinstance(what, dict):
                # word is part of multiword command name
                word_map = what
                self._error = ("Incomplete command: %s"
                               % self.current_text[start:self.amount_parsed])
                continue
            assert(isinstance(what, CmdDesc))
            cmd_name = self.current_text[start:self.amount_parsed]
            cmd_name = ' '.join(cmd_name.split())   # canonicalize
            if (used_aliases is not None and isinstance(what.function, _Alias)
                    and cmd_name in used_aliases):
                what = _aliased_commands[cmd_name]
            self._ci = what
            self.command_name = cmd_name
            self._error = ''
            return

    def _process_positional_arguments(self):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, updates self._kwargs
        session = self._session()  # resolve back reference
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
                    # matches keyword, so done with positional arguments
                    break
            else:
                self._error = "Missing required argument %s" % name
            m = _whitespace.match(text)
            start = m.end()
            if start:
                self.amount_parsed += start
                text = text[start:]
            if not text:
                break
            if text[0] == ';':
                break
            try:
                value, text = self._parse_arg(anno, text, session, False)
                if iskeyword(name):
                    self._kwargs['%s_' % name] = value
                else:
                    self._kwargs[name] = value
                self._error = ""
            except ValueError as err:
                if isinstance(err, AnnotationError) and err.offset is not None:
                    # We got an error with an offset, that means that an
                    # argument was partially matched, so assume that is the
                    # error the user wants to see.
                    self.amount_parsed += err.offset
                    self._error = "Invalid argument %r: %s" % (name, err)
                    return
                if name in self._ci._required:
                    self._error = "Invalid argument %r: %s" % (name, err)
                    return
                # optional and wrong type, try as keyword
                break

    def _process_keyword_arguments(self, final):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, updates self._kwargs
        session = self._session()  # resolve back reference
        m = _whitespace.match(self.current_text, self.amount_parsed)
        self.amount_parsed = m.end()
        text = self.current_text[self.amount_parsed:]
        if not text:
            return
        while 1:
            word, chars, text = next_token(text)
            if _debugging:
                print('key next_token(%r) -> %r %r' % (text, word, chars))
            if not word or word == ';':
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
                    self._replace(chars, c)
                    text = self.current_text[self.amount_parsed:]
                    self.completions = []
                    continue
                if len(self._ci._keyword) > 0:
                    self._error = "Expected keyword, got '%s'" % word
                else:
                    self._error = "Too many arguments"
                return
            arg_name = word
            self.amount_parsed += len(chars)
            m = _whitespace.match(text)
            start = m.end()
            if start:
                self.amount_parsed += start
                text = text[start:]
            if not text:
                self._error = "Missing argument %r" % arg_name
                break

            anno = self._ci._keyword[arg_name]
            self.completion_prefix = ''
            self.completions = []
            try:
                value, text = self._parse_arg(anno, text, session, final)
                if iskeyword(name):
                    self._kwargs['%s_' % arg_name] = value
                else:
                    self._kwargs[arg_name] = value
            except ValueError as err:
                if isinstance(err, AnnotationError) and err.offset is not None:
                    self.amount_parsed += err.offset
                self._error = "Invalid argument %r: %s" % (arg_name, err)
                return
            m = _whitespace.match(text)
            start = m.end()
            if start:
                self.amount_parsed += start
                text = text[start:]
            if not text:
                break

    def parse_text(self, text, final=False, _used_aliases=None):
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

        while 1:
            self._find_command_name(final, _used_aliases)
            if not self._ci:
                return
            self._process_positional_arguments()
            if self._error:
                return
            self._process_keyword_arguments(final)
            if self._error:
                return
            self._multiple.append((self.command_name, self._ci, self._kwargs))
            self.command_name = None
            self._ci = None
            self._kwargs = {}
            m = _whitespace.match(self.current_text, self.amount_parsed)
            self.amount_parsed = m.end()
            if self.amount_parsed == len(self.current_text):
                return
            self.amount_parsed += 1  # skip semicolon


def command_function(name):
    """Return callable for given command name

    :param name: the name of the command
    :returns: the callable that implements the command
    """
    cmd = Command(None)
    cmd.current_text = name
    cmd._find_command_name(True)
    if not cmd._ci or cmd.amount_parsed != len(cmd.current_text):
        raise ValueError("'%s' is not a command name" % name)
    return cmd._ci.function


def command_help(name):
    """Return help for given command name

    :param name: the name of the command
    :returns: the help object registered with the command
    """
    cmd = Command(None)
    cmd.current_text = name
    cmd._find_command_name(True)
    if not cmd._ci or cmd.amount_parsed != len(cmd.current_text):
        raise ValueError("'%s' is not a command name" % name)
    return cmd._ci.help


def usage(name):
    """Return usage string for given command name

    :param name: the name of the command
    :returns: a usage string for the command
    """
    cmd = Command(None)
    cmd.current_text = name
    cmd._find_command_name(True)
    if not cmd._ci or cmd.amount_parsed != len(cmd.current_text):
        raise ValueError("'%s' is not a command name" % name)

    usage = cmd.command_name
    ci = cmd._ci
    for arg_name in ci._required:
        usage += ' %s' % arg_name
    num_opt = 0
    for arg_name in ci._optional:
        usage += ' [%s' % arg_name
        num_opt += 1
    usage += ']' * num_opt
    for arg_name in ci._keyword:
        type = ci._keyword[arg_name].name
        usage += ' [%s _%s_]' % (arg_name, type.replace(' ', '_'))
    return usage


def html_usage(name):
    """Return usage string in HTML for given command name

    :param name: the name of the command
    :returns: a HTML usage string for the command
    """
    cmd = Command(None)
    cmd.current_text = name
    cmd._find_command_name(True)
    if not cmd._ci or cmd.amount_parsed != len(cmd.current_text):
        raise ValueError("'%s' is not a command name" % name)

    from html import escape
    if cmd._ci.help is None:
        usage = '<b>%s</b>' % escape(cmd.command_name)
    else:
        usage = '<b><a href="%s">%s</a></b>' % (
            cmd._ci.help, escape(cmd.command_name))
    ci = cmd._ci
    for arg_name in ci._required:
        arg = ci._required[arg_name]
        type = arg.name
        if arg.help is None:
            name = escape(arg_name)
        else:
            name = '<a href="%s">%s</a>' % (arg.help, escape(arg_name))
        usage += ' <span title="%s"><i>%s</i></span>' % (escape(type), name)
    num_opt = 0
    for arg_name in ci._optional:
        num_opt += 1
        arg = ci._optional[arg_name]
        type = arg.name
        if arg.help is None:
            name = escape(arg_name)
        else:
            name = '<a href="%s">%s</a>' % (arg.help, escape(arg_name))
        usage += ' [<span title="%s"><i>%s</i></span>' % (escape(type), name)
    usage += ']' * num_opt
    for arg_name in ci._keyword:
        arg = ci._keyword[arg_name]
        if arg.help is None:
            type = escape(arg.name)
        else:
            type = '<a href="%s">%s</a>' % (arg.help, escape(arg.name))
        usage += ' [<b>%s</b> <i>%s</i>]' % (escape(arg_name), type)
    return usage


def registered_commands():
    """Return a list of the currently registered commands"""
    return list(_commands.keys())


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

    def __call__(self, session, *args, optional='', used_aliases=None):
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
        self.cmd = Command(session, text, final=True,
                           _used_aliases=used_aliases)
        return self.cmd.execute(_used_aliases=used_aliases)


_cmd_aliases = set()


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
    logger = session.logger if session else None
    if not name:
        # list aliases
        names = ', '.join(_cmd_aliases)
        if names:
            if logger is not None:
                logger.info('Aliases: %s' % names)
        else:
            if logger is not None:
                logger.status('No aliases.')
        return
    if not text:
        if name not in _cmd_aliases:
            if logger is not None:
                logger.status('No alias named %r found.' % name)
        else:
            if logger is not None:
                logger.info('Aliased %r to %r'
                            % (name, _cmd_aliases[name].original_text))
        return
    name = ' '.join(name.split())   # canonicalize
    cmd = _Alias(text)
    try:
        register(name, cmd.desc(), cmd, logger=logger)
        _cmd_aliases.add(name)
    except:
        raise


@register('~alias', CmdDesc(required=[('name', StringArg)]))
def unalias(session, name):
    """Remove command alias

    :param name: name of the alias
    """
    # remove command alias
    words = name.split()
    name = ' '.join(words)  # canonicalize
    try:
        _cmd_aliases.remove(name)
    except KeyError:
        raise UserError('No alias named %r exists' % name)

    cmd_map = _commands
    for word in words[:-1]:
        what = cmd_map.get(word, None)
        if isinstance(what, dict):
            cmd_map = what
            continue
        raise RuntimeError("internal error")
    word = words[-1]
    what = cmd_map.get(word, None)
    if what is None:
        raise RuntimeError("internal error")
    previous_cmd = _aliased_commands.get(name, None)
    if previous_cmd:
        del _aliased_commands[name]
        cmd_map[word] = previous_cmd
        return
    del cmd_map[word]

if __name__ == '__main__':
    from utils import flattened

    class ColorArg(Annotation):
        name = 'a color'

        Builtin_Colors = {
            "light gray": (211, 211, 211),
            "red": (255, 0, 0)
        }

        @staticmethod
        def parse(text, session):
            # Allow for color names to be truncated
            # and to be composed of multiple words.
            # Might want to accept non-prefix abbreviations,
            # eg., "lt" for "light".
            token, chars, rest = next_token(text)
            token = token.casefold()
            color_name = token
            while 1:
                try:
                    color = ColorArg.Builtin_Colors[color_name]
                    return ([x / 255.0 for x in color]), color_name, rest
                except KeyError:
                    pass
                # check if color_name is a prefix
                names = [n for n in ColorArg.Builtin_Colors
                         if n.startswith(color_name)]
                if len(names) == 0:
                    raise ValueError("Invalid color name: %r" % color_name)
                suffix = names[0][len(color_name):]
                if ' ' not in suffix:
                    color_name = names[0]
                    continue
                if not suffix[0].isspace():
                    color_name += suffix.split(None, 1)[0]

                m = _whitespace.match(rest)
                rest = rest[m.end():]
                color_name += ' '

                token, chars, rest = next_token(rest)
                token = token.casefold()
                color_name += token

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
        keyword=[('color', ColorArg), ('radius', FloatArg)]
    )

    @register('test2', test2_desc)
    def test2(session, color=None, radius=0):
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

    def lazy_reg():
        test11_desc = CmdDesc()

        def test11(session):
            print('delayed')
        register('xyzzy subcmd', test11_desc, test11)
    delay_registration('xyzzy', lazy_reg)

    if len(sys.argv) > 1:
        _debugging = 'd' in sys.argv[1]

        @register('exit')
        def exit(session):
            raise SystemExit(0)

        @register('echo', CmdDesc(optional=[('text', RestOfLine)]))
        def echo(session, text=''):
            return text

        register('usage', CmdDesc(required=[('name', RestOfLine)]), usage)

        register('html_usage', CmdDesc(required=[('name', RestOfLine)]),
                 html_usage)
        prompt = 'cmd> '
        cmd = Command(None)
        while True:
            try:
                text = input(prompt)
                cmd.parse_text(text, final=True)
                results = cmd.execute()
                for result in flattened(results):
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
        (True, True, 'xyzzy'),
        (False, True, 'xyzzy subcmd'),
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
        (True, True, 'test6 3.4, abc, 7.8'),
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
        (False, True, 'test10 re 0.5'),
        (True, False, 'test10 red, light gray'),
        (False, False, 'test10 light  gray, red 0.33, 0.67'),
        (True, False, 'test10 light  gray, red 0.33'),
        (True, False, 'test10 li  gr, red'),
        (True, False, 'test10 "red"10.3'),
        (False, False, 'test4 2; test4 3'),
        (False, False, 'test10 red 2; test10 li gr 3'),
        (True, False, 'test10 red 2; test10 3 red'),
        (True, False, 'test10 red 2; test6 123, red'),
    ]
    # TODO: delayed registration tests
    sys.stdout = sys.stderr
    successes = 0
    failures = 0
    cmd = Command(None)
    for t in tests:
        fail, final, text = t
        try:
            print("\nTEST: '%s'" % text)
            cmd.parse_text(text, final=final)
            print(cmd.current_text)
            results = cmd.execute()
            if results:
                for result in flattened(results):
                    if result is not None:
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
