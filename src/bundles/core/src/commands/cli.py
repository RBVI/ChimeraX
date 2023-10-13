# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
cli: Command line interface
===========================

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
If the argument name is the same as a Python keyword,
then an underscore appended to it to form the Python argument name.
The names of the optional arguments are used to
let them be given as keyword arguments as well.
Multiple value arguments are separated by commas
and the commas may be followed by whitespace.
Depending on the type of an argument, *e.g.*, a color name,
whitespace can also appear within an argument value.
String argument values may be quoted with double or single quotes,
and Python's textual escape sequences are recognized,
*e.g.*, ``\\N{LATIN CAPITAL LETTER A WITH RING ABOVE}`` for the ångström sign,
\N{LATIN CAPITAL LETTER A WITH RING ABOVE}.

Words in the command name may be truncated
and are automatically completed to the first registered
command with the given prefix.  Likewise for keyword arguments.

Keywords are case sensitive, and are expected to be all lowercase.
Underscores are elided, but are documented as mixed case.
For example, a ``bg_color`` keyword would be documented as ``bgColor``.

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

.. _Type Annotations:

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

    import from chimerax.core.commands import Command, CmdDesc, RestOfLine
    import from chimerax.core import errors
    @register("echo", CmdDesc(optional=[('text', RestOfLine)]))
    def echo(session, text=''):
        print(text)
    ...
    command = Command(session)
    try:
        status = command.run(text)
        if status:
            print(status)
    except errors.UserError as err:
        print(err, file=sys.stderr)


Command Aliases
---------------

    Normally, command aliases are made with the alias command, but
    they can also be explicitly register with :py:func:`alias` and
    removed with :py:func:`remove_alias`.

    An alias definition uses **$n** to refer to passed in arguments.
    $1 may appear more than once.  $$ is $.

    To register a multiword alias, quote the command name.
"""

import abc
from keyword import iskeyword as is_python_keyword
import re
import sys
from collections import OrderedDict
from typing import Callable, Any
from contextlib import contextmanager
from ..errors import UserError

_debugging = False
_normal_token = re.compile(r"[^;\s]*")
_single_quote = re.compile(r"'(.|\')*?'(\s|$)")
_internal_single_quote = re.compile(r"'\s")
_double_quote = re.compile(r'"(.|\")*?"(\s|$)')
_internal_double_quote = re.compile(r'"\s')
_whitespace = re.compile(r"\s*")
_internal_whitespace = re.compile(r"\s+")


def commas(text_seq, conjunction='or'):
    """Return comma separated list of words

    :param text_seq: a sequence of text strings
    :param conjunction: a word to put for last item (default 'or')
    """
    if not isinstance(text_seq, (list, tuple)):
        text_seq = tuple(text_seq)
    seq_len = len(text_seq)
    if seq_len == 0:
        return ""
    if seq_len == 1:
        return text_seq[0]
    if seq_len == 2:
        return '%s %s %s' % (text_seq[0], conjunction, text_seq[1])
    text = '%s, %s %s' % (', '.join(text_seq[:-1]), conjunction, text_seq[-1])
    return text


def plural_form(things, word, plural=None):
    """Return plural of word based on length of sequence

    :param things: a sequence of objects or an integer
    :param word: word to form the plural of
    :param plural: optional explicit plural of word, otherwise best guess
    """
    count = things if isinstance(things, int) else len(things)
    if count == 1:
        return word
    if plural is None:
        return plural_of(word)
    return plural


def plural_of(word):
    """Return best guess of the American English plural of the word

    :param word: the word to form the plural version

    The guess is rudimentary, e.g., it does not handle changing "leaf" to
    "leaves".  So in general, use the :py:func:`plural_form` function
    and explicitly give the plural.
    """
    if word.endswith('o'):
        if word.casefold() in ('zero', 'photo', 'quarto'):
            return word + 's'
        return word + 'es'
    if word.endswith('ius'):
        return word[:-2] + 'i'
    if word.endswith('ix'):
        return word[:-1] + 'ces'
    if word.endswith(('sh', 'ch', 'ss', 'x')):
        return word + 'es'
    if word.endswith('y'):
        if word[-2] not in 'aeiou' or word.endswith('quy'):
            return word[:-1] + 'ies'
        return word + 's'
    if word.endswith('s'):
        # 'ss' special-cased above; other special cases may be needed
        return word
    # TODO: special case words, e.g. leaf -> leaves, hoof -> hooves
    return word + 's'


def discard_article(text):
    """remove leading article from text"""
    text_seq = text.split(None, 1)
    if text_seq[0] in ('a', 'an', 'the', 'some'):
        return text_seq[1]
    return text


_small_ordinals = {
    1: "first",
    2: "second",
    3: "third",
    4: "forth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}


def ordinal(i):
    """Return ordinal number name of number"""
    if i <= 10:
        return _small_ordinals[i]
    if i % 100 == 11 or i % 10 != 1:
        return "%dth" % i
    return "%dst" % i


def dq_repr(obj):
    """Like repr, but use double quotes"""
    r = repr(obj)
    if r[0] != "'":
        return r
    result = []
    for c in r:
        if c == '"':
            result.append("'")
        elif c == "'":
            result.append('"')
        else:
            result.append(c)
    return ''.join(result)


def user_kw(kw_name):
    """Return user version of a keyword argument name."""
    words = kw_name.split('_')
    return words[0] + ''.join([x.capitalize() for x in words[1:]])


def _user_kw_cnt(kw_name):
    """Return user version of a keyword argument name and number of words."""
    words = kw_name.split('_')
    return words[0] + ''.join([x.capitalize() for x in words[1:]]), len(words)


class AnnotationError(UserError, ValueError):
    """Error, with optional offset, in an annotation"""

    def __init__(self, message, offset=None):
        super().__init__(message)
        self.offset = offset


class Annotation(metaclass=abc.ABCMeta):
    """Supported API.  Base class for all annotations

    Annotations encapsulate how to parse text into a particular
    type.  Both subclasses and instances of subclasses can be
    used as annotations.  An instance is used when an annotation
    is parameterized, e.g., with the length of a sequence.

    Each annotation should have the following attributes:

    .. py:attribute:: name

        Set to textual description of the annotation, including
        the leading article, *e.g.*, `"an integer"`.

    .. py:attribute:: url

       URL for help information.

    .. py:method:: html_name

        A class or instance method that returns the HTML version of the name.
    """
    name = "an unspecified type"
    url = None
    _html_name = None

    def __init__(self, name=None, url=None, html_name=None):
        if name is not None:
            self.name = name
        if url is not None:
            self.url = url
        if html_name is not None:
            self._html_name = html_name
        elif name is not None:
            from html import escape
            self._html_name = escape(name)
        # If __init__ is called, then an Annotation instance is being
        # created, and we should use the instance's HTML name
        self.html_name = self.inst_html_name

    @staticmethod
    def parse(text, session):
        """Supported API.  Convert text to appropriate type.

        :param text: command line text to parse
        :param session: for session-dependent data types
        :returns: 3-tuple with the converted value, consumed text
            (possibly altered with expanded abbreviations), and the
            remaining unconsumed text
        :raises AnnotationError: if unable to convert text

        The leading space in text must already be removed.
        It is up to the particular annotation to support abbreviations.

        Empty text should raise a ValueError or AnnotationError exception
        (the exceptions being :py:class:`NoArg` and :py:class:`EmptyArg`).
        """
        raise NotImplementedError

    @classmethod
    def unparse(cls, value, session=None):
        """Convert value to string that can parsed back into value

        :param value: the value to convert
        :param session: requried for session-dependent data types
        :returns: string version of the value

        The return value should be what would be typed if the value
        were given on the command line.
        """
        raise NotImplementedError("Please implement unparse for %s" % cls.__name__)

    @classmethod
    def html_name(cls, name=None):
        """Supported API.  Return HTML verison of name attribute

        If the :py:attr:`url` attribute is set, then a HTML href using that
        URL is returned."""
        if cls._html_name is not None:
            return cls._html_name
        from html import escape
        if name is None:
            name = cls.name
        if cls.url is None:
            return escape(name, quote=False)
        return '<a href="%s">%s</a>' % (escape(cls.url), escape(name, quote=False))

    def inst_html_name(self, name=None):
        # used to override html_name class method with an instance method
        if self._html_name is not None:
            return self._html_name
        from html import escape
        if name is None:
            name = self.name
        if self.url is None:
            return escape(name, quote=False)
        return '<a href="%s">%s</a>' % (escape(self.url), escape(name, quote=False))


class Aggregate(Annotation):
    """Common class for collections of values.

    Aggregate(annotation, constructor, add_to, min_size=None, max_size=None,
            name=None, url=None) -> annotation

    :param annotation: annotation for values in the collection.
    :param min_size: minimum size of collection, default `None`.
    :param max_size: maximum size of collection, default `None`.
    :param name: optionally override name in error messages.
    :param url: optionally give documentation URL.
    :param prefix: optionally required prefix to aggregate.

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
                 max_size=None, name=None, url=None, prefix=None):
        if (not isinstance(annotation, Annotation) and (
                not isinstance(annotation, type) or not issubclass(annotation, Annotation))):
            raise ValueError("need an annotation, not %s" % annotation)
        Annotation.__init__(self, name, url)
        self.annotation = annotation
        if min_size is not None:
            self.min_size = min_size
        if max_size is not None:
            self.max_size = max_size
        if name is None:
            if ',' in annotation.name:
                self.name = "a collection of %s" % annotation.name
                self._html_name = "a collection of %s" % annotation.html_name()
            else:
                noun = plural_of(discard_article(annotation.name))
                self.name = "some %s" % noun
                self._html_name = "some %s" % annotation.html_name(noun)
        self.prefix = prefix

    def add_to(self, container, element):
        """Add to add an element to the container

        :param container: the container to add elements to
        :param element: the element to add to the container
        :returns: None for mutable containers, or a new
            container if immutable.
        """
        raise NotImplementedError

    def parse(self, text, session):
        result = self.constructor()
        used = ''
        if self.prefix and text.startswith(self.prefix):
            text = text[len(self.prefix):]
            used += self.prefix
        while True:
            # find next list-separator character while honoring quoting
            quote_mark = None
            for i, c in enumerate(text):
                if quote_mark:
                    if c == quote_mark:
                        quote_mark = None
                    continue
                if c in ["'", '"']:
                    quote_mark = c
                    continue
                if c == self.separator:
                    break
            else:
                i = -1
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
            raise AnnotationError("Need %s %d '%s'-separated %s" % (
                qual, self.min_size, self.separator,
                discard_article(self.name)), len(used))
        if len(result) > self.max_size:
            if self.min_size == self.max_size:
                qual = "exactly"
            else:
                qual = "at most"
            raise AnnotationError("Need %s %d '%s'-separated %s" % (
                qual, self.max_size, self.separator,
                discard_article(self.name)), len(used))
        return result, used, rest

    def unparse(self, value, session=None):
        # consecutive tuples are easier to "visually parse" if no internal spaces
        return self.separator.join(self.annotation.unparse(v) for v in value)


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

    def __init__(self, annotation, size, name=None, url=None):
        return Aggregate.__init__(self, annotation, size, size, name=name,
                                  url=url)

    def add_to(self, container, value):
        return container + (value,)


class DottedTupleOf(Aggregate):
    """Annotation for dot-separated lists of a single type

    DottedListOf(annotation, min_size=None, max_size=None) -> annotation
    """
    separator = '.'
    constructor = tuple

    def __init__(self, annotation, min_size=None,
                 max_size=None, name=None, url=None, prefix=None):
        Aggregate.__init__(self, annotation, min_size, max_size, name, url,
                           prefix)
        if name is None:
            if ',' in annotation.name:
                self.name = "a dotted list of %s" % annotation.name
                self._html_name = "a dotted list of %s" % annotation.html_name()
            else:
                name = discard_article(annotation.name)
                self.name = "dotted %s(s)" % name
                self._html_name = "dotted %s(s)" % annotation.html_name(name)

    def add_to(self, container, value):
        return container + (value,)


class RepeatOf(Annotation):
    '''
    Annotation for keyword options that can occur multiple times.

    RepeatOf(annotation) -> annotation

    Option values are put in list even if option occurs only once.
    '''
    allow_repeat = True

    def __init__(self, annotation):
        Annotation.__init__(self)
        self.name = annotation.name + ', repeatable'
        self._html_name = annotation.html_name() + ', <i>repeatable</i>'
        self.parse = annotation.parse


class Bounded(Annotation):
    """Support bounded numerical values

    Bounded(annotation, min=None, max=None, inclusive=True, name=None, url=None) -> an Annotation

    :param annotation: numerical annotation
    :param min: optional lower bound
    :param max: optional upper bound
    :param inclusive: is the bound itself in the range
    :param name: optional explicit name for annotation
    :param url: optionally give documentation URL.
    """

    def __init__(self, annotation, min=None, max=None, *, inclusive=True, name=None, url=None, html_name=None):
        Annotation.__init__(self, name, url)
        self.anno = annotation
        self.min = min
        self.max = max
        self.inclusive = inclusive
        if name is None:
            op = "=" if inclusive else ""
            if min is not None and max is not None:
                self.name = "%s >%s %s and <%s %s" % (annotation.name, op, min, op, max)
            elif min is not None:
                self.name = "%s >%s %s" % (annotation.name, op, min)
            elif max is not None:
                self.name = "%s <%s %s" % (annotation.name, op, max)
            else:
                self.name = annotation.name
        if html_name is None:
            if inclusive:
                l, g = "le", "ge"
            else:
                l, g = "lt", "gt"
            if min is not None and max is not None:
                self._html_name = "%s &%s; %s and &%s; %s" % (annotation.html_name(), g, min, l, max)
            elif min is not None:
                self._html_name = "%s &%s; %s" % (annotation.html_name(), g, min)
            elif max is not None:
                self._html_name = "%s &%s; %s" % (annotation.html_name(), l, max)
            else:
                self._html_name = annotation.html_name()

    def parse(self, text, session):
        value, new_text, rest = self.anno.parse(text, session)
        if self.min is not None and ((value < self.min) if self.inclusive else (value <= self.min)):
            raise AnnotationError(
                "Must be greater than %s%s" % (
                    "or equal to " if self.inclusive else "", self.min), len(text) - len(rest))
        if self.max is not None and ((value > self.max) if self.inclusive else (value >= self.max)):
            raise AnnotationError(
                "Must be less than %s%s" % (
                    "or equal to " if self.inclusive else "", self.max), len(text) - len(rest))
        return value, new_text, rest

    def unparse(self, value, session=None):
        return self.anno.unparse(value, session)


class EnumOf(Annotation):
    """Support enumerated types

    EnumOf(values, ids=None, name=None, url=None) -> an Annotation

    :param values: iterable of values
    :param ids: optional iterable of identifiers
    :param abbreviations: if not None, then override allow_truncated
    :param name: optional explicit name for annotation
    :param url: optionally give documentation URL.

    .. data: allow_truncated

        (Defaults to True.)  If true, then recognize truncated ids.

    If the *ids* are given, then there must be one for each
    and every value, otherwise the values are used as the identifiers.
    The identifiers must all be strings.
    """

    allow_truncated = True

    def __init__(self, values, ids=None, abbreviations=None, name=None, url=None, case_sensitive=False):
        from collections.abc import Iterable
        if isinstance(values, Iterable):
            values = list(values)
        if ids is not None:
            if isinstance(ids, Iterable):
                ids = list(ids)
            if len(values) != len(ids):
                raise ValueError("Must have an identifier for "
                                 "each and every value")
        Annotation.__init__(self, name, url)
        # We make sure the ids are sorted so that even if we are given
        # values=["abc", "ab"], an input of "ab" will still match
        # "ab", not "abc".
        if ids is not None:
            assert all([isinstance(x, str) for x in ids])
            pairs = sorted(zip(ids, values))
            self.ids = [p[0] for p in pairs]
            self.values = [p[1] for p in pairs]
        else:
            assert all([isinstance(x, str) for x in values])
            self.ids = self.values = sorted(values)
        if name is None:
            from html import escape
            if len(self.ids) == 1:
                self.name = "'%s'" % self.ids[0]
                self._html_name = '<b>%s</b>' % escape(self.ids[0])
            else:
                self.name = "one of %s" % commas(["'%s'" % i for i in self.ids])
                self._html_name = "one of %s" % commas(["<b>%s</b>" % escape(i) for i in self.ids])
        if abbreviations is not None:
            self.allow_truncated = abbreviations

        self._case_sensitive = case_sensitive

    def parse(self, text, session):
        if not text:
            raise AnnotationError("Expected %s" % self.name)
        token, text, rest = next_token(text, convert=True)
        case = self._case_sensitive
        word = token if case else token.casefold()
        matches = []
        for i, ident in enumerate(self.ids):
            id = ident if case else ident.casefold()
            if id == word:
                return self.values[i], quote_if_necessary(ident), rest
            elif self.allow_truncated:
                if id.startswith(word):
                    matches.append((i, ident))
        if len(matches) == 1:
            i, ident = matches[0]
            return self.values[i], quote_if_necessary(ident), rest
        elif len(matches) > 1:
            ms = ', '.join(self.ids[i] for i, ident in matches)
            raise AnnotationError("'%s' is ambiguous, could be %s" % (token, ms))
        raise AnnotationError("Should be %s" % self.name)

    def unparse(self, value, session=None):
        i = self.values.index(value)
        return quote_if_necessary(self.ids[i])


class DynamicEnum(Annotation):
    '''Enumerated type where enumeration values computed from a function.'''

    def __init__(self, values_func, name=None, url=None, html_name=None,
                 case_sensitive=False):
        Annotation.__init__(self, url=url)
        self.__name = name
        self.__html_name = html_name
        self._case_sensitive = case_sensitive
        self.values_func = values_func

    def parse(self, text, session):
        e = EnumOf(self.values_func(), case_sensitive=self._case_sensitive)
        return e.parse(text, session)

    def unparse(self, value, session=None):
        e = EnumOf(self.values_func(), case_sensitive=self._case_sensitive)
        return e.unparse(value, session)

    @property
    def name(self):
        if self.__name is not None:
            return self.__name
        return 'one of ' + commas(
            ["'%s'" % str(v) for v in sorted(self.values_func(), key=lambda k: (k.lower(), k))])

    @property
    def _html_name(self):
        if self.__html_name is not None:
            return self.__html_name
        from html import escape
        if self.__name is not None:
            name = self.__name
        else:
            name = 'one of ' + commas([
                "<b>%s</b>" % escape(str(v))
                for v in sorted(self.values_func(), key=lambda k: (k.lower(), k))])
        if self.url is None:
            return name
        return '<a href="%s">%s</a>' % (escape(self.url), name)


class Or(Annotation):
    """Support two or more alternative annotations

    Or(annotation, annotation [, annotation]*, name=None, url=None) -> an Annotation

    :param name: optional explicit name for annotation
    :param url: optionally give documentation URL.
    """

    def __init__(self, *annotations, name=None, url=None, html_name=None):
        if len(annotations) < 2:
            raise ValueError("Need at least two alternative annotations")
        Annotation.__init__(self, name, url)
        self.annotations = annotations
        if name is None:
            self.name = commas([a.name for a in annotations])
        if html_name is None:
            from html import escape
            if name is not None:
                self._html_name = escape(name)
            else:
                self._html_name = commas([a.html_name() for a in annotations])

    def parse(self, text, session):
        for anno in self.annotations:
            try:
                return anno.parse(text, session)
            except AnnotationError as err:
                if err.offset:
                    raise
            except ValueError:
                pass
        raise AnnotationError("Expected %s" % self.name)

    def unparse(self, value, session=None):
        for a in self.annotations:
            try:
                return a.unparse(value, session)
            except ValueError:
                pass
        raise ValueError("value does not match known types")


class BoolArg(Annotation):
    """Annotation for boolean literals"""
    name = "true or false"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % BoolArg.name)
        token, text, rest = next_token(text)
        token = token.casefold()
        if token == "0" or "false".startswith(token) or token == "off":
            return False, "false", rest
        if token == "1" or "true".startswith(token) or token == "on":
            return True, "true", rest
        raise AnnotationError("Expected true or false (or 1 or 0)")

    @staticmethod
    def unparse(value, session=None):
        return str(bool(value)).casefold()


class NoArg(Annotation):
    """Annotation for keyword whose presence indicates True"""
    name = "nothing"

    @staticmethod
    def parse(text, session):
        return True, "", text

    @staticmethod
    def unparse(value, session=None):
        return ""


class EmptyArg(Annotation):
    """Annotation for optionally missing 'required' argument"""
    name = "nothing"

    @staticmethod
    def parse(text, session):
        return None, "", text

    @staticmethod
    def unparse(value, session=None):
        return ""


class NoneArg(Annotation):
    """Annotation for 'none' (typically used with Or)"""
    name = "none"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % NoneArg.name)
        token, text, rest = next_token(text)
        if token.lower() == "none":
            return None, text, rest
        raise AnnotationError("Expected %s" % NoneArg.name)

    @staticmethod
    def unparse(value, session=None):
        assert value is None
        return "none"


class IntArg(Annotation):
    """Annotation for integer literals"""
    name = "an integer"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % IntArg.name)
        token, text, rest = next_token(text)
        try:
            return int(token), text, rest
        except ValueError:
            raise AnnotationError("Expected %s" % IntArg.name)

    @staticmethod
    def unparse(value, session=None):
        assert isinstance(value, int)
        return str(value)


class FloatArg(Annotation):
    """Annotation for floating point literals"""
    name = "a number"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % FloatArg.name)
        token, text, rest = next_token(text)
        try:
            return float(token), text, rest
        except ValueError:
            raise AnnotationError("Expected %s" % FloatArg.name)

    @staticmethod
    def unparse(value, session=None):
        assert isinstance(value, (float, int))
        return str(value)


class FloatOrDeltaArg(Annotation):
    """Annotation for non-negative floating point literals, but if preceded by explicit +/- then interpreted as a delta"""
    name = "a number >= 0 or a +/- delta"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % FloatOrDeltaArg.name)
        is_delta = text[0] in "+-"
        token, text, rest = next_token(text)
        try:
            return (is_delta, float(token)), text, rest
        except ValueError:
            raise AnnotationError("Expected %s" % FloatOrDeltaArg.name)

    @staticmethod
    def unparse(value, session=None):
        is_delta, num = value
        assert isinstance(num, (float, int))
        return ('+' if is_delta and num > 0 else '') + str(num)


class StringArg(Annotation):
    """Annotation for text (a word or quoted)"""
    name = "a text string"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % StringArg.name)
        token, text, rest = next_token(text, convert=True)
        # use quote_if_necessary(token) since token will have had
        # it's surrounding quotes stripped, and you don't want them
        # quoted a second time
        return token, quote_if_necessary(token), rest

    @staticmethod
    def unparse(value, session=None):
        assert isinstance(value, str)
        return quote_if_necessary(value)


class CharacterArg(StringArg):
    name = "a single character"

    @staticmethod
    def parse(text, session):
        token, log_token, rest = StringArg.parse(text, session)
        if len(token) != 1:
            raise AnnotationError("must be a single character")
        return token, log_token, rest


class PasswordArg(StringArg):
    """Annotation for a password (should not be echoed to log)"""
    name = "a password"

    @staticmethod
    def parse(text, session):
        token, text, rest = StringArg.parse(text, session)
        return token, "\N{BULLET}\N{BULLET}\N{BULLET}\N{BULLET}\N{BULLET}\N{BULLET}", rest

    @staticmethod
    def unparse(value, session=None):
        assert isinstance(value, str)
        return quote_if_necessary(value)


class AttrNameArg(StringArg):
    """Annotation for a Python attribute name"""
    name = "a Python attribute name"

    @staticmethod
    def parse(text, session):
        token, text, rest = StringArg.parse(text, session)
        if not text:
            raise AnnotationError("Attribute names can't be empty strings")
        non_underscore = text.replace('_', '')
        if not non_underscore.isalnum():
            raise AnnotationError("Attribute names can consist only of alphanumeric"
                                  " characters and underscores")
        if text[0].isdigit():
            raise AnnotationError("Attribute names cannot start with a digit")
        return token, text, rest

    @staticmethod
    def unparse(value, session=None):
        assert value and isinstance(value, str)
        return quote_if_necessary(value)


class FileNameArg(Annotation):
    """Base class for Open/SaveFileNameArg"""
    name = "a file name"
    # name_filter should be a string compatible with QFileDialog.setNameFilter(),
    # or None (which means all ChimeraX-openable types)
    name_filter = None

    @classmethod
    def parse(cls, text, session):
        if not text:
            raise AnnotationError("Expected %s" % cls.name)
        token, text, rest = next_token(text)
        import os.path
        token = os.path.expanduser(token)
        return token, text, rest

    @staticmethod
    def unparse(value, session=None):
        import os
        assert isinstance(value, (str, os.PathLike))
        return quote_path_if_necessary(str(value))


_BROWSE_STRING = "browse"


def _browse_parse(text, session, item_kind, name_filter, accept_mode, dialog_mode, check_existence,
                  *, return_list=False):
    path, text, rest = FileNameArg.parse(text, session)
    if path == _BROWSE_STRING:
        if not session.ui.is_gui:
            raise AnnotationError("Cannot browse for %s name in nogui mode" % item_kind)
        from Qt.QtWidgets import QFileDialog
        dlg = QFileDialog()
        dlg.setAcceptMode(accept_mode)
        if name_filter is not None:
            dlg.setNameFilter(name_filter)
        elif accept_mode == QFileDialog.AcceptMode.AcceptOpen and dialog_mode != QFileDialog.FileMode.Directory:
            from chimerax.open_command.dialog import make_qt_name_filters
            dlg.setNameFilters(make_qt_name_filters(session)[0])
        dlg.setFileMode(dialog_mode)
        if dlg.exec():
            paths = dlg.selectedFiles()
            if not paths:
                raise AnnotationError("No %s selected by browsing" % item_kind)
        else:
            from chimerax.core.errors import CancelOperation
            raise CancelOperation("%s browsing cancelled" % item_kind.capitalize())
    else:
        paths = [path]
    if check_existence:
        import os.path
        for path in paths:
            if not os.path.exists(os.path.expanduser(path)):
                raise AnnotationError("File '%s' does not exist" % path)
    return (paths if return_list else paths[0]), " ".join([quote_path_if_necessary(p) for p in paths]), rest


class OpenFileNameArg(FileNameArg):
    """Annotation for a file to open"""
    name = "name of a file to open/read; a name of 'browse' will bring up a file browser"
    check_existence = True

    @classmethod
    def parse(cls, text, session):
        if session.ui.is_gui:
            from Qt.QtWidgets import QFileDialog
            accept_mode, dialog_mode = QFileDialog.AcceptMode.AcceptOpen, QFileDialog.FileMode.ExistingFile
        else:
            accept_mode = dialog_mode = None
        return _browse_parse(text, session, "file", cls.name_filter, accept_mode, dialog_mode,
                             cls.check_existence)


class OpenFileNamesArg(Annotation):
    """Annotation for opening one or more files"""
    name = "file names to open; a name of 'browse' will bring up a file browser"
    # name_filter should be a string compatible with QFileDialog.setNameFilter(),
    # or None (which means all ChimeraX-openable types)
    name_filter = None
    allow_repeat = "expand"
    check_existence = True

    @classmethod
    def parse(cls, text, session):
        # horrible hack to get repeatable-parsing to work when 'browse' could return multiple files
        if session.ui.is_gui:
            from Qt.QtWidgets import QFileDialog
            accept_mode, dialog_mode = QFileDialog.AcceptMode.AcceptOpen, QFileDialog.FileMode.ExistingFiles
        else:
            accept_mode = dialog_mode = None
        return _browse_parse(text, session, "file", cls.name_filter, accept_mode, dialog_mode,
                             cls.check_existence, return_list=True)

    @staticmethod
    def unparse(value, session=None):
        import os
        assert not isinstance(value, (str, os.PathLike))
        return ' '.join(quote_path_if_necessary(p) for p in value)


class SaveFileNameArg(FileNameArg):
    """Annotation for a file to save"""
    name = "name of a file to save/write; a name of 'browse' will bring up a file browser"
    check_existence = False

    @classmethod
    def parse(cls, text, session):
        if session.ui.is_gui:
            from Qt.QtWidgets import QFileDialog
            accept_mode, dialog_mode = QFileDialog.AcceptMode.AcceptSave, QFileDialog.FileMode.AnyFile
        else:
            accept_mode = dialog_mode = None
        return _browse_parse(text, session, "file", cls.name_filter, accept_mode, dialog_mode,
                             cls.check_existence)


class OpenFolderNameArg(FileNameArg):
    """Annotation for a folder to open from"""
    name = "name of a folder to open/read; a name of 'browse' will bring up a file browser"
    check_existence = True

    @classmethod
    def parse(cls, text, session):
        if session.ui.is_gui:
            from Qt.QtWidgets import QFileDialog
            accept_mode, dialog_mode = QFileDialog.AcceptMode.AcceptOpen, QFileDialog.FileMode.Directory
        else:
            accept_mode = dialog_mode = None
        return _browse_parse(text, session, "folder", cls.name_filter, accept_mode, dialog_mode,
                             cls.check_existence)


class SaveFolderNameArg(FileNameArg):
    """Annotation for a folder to save to"""
    name = "name of a folder to save/write; a name of 'browse' will bring up a file browser"
    check_existence = False

    @classmethod
    def parse(cls, text, session):
        if session.ui.is_gui:
            from Qt.QtWidgets import QFileDialog
            accept_mode, dialog_mode = QFileDialog.AcceptMode.AcceptSave, QFileDialog.FileMode.Directory
        else:
            accept_mode = dialog_mode = None
        return _browse_parse(text, session, "folder", cls.name_filter, accept_mode, dialog_mode,
                             cls.check_existence)


# Atom Specifiers are used in lots of places
# avoid circular import by importing here
from .atomspec import AtomSpecArg  # noqa


class ModelsArg(AtomSpecArg):
    """Parse command models specifier"""
    name = "a models specifier"

    @classmethod
    def parse(cls, text, session):
        aspec, text, rest = super().parse(text, session)
        models = aspec.evaluate(session).models
        return models, text, rest

    @classmethod
    def unparse(cls, models, session):
        from .run import concise_model_spec
        return concise_model_spec(session, models)


class TopModelsArg(AtomSpecArg):
    """Parse command models specifier"""
    name = "a models specifier"

    @classmethod
    def parse(cls, text, session):
        aspec, text, rest = super().parse(text, session)
        models = aspec.evaluate(session).models
        tmodels = _remove_child_models(models)
        return tmodels, text, rest

    @classmethod
    def unparse(cls, tmodels, session):
        from .run import concise_model_spec
        return concise_model_spec(session, tmodels)


class ObjectsArg(AtomSpecArg):
    """Parse command objects specifier"""
    name = "an objects specifier"

    @classmethod
    def parse(cls, text, session):
        aspec, text, rest = super().parse(text, session)
        objects = aspec.evaluate(session)
        objects.spec = str(aspec)
        return objects, text, rest


class ModelArg(AtomSpecArg):
    """Parse command model specifier"""
    name = "a model specifier"

    @classmethod
    def parse(cls, text, session):
        aspec, text, rest = super().parse(text, session)
        models = _remove_child_models(aspec.evaluate(session).models)
        if len(models) != 1:
            raise AnnotationError('Must specify 1 model, got %d' % len(models), len(text))
        return tuple(models)[0], text, rest

    @classmethod
    def unparse(cls, model, session):
        return model.atomspec


class SurfacesArg(ModelsArg):
    """Parse command surfaces specifier"""
    name = "a surfaces specifier"

    @classmethod
    def parse(cls, text, session):
        models, text, rest = super().parse(text, session)
        from ..models import Surface
        surfs = [m for m in models if isinstance(m, Surface)]
        return surfs, text, rest


class SurfaceArg(SurfacesArg):
    """Parse command surfaces specifier"""
    name = "a surface specifier"

    @classmethod
    def parse(cls, text, session):
        surfs, text, rest = super().parse(text, session)
        if len(surfs) != 1:
            raise AnnotationError('Require 1 surface, got %d' % len(surfs))
        return surfs[0], text, rest


class AxisArg(Annotation):
    '''Annotation for axis vector that can be 3 floats or "x", or "y", or "z"
    or two atoms.'''
    name = 'an axis vector'

    named_axes = {
        'x': (1, 0, 0), 'X': (1, 0, 0), '-x': (-1, 0, 0), '-X': (-1, 0, 0),
        'y': (0, 1, 0), 'Y': (0, 1, 0), '-y': (0, -1, 0), '-Y': (0, -1, 0),
        'z': (0, 0, 1), 'Z': (0, 0, 1), '-z': (0, 0, -1), '-Z': (0, 0, -1)
    }

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % AxisArg.name)
        axis = None

        # "x" or "y" or "z"
        if axis is None:
            token, atext, rest = next_token(text)
            try:
                coords = AxisArg.named_axes[token]
            except KeyError:
                pass
            else:
                axis = Axis(coords)

        # 3 comma-separated numbers
        if axis is None:
            try:
                coords, atext, rest = Float3Arg.parse(text, session)
            except Exception:
                pass
            else:
                axis = Axis(coords)

        # Two atoms or a bond.
        if axis is None:
            try:
                from chimerax.atomic import AtomsArg
                atoms, atext, rest = AtomsArg.parse(text, session)
            except Exception:
                pass
            else:
                if len(atoms) == 2:
                    axis = Axis(atoms=atoms)
                elif len(atoms) > 0:
                    raise AnnotationError('Axis argument requires 2 atoms, got %d atoms' % len(atoms))

        # AxisModel
        if axis is None:
            try:
                from chimerax.axes_planes import AxisModelArg
                axis_model, atext, rest = AxisModelArg.parse(text, session)
            except Exception:
                pass
            else:
                axis = Axis(axis_model=axis_model)

        if axis is None:
            raise AnnotationError('Expected 3 floats or "x", or "y", or "z" or two atoms')

        return axis, atext, rest


class Axis:

    def __init__(self, coords=None, atoms=None, axis_model=None):
        if coords is not None:
            from numpy import array, float32
            coords = array(coords, float32)
        self.coords = coords   # Camera coordinates
        self.atoms = atoms
        self.axis_model = axis_model

    def scene_coordinates(self, coordinate_system=None, camera=None, normalize=True):
        atoms = self.atoms
        if atoms is not None:
            a = atoms[1].scene_coord - atoms[0].scene_coord
        elif self.coords is not None:
            if coordinate_system is not None:
                # Camera coords are actually coordinate_system coords if
                # coordinate_system is not None.
                a = coordinate_system.transform_vectors(self.coords)
            elif camera:
                a = camera.position.transform_vectors(self.coords)
            else:
                a = self.coords
        else:
            model = self.axis_model
            a = model.scene_position.transform_vector(model.direction)
        if normalize:
            from chimerax import geometry
            a = geometry.normalize_vector(a)
        return a

    def base_point(self):
        if self.atoms is not None:
            return self.atoms[0].scene_coord
        if self.axis_model is not None:
            model = self.axis_model
            return model.scene_position * model.center
        return None


class CenterArg(Annotation):
    '''Annotation for a center point that can be 3 floats or objects.'''
    name = 'center point'

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % CenterArg.name)
        c = None

        # 3 comma-separated numbers
        if c is None:
            try:
                coords, atext, rest = Float3Arg.parse(text, session)
            except Exception:
                pass
            else:
                c = Center(coords)

        # Center at camera
        if c is None:
            try:
                cam, atext, rest = EnumOf(['camera']).parse(text, session)
            except Exception:
                pass
            else:
                c = Center(coords=session.main_view.camera.position.origin())

        # Center at center of rotation
        if c is None:
            try:
                cam, atext, rest = EnumOf(['cofr']).parse(text, session)
            except Exception:
                pass
            else:
                c = Center(coords=session.main_view.center_of_rotation)

        # Objects
        if c is None:
            try:
                obj, atext, rest = ObjectsArg.parse(text, session)
            except Exception:
                pass
            else:
                if obj.empty():
                    raise AnnotationError('Center argument no objects specified')
                elif obj.bounds() is None:
                    raise AnnotationError('Center argument has no object bounds')
                c = Center(objects=obj)

        if c is None:
            raise AnnotationError('Expected 3 floats or object specifier')

        return c, atext, rest


class Center:

    def __init__(self, coords=None, objects=None):
        if coords is not None:
            from numpy import array, float32
            coords = array(coords, float32)
        self.coords = coords
        self.objects = objects

    def scene_coordinates(self, coordinate_system=None):
        obj = self.objects
        if obj is not None:
            c = obj.bounds().center()
        elif coordinate_system is not None:
            c = coordinate_system * self.coords
        else:
            c = self.coords
        return c


class CoordSysArg(ModelArg):
    """
    Annotation for coordinate system for AxisArg and CenterArg
    when specified as tuples of numbers.  Coordinate system is
    specified as a Model specifier.
    """
    name = "a coordinate-system"

    @classmethod
    def parse(cls, text, session):
        m, text, rest = super().parse(text, session)
        return m.position, text, rest

    @classmethod
    def unparse(cls, value, session):
        raise NotImplementedError("No longer know model for %s" % cls.__name__)


class PlaceArg(Annotation):
    """
    Annotation for positioning matrix as 12 floats
    defining a 3 row, 4 column matrix where the first
    3 columns are x,y,z coordinate axes, and the last column
    is the origin.
    """
    name = "a position"

    @staticmethod
    def parse(text, session):
        if not text:
            raise AnnotationError("Expected %s" % PlaceArg.name)
        token, text, rest = next_token(text)
        p = PlaceArg.parse_place(token.split(','))
        return p, text, rest

    @staticmethod
    def parse_place(fields):
        if len(fields) != 12:
            raise AnnotationError("Expected 12 comma-separated values")
        try:
            values = [float(x) for x in fields]
        except ValueError:
            raise AnnotationError("Require numeric values")
        from chimerax.geometry import Place
        p = Place(matrix=(values[0:4], values[4:8], values[8:12]))
        return p


def _remove_child_models(models):
    s = set(models)
    for m in models:
        for c in m.child_models():
            s.discard(c)
    return tuple(m for m in models if m in s)


def make_converter(annotation):
    """Supported API.  Use any annotation to convert text to annotation's type

    :param annotation: the annotation to use
    :raises AnnotationError: if unable to convert text

    For example:

        from chimerax.core.commands import make_converter, ColorArg
        convert_to_color = make_converter(ColorArg)
        color = convert_to_color(session, "red")
    """
    def use_annotation(session, text):
        value, _, rest = annotation.parse(text, session)
        if rest:
            offset = len(text) - len(rest)
            raise AnnotationError("extra text at end of %s" % discard_article(
                annotation.name), offset)
        return value
    use_annotation.__doc__ = "Convert text to %s" % annotation.name
    return use_annotation


# TODO: @deprecated(version='0.91', reason='Use make_converter() instead')
def as_parser(annotation):
    """Deprecated API.  Use make_converter() instead"""
    return make_converter(annotation)


def quote_path_if_necessary(path):
    """quote a string

    So a :py:class:`FileNameArg` (or subclass) treats it like a singe value.
    String escape sequences are not recognized in paths, so there no way to
    quote a path so that both single and double quotes that would need
    quoting.
    """
    if not path:
        return '""'
    has_single_quote = path[0] == "'" or _internal_single_quote.search(path) is not None
    has_double_quote = path[0] == '"' or _internal_double_quote.search(path) is not None
    if has_single_quote and has_double_quote:
        raise ValueError("Unable to quote path: %r" % path)
    if has_single_quote:
        return '"%s"' % path
    if has_double_quote:
        return '"%s"' % path
    has_whitespace = _internal_whitespace.search(path) is not None
    if has_whitespace:
        return '"%s"' % path
    return path


def quote_if_necessary(s, additional_special_map={}):
    """quote a string

    So :py:class:`StringArg` treats it like a single value"""
    if not s:
        return '""'
    import unicodedata
    has_single_quote = s[0] == "'" or _internal_single_quote.search(s) is not None
    has_double_quote = s[0] == '"' or _internal_double_quote.search(s) is not None
    has_special = False
    use_single_quote = not has_single_quote and has_double_quote
    special_map = {
        '\a': '\\a',
        '\b': '\\b',
        '\f': '\\f',
        '\n': '\\n',
        '\r': '\\r',
        '\t': '\\t',
        '\v': '\\v',
        '\\': '\\\\',
        ';': ';',
        ' ': ' ',
    }
    special_map.update(additional_special_map)

    result = []
    for ch in s:
        i = ord(ch)
        if ch == "'":
            result.append(ch)
        elif ch == '"':
            if use_single_quote:
                result.append('"')
            elif has_double_quote:
                result.append('\\"')
            else:
                result.append('"')
        elif ch in special_map:
            has_special = True
            result.append(special_map[ch])
        elif i < 32:
            has_special = True
            result.append('\\x%02x' % i)
        elif ch.strip() == '':
            # non-space and non-newline spaces
            has_special = True
            result.append('\\N{%s}' % unicodedata.name(ch))
        else:
            result.append(ch)
    if has_single_quote or has_double_quote or has_special:
        if use_single_quote:
            return "'%s'" % ''.join(result)
        else:
            return '"%s"' % ''.join(result)
    return ''.join(result)


_escape_table = {
    "'": "'",
    '"': '"',
    '\\': '\\',
    '\n': '',
    'a': '\a',  # alarm
    'b': '\b',  # backspace
    'f': '\f',  # formfeed
    'n': '\n',  # newline
    'r': '\r',  # return
    't': '\t',  # tab
    'v': '\v',  # vertical tab
}


def unescape(text):
    """Replace backslash escape sequences with actual character.

    :param text: the input text
    :returns: the processed text

    Follows Python's :ref:`string literal <python:literals>` syntax
    for escape sequences."""
    return unescape_with_index_map(text)[0]


def unescape_with_index_map(text):
    """Replace backslash escape sequences with actual character.

    :param text: the input text
    :returns: the processed text and index map from processed to input text

    Follows Python's :ref:`string literal <python:literals>` syntax
    for escape sequences."""
    # standard Python backslashes including \N{unicode name}
    start = 0
    index_map = list(range(len(text)))
    while start < len(text):
        index = text.find('\\', start)
        if index == -1:
            break
        if index == len(text) - 1:
            break
        escaped = text[index + 1]
        if escaped in _escape_table:
            text = text[:index] + _escape_table[escaped] + text[index + 2:]
            # Assumes that replacement is a single character
            index_map = index_map[:index] + index_map[index + 1:]
            start = index + 1
        elif escaped in '01234567':
            # up to 3 octal digits
            for count in range(2, 5):
                if text[index + count] not in '01234567':
                    break
            try:
                char = chr(int(text[index + 1: index + count], 8))
                text = text[:index] + char + text[index + count:]
                index_map = index_map[:index] + index_map[index + count - 1:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'x':
            # 2 hex digits
            try:
                char = chr(int(text[index + 2: index + 4], 16))
                text = text[:index] + char + text[index + 4:]
                index_map = index_map[:index] + index_map[index + 3:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'u':
            # 4 hex digits
            try:
                char = chr(int(text[index + 2: index + 6], 16))
                text = text[:index] + char + text[index + 6:]
                index_map = index_map[:index] + index_map[index + 5:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'U':
            # 8 hex digits
            try:
                char = chr(int(text[index + 2: index + 10], 16))
                text = text[:index] + char + text[index + 10:]
                index_map = index_map[:index] + index_map[index + 9:]
            except ValueError:
                pass
            start = index + 1
        elif escaped == 'N':
            # named unicode character
            if len(text) < index + 2 or text[index + 2] != '{':
                start = index + 1
                continue
            end = text.find('}', index + 3)
            if end > 0:
                import unicodedata
                char_name = text[index + 3:end]
                try:
                    char = unicodedata.lookup(char_name)
                    text = text[:index] + char + text[end + 1:]
                    index_map = index_map[:index] + index_map[end:]
                except KeyError:
                    pass
            start = index + 1
        else:
            # leave backslash in text like Python
            start = index + 1
    return text, index_map


def next_token(text, convert=False):
    """
    Extract next, optionally quoted, token from given text.

    :param text: text to parse without leading whitespace
    :param convert: convert escape sequences to characters
    :returns: a 3-tuple of first argument in text, the actual text used,
              and the rest of the text.

    If the token is quoted, then the text between the quotes is returned.
    End quotes must be followed by whitespace or the end of the line.
    If convert is True, then escape sequences in the token are converted
    to the actual character.  The escape sequences are the same as Python's
    as documented in :ref:`literals`.
    """
    assert text and not text[0].isspace()
    if text[0] == '"':
        m = _double_quote.match(text)
        if not m:
            raise AnnotationError("incomplete quoted text")
        end = m.end()
        if text[end - 1].isspace():
            end -= 1
        token = text[1:end - 1]
    elif text[0] == "'":
        m = _single_quote.match(text)
        if not m:
            raise AnnotationError("incomplete quoted text")
        end = m.end()
        if text[end - 1].isspace():
            end -= 1
        token = text[1:end - 1]
    elif text[0] == ';':
        return ';', ';', text[1:]
    else:
        m = _normal_token.match(text)
        end = m.end()
        token = text[0:end]
    if convert:
        token = unescape(token)
    return token, text[:end], text[end:]


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
        elif text[start] == "'":
            m = _single_quote.match(text, start)
            if m:
                start = m.end()
            else:
                start = size
                raise AnnotationError("incomplete quoted text")
                break
        elif text[start] == ';':
            break
        else:
            m = _normal_token.match(text, start)
            start = m.end()
    return text[:start], text[start:]


class RestOfLine(Annotation):
    """Return the rest of the line up to a semicolon"""
    name = "the rest of line"

    @staticmethod
    def parse(text, session):
        m = _whitespace.match(text)
        start = m.end()
        leading = text[:start]
        text, rest = _upto_semicolon(text[start:])
        return text, leading + text, rest

    @staticmethod
    def unparse(value, session=None):
        assert isinstance(value, str)
        return value


class WholeRestOfLine(Annotation):
    """Return the whole rest of the line including semicolons"""
    name = "the rest of line"

    @staticmethod
    def parse(text, session):
        return text, text, ''

    @staticmethod
    def unparse(value, session=None):
        assert isinstance(value, str)
        return value


Bool2Arg = TupleOf(BoolArg, 2)
Bool3Arg = TupleOf(BoolArg, 3)
IntsArg = ListOf(IntArg)
Int2Arg = TupleOf(IntArg, 2)
Int3Arg = TupleOf(IntArg, 3)
FloatsArg = ListOf(FloatArg)
Float2Arg = TupleOf(FloatArg, 2)
Float3Arg = TupleOf(FloatArg, 3)
NonNegativeIntArg = Bounded(IntArg, min=0, name="an integer >= 0")
PositiveIntArg = Bounded(IntArg, min=1, name="an integer >= 1")
NonNegativeFloatArg = Bounded(FloatArg, min=0, name="a number >= 0")
PositiveFloatArg = Bounded(FloatArg, min=0, inclusive=False, name="a number > 0")
ModelIdArg = DottedTupleOf(PositiveIntArg, name="a model id", prefix='#')


class Postcondition(metaclass=abc.ABCMeta):
    """Base class for postconditions"""

    def check(self, kw_args):
        """Assert arguments match postcondition

        :param kw_args: dictionary of arguments that will be passed
            to command callback function.
        :returns: True if arguments are consistent
        """
        raise NotImplementedError

    def error_message(self):
        """Appropriate error message if check fails.

        :returns: error message
        """
        raise NotImplementedError


class Limited(Postcondition):
    """Bounded numerical values postcondition

    Limited(name, min=None, max=None) -> Postcondition

    :param arg_name: name of argument to check
    :param min: optional inclusive lower bound
    :param max: optional inclusive upper bound

    If possible, use the Bounded annotation because the location of
    the error is the beginning of the argument, not the end of the line.
    """

    __slots__ = ['arg_name', 'min', 'max']

    def __init__(self, arg_name, min=None, max=None):
        self.arg_name = arg_name
        self.min = min
        self.max = max

    def check(self, kw_args):
        arg = kw_args.get(self.arg_name, None)
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
        message = "Invalid argument %s: " % dq_repr(self.arg_name)
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


def _check_autocomplete(word, mapping, name):
    # Already have tool. vs toolshed, expect more in the future as external
    # developers add commands.
    return
    # This is a primary debugging aid for developers,
    # but it warns about existing abbreviated commands from changing
    # what command they correspond to.
    if word not in mapping:
        for key in mapping:
            if key.startswith(word) and key != word:
                if word != name:
                    raise ValueError("'%s' in '%s' is a prefix of an existing command '%s'" % (word, name, key))
                else:
                    raise ValueError("'%s' is a prefix of an existing command '%s'" % (word, key))


class CmdDesc:
    """Describe command arguments.

    :param required: required positional arguments sequence
    :param optional: optional positional arguments sequence
    :param keyword: keyword arguments sequence
    :param required_arguments: sequence of argument names that must be given
    :param non_keyword: sequence of optional arguments that cannot be keywords
    :param hidden: sequence of keyword arguments that should be omitted from usage
    :param url: URL to help page
    :param synopsis: one line description

    .. data: function

        function that implements command

    Each :param required:, :param optional:, :param keyword: sequence
    contains 2-tuples with the argument name and a type annotation.
    The command line parser uses the :param optional: argument names as
    additional keyword arguments.
    :param required_arguments: are for Python function arguments that
    don't have default values, but should be given on the command line
    (typically *keyword* arguments, but could be used for syntactically
    *optional* arguments).
    """
    __slots__ = [
        '_required', '_optional', '_keyword', '_keyword_map',
        '_required_arguments', '_postconditions', '_function',
        '_hidden', '_can_return_json', 'url', 'synopsis', 'self_logging'
    ]

    def __init__(self, required=(), optional=(), keyword=(),
                 postconditions=(), required_arguments=(),
                 non_keyword=(), hidden=(), url=None, synopsis=None, self_logging=False):
        self._required = OrderedDict(required)
        self._optional = OrderedDict(optional)
        self._keyword = dict(keyword)
        optional_keywords = [i for i in self._optional.items()
                             if i[0] not in non_keyword]
        self._keyword.update(optional_keywords)
        self._hidden = set(hidden)
        # keyword_map is what would user would type

        def fill_keyword_map(n):
            kw, cnt = _user_kw_cnt(n)
            return kw, (n, cnt)
        self._keyword_map = {}
        self._keyword_map.update(fill_keyword_map(n) for n in self._keyword)
        self._postconditions = postconditions
        self._required_arguments = required_arguments
        self.url = url
        self.synopsis = synopsis
        self.self_logging = self_logging
        self._function = None
        self._can_return_json = False

    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, function):
        """Set the function to call when the command matches.

        Double check that all function arguments, that do not
        have default values, are 'required'.
        """
        if function is not None:
            if self._function:
                raise ValueError("Can not reuse CmdDesc instances")
            import inspect
            empty = inspect.Parameter.empty
            var_positional = inspect.Parameter.VAR_POSITIONAL
            var_keyword = inspect.Parameter.VAR_KEYWORD
            signature = inspect.signature(function)
            sig_params = signature.parameters
            self._can_return_json = 'return_json' in sig_params
            params = list(sig_params.values())
            if len(params) < 1 or params[0].name != "session":
                raise ValueError('Missing initial "session" argument')
            for p in params[1:]:
                if (p.default != empty or p.name in self._required or
                        p.name in self._required_arguments or
                        p.kind in (var_positional, var_keyword)):
                    continue
                raise ValueError("Wrong function or '%s' argument must be "
                                 "required or have a default value" % p.name)
        self._function = function

    @property
    def can_return_json(self):
        return self._can_return_json

    def copy(self):
        """Return a copy suitable for use with another function."""
        import copy
        ci = copy.copy(self)
        ci._function = None
        return ci

    def is_alias(self):
        return isinstance(self.function, Alias)


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

        from chimerax.core.commands import cli

        def lazy_reg():
            import module
            cli.register('cmd subcmd1', module.subcmd1_desc, module.subcmd1)
            cli.register('cmd subcmd2', module.subcmd2_desc, module.subcmd2)

        cli.delay_registration('cmd', lazy_reg)
    """
    register(name, None, _Defer(proxy_function), logger=logger)


# _command_info.commands is a map of command words to command information.  Except when
# it is a multiword command name, then the preliminary words map to
# dictionaries that map to the command information.
# An OrderedDict is used so for autocompletion, the prefix of the first
# registered command with that prefix is used.


class _WordInfo:
    # Internal information about a word in a command

    def __init__(self, registry, cmd_desc=None):
        self.registry = registry
        self.cmd_desc = cmd_desc
        self.subcommands = OrderedDict()   # { 'word': _WordInfo() }
        self.parent = None

    def has_command(self):
        return self.cmd_desc is not None

    def has_subcommands(self):
        return len(self.subcommands) > 0

    def is_alias(self):
        return (isinstance(self.cmd_desc, CmdDesc) and
                isinstance(self.cmd_desc.function, Alias))

    def is_user_alias(self):
        return (isinstance(self.cmd_desc, CmdDesc) and
                isinstance(self.cmd_desc.function, Alias) and
                self.cmd_desc.function.user_generated)

    def alias(self):
        if not self.is_alias():
            raise RuntimeError('not an alias')
        return self.cmd_desc.function

    def is_deferred(self):
        return isinstance(self.cmd_desc, _Defer)

    def lazy_register(self, cmd_name):
        deferred = self.cmd_desc
        assert(isinstance(deferred, _Defer))
        try:
            deferred.call()
        except Exception as e:
            raise RuntimeError("delayed command registration for %r failed (%s)" % (cmd_name, e))
        if self.cmd_desc is None and not self.has_subcommands():
            raise RuntimeError("delayed command registration for %r didn't register the command" % cmd_name)
        if isinstance(self.cmd_desc, _Defer):
            self.cmd_desc = None  # prevent reuse

    def add_subcommand(self, word, name, cmd_desc=None, *, logger=None):
        try:
            _check_autocomplete(word, self.subcommands, name)
        except ValueError as e:
            if isinstance(cmd_desc, CmdDesc) and isinstance(cmd_desc.function, Alias):
                if logger is not None:
                    logger.warning("Alias %s hides existing command" % dq_repr(name))
            elif logger is not None:
                logger.warning(str(e))
        if word not in self.subcommands:
            w = self.subcommands[word] = _WordInfo(self.registry, cmd_desc)
            w.parent = self
            return
        # command word previously registered
        if cmd_desc is None:
            return
        word_info = self.subcommands[word]

        if isinstance(cmd_desc, CmdDesc) and isinstance(cmd_desc.function, Alias) and cmd_desc.function.user_generated:
            # adding a user-generated alias
            if word_info.is_user_alias():
                # replacing user alias with another user alias
                word_info.cmd_desc = cmd_desc
            else:
                # only save/restore "system" version of command
                self.registry.aliased_commands[name] = word_info
                self.subcommands[word] = _WordInfo(self.registry, cmd_desc)
                if logger is not None:
                    logger.info("FYI: alias is hiding existing command" %
                                dq_repr(name))
        elif word_info.is_user_alias():
            # command is aliased, but new one isn't, so replaced saved version
            if name in self.registry.aliased_commands:
                self.registry.aliased_commands[name].cmd_desc = cmd_desc
            else:
                self.registry.aliased_commands[name] = _WordInfo(self.registry, cmd_desc)
        else:
            if logger is not None and word_info.has_command() and not word_info.is_deferred():
                logger.info("FYI: command is replacing existing command: %s" %
                            dq_repr(name))
            word_info.cmd_desc = cmd_desc


class RegisteredCommandInfo:

    def __init__(self):
        # keep track of commands
        self.commands = _WordInfo(self)
        # keep track of commands that have been overridden by an alias
        self.aliased_commands = {}  # { name: _WordInfo instance }


_command_info = RegisteredCommandInfo()
# keep track of available commands
_available_commands = None

def register(name: str, cmd_desc: CmdDesc = (), function: Callable[..., Any] = None, *, logger=None, registry=None) -> Callable[..., Any]:
    """register function that implements command

    :param name: the name of the command and may include spaces.
    :param cmd_desc: information about the command, either an
        instance of :py:class:`CmdDesc`, or the tuple with CmdDesc
        parameters.
    :param function: the callback function.
    :param logger: optional logger

    If the function is None, then it assumed that :py:func:`register`
    is being used as a decorator.

    registry is a RegisteredCommandInfo instance and is only used if you want
    your own separate registry of commands, to be parsed/executed on text you provide.
    One usage scenario is a set of "subcommands" organized under an "umbrella" command, e.g.:

    sequence viewer <viewer-id> command-to-viewer

    The 'command-to-viewer' commands could have their own separate registry.


    To delay introspecting the function until it is actually used,
    register using the :py:func:`delay_registration` function.
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
    if cmd_desc is not None and cmd_desc.url is None:
        url = _get_help_url(words)
        if url is not None:
            cmd_desc.url = url
    if registry is None:
        _parent_info = _command_info.commands
    else:
        _parent_info = registry.commands
    for word in words[:-1]:
        _parent_info.add_subcommand(word, name, logger=logger)
        _parent_info = _parent_info.subcommands[word]

    if isinstance(function, _Defer):
        cmd_desc = function
    else:
        cmd_desc.function = function
        if cmd_desc.synopsis is None:
            msg = 'Command "%s" is missing a synopsis' % name
            if logger is None:
                print(msg)
            else:
                logger.warning(msg)
    _parent_info.add_subcommand(words[-1], name, cmd_desc, logger=logger)
    return function     # needed when used as a decorator


def _get_help_url(words):
    # return help: URLs so links work in log as well as help viewer
    import os
    from urllib.parse import urlunparse
    cname = words[0]
    if cname.startswith('~'):
        cname = cname[1:]
        frag = '-'.join(words)
    else:
        frag = '-'.join(words[1:])
    from .. import toolshed
    help_directories = toolshed.get_help_directories()
    cmd_subpath = os.path.join('user', 'commands', '%s.html' % cname)
    for hd in help_directories:
        cpath = os.path.join(hd, cmd_subpath)
        if os.path.exists(cpath):
            return urlunparse(('help', '', "user/commands/%s.html" % cname, '', '', frag))
    return None


def deregister(name, *, is_user_alias=False, registry=None):
    """Remove existing command and subcommands

    :param name: the name of the command

    If the command was an alias, the previous version is restored"""
    # none of the exceptions below should happen
    words = name.split()
    name = ' '.join(words)  # canonicalize
    if registry is None:
        registry = _command_info
    _parent_info = registry.commands
    for word in words:
        word_info = _parent_info.subcommands.get(word, None)
        if word_info is None:
            if is_user_alias:
                raise UserError('No alias named %s exists' % dq_repr(name))
            raise RuntimeError('unregistering unknown command: "%s"' % name)
        _parent_info = word_info
    if is_user_alias and not _parent_info.is_user_alias():
        raise UserError('%s is not a user alias' % dq_repr(name))

    if word_info.has_subcommands():
        for subword in list(word_info.subcommands.keys()):
            deregister("%s %s" % (name, subword), registry=registry)

    hidden_word = registry.aliased_commands.get(name, None)
    if hidden_word:
        _parent_info = hidden_word.parent
        _parent_info.subcommands[word] = hidden_word
        del registry.aliased_commands[name]
    else:
        # allow command to be reregistered with same cmd_desc
        if word_info.cmd_desc and not word_info.is_deferred():
            word_info.cmd_desc.function = None
        # remove association between cmd_desc and word
        word_info.cmd_desc = None
        _parent_info = word_info.parent
        assert len(word_info.subcommands) == 0
        del _parent_info.subcommands[word]


def register_available(*args, **kw):
    return register(*args, registry=_available_commands, **kw)


def clear_available():
    global _available_commands
    _available_commands = None


def _compute_available_commands(session):
    global _available_commands
    if _available_commands is not None:
        return
    from .. import toolshed
    _available_commands = RegisteredCommandInfo()
    ts = toolshed.get_toolshed()
    ts.register_available_commands(session.logger)


class _FakeSession:
    pass


class Command:
    """Keep track of (partially) typed command with possible completions

    :param session: the session to run the command in (may be None for testing)

    .. data: current_text

        The expanded version of the command.

    .. data: amount_parsed

        Amount of current text that has been successfully parsed.

    .. data: start

        Start of current command in current_text

    .. data: completions

        Possible command or keyword completions if given an incomplete command.
        The first one will be used if the command is executed.

    .. data: completion_prefix

        Partial word used for command completions.
    """
    # nested = 0  # DEBUG nested aliases

    def __init__(self, session, *, registry=None):
        import weakref
        if session is None:
            session = _FakeSession()
        self._session = weakref.ref(session)
        self._reset()
        self.registry = _command_info if registry is None else registry

    def _reset(self):
        self.current_text = ""
        self.amount_parsed = 0
        self.completion_prefix = ""
        self.completions = []
        self._multiple = []
        self._error = ""
        self._ci = None
        self.command_name = None
        self._kw_args = {}

    def _replace(self, chars, replacement):
        # insert replacement (quotes are already in replacement text)
        i = len(chars)
        j = self.amount_parsed
        t = self.current_text
        self.current_text = t[0:j] + replacement + t[i + j:]
        return len(replacement)

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

    def _find_command_name(self, final=True, no_aliases=False, used_aliases=None, parent_info=None):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, sets self._ci
        self._error = "Missing command"
        self.word_info = None  # filled in when partial command is matched
        if parent_info is None:
            parent_info = self.registry.commands
        cmd_name = None
        self.start = self.amount_parsed
        start = self.start
        while 1:
            m = _whitespace.match(self.current_text, self.amount_parsed)
            cur_end = m.end()
            text = self.current_text[cur_end:]
            if not text:
                if self.amount_parsed == start:
                    self._error = ''
                    self.amount_parsed = cur_end
                self.word_info = parent_info
                self.command_name = cmd_name
                return
            if self.amount_parsed == start:
                self.start = cur_end
            if text.startswith('#') and self.amount_parsed == start:
                self._error = ''
                self.amount_parsed = len(self.current_text)
                self.word_info = None
                self.command_name = text
                return
            if text.startswith(';'):
                if cmd_name is None:
                    self._error = None
                self.amount_parsed = cur_end
                self.word_info = parent_info
                self.command_name = cmd_name
                return
            if _debugging:
                orig_text = text
            try:
                word, chars, text = next_token(text)
            except AnnotationError as err:
                self._error = str(err)
                return
            if _debugging:
                print('cmd next_token(%r) -> %r %r %r' % (
                    orig_text, word, chars, text))
            what = parent_info.subcommands.get(word, None)
            if what is None:
                self.completion_prefix = word
                self.completions = [
                    x for x in parent_info.subcommands if x.startswith(word)]
                if word and (final or len(text) > len(chars)) \
                        and self.completions:
                    # If final version of text, or if there
                    # is following text, make best guess,
                    # and retry
                    self.amount_parsed = cur_end
                    c = self.completions[0]
                    self._replace(chars, c)
                    text = self.current_text[self.amount_parsed:]
                    continue
                if word and self._ci is None:
                    self._error = "Unknown command: %s" % self.current_text[self.start:]
                return
            self.amount_parsed = cur_end
            self._ci = None
            self.word_info = what
            self.command_name = None
            self.amount_parsed += len(chars)
            cmd_name = self.current_text[self.start:self.amount_parsed]
            cmd_name = ' '.join(cmd_name.split())   # canonicalize
            if what.is_deferred():
                what.lazy_register(cmd_name)
            if what.cmd_desc is not None:
                if no_aliases:
                    if what.is_alias():
                        if cmd_name not in self.registry.aliased_commands:
                            self._error = 'Alias did not hide a command'
                            return
                        what = self.registry.aliased_commands[cmd_name]
                        if what.cmd_desc is None:
                            parent_info = what
                            continue
                elif (used_aliases is not None and
                        what.is_alias() and
                        cmd_name in used_aliases):
                    if cmd_name not in self.registry.aliased_commands:
                        self._error = "Aliasing loop detected"
                        return
                    what = self.registry.aliased_commands[cmd_name]
                    if what.cmd_desc is None:
                        parent_info = what
                        continue
                self._ci = what.cmd_desc
                self.command_name = cmd_name
                self._error = ''
            parent_info = what

            if not parent_info.has_subcommands():
                return
            # word might be part of multiword command name
            if parent_info.cmd_desc is None:
                self.command_name = self.current_text[self.start:self.amount_parsed]
                self._error = "Incomplete command: %s" % self.command_name

    def _process_positional_arguments(self):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, updates self._kw_args
        # for better error messages return:
        #   (last successful annotation, failed optional annotation)
        session = self._session()  # resolve back reference
        text = self.current_text[self.amount_parsed:]
        positional = self._ci._required.copy()
        positional.update(self._ci._optional)
        self.completion_prefix = ''
        self.completions = []
        last_anno = None
        for kw_name, anno in positional.items():
            if kw_name in self._ci._optional:
                self._error = ""
            else:
                if isinstance(kw_name, int):
                    # alias argument position
                    required = "%s required" % ordinal(kw_name)
                else:
                    required = 'required "%s"' % user_kw(kw_name)
                self._error = 'Missing %s positional argument' % required
            text = self._skip_white_space(text)
            if kw_name in self._ci._optional and self._start_of_keywords(text):
                return last_anno, None
            try:
                value, text = self._parse_arg(anno, text, session, False)
                kwn = '%s_' % kw_name if is_python_keyword(kw_name) else kw_name
                self._kw_args[kwn] = value
                self._error = ""
                last_anno = anno
                if hasattr(anno, 'allow_repeat') and anno.allow_repeat:
                    expand = anno.allow_repeat == "expand"
                    if expand and isinstance(value, list):
                        self._kw_args[kwn] = values = value
                    else:
                        self._kw_args[kwn] = values = [value]
                    while True:
                        text = self._skip_white_space(text)
                        if self._start_of_keywords(text):
                            return last_anno, None
                        try:
                            value, text = self._parse_arg(anno, text, session, False)
                        except Exception:
                            break
                        if expand and isinstance(value, list):
                            values.extend(value)
                        else:
                            values.append(value)
            except ValueError as err:
                if isinstance(err, AnnotationError) and err.offset:
                    # We got an error with an offset, that means that an
                    # argument was partially matched, so assume that is the
                    # error the user wants to see.
                    self.amount_parsed += err.offset
                    if isinstance(kw_name, int):
                        arg_name = ordinal(kw_name)
                    else:
                        arg_name = '"%s"' % user_kw(kw_name)
                    self._error = 'Missing or invalid %s argument: %s' % (arg_name, err)
                    return None, None
                if kw_name in self._ci._required:
                    if isinstance(kw_name, int):
                        arg_name = ordinal(kw_name)
                    else:
                        arg_name = '"%s"' % user_kw(kw_name)
                    self._error = 'Missing or invalid %s argument: %s' % (arg_name, err)
                    return None, None
                # optional and wrong type, try as keyword
                return last_anno, anno
        return last_anno, None

    def _skip_white_space(self, text):
        m = _whitespace.match(text)
        start = m.end()
        if start:
            self.amount_parsed += start
            text = text[start:]
        if text and text[0] == ';':
            text = ''
        return text

    def _start_of_keywords(self, text):
        # check if next token matches a keyword
        if not text:
            return True
        try:
            _, tmp, _ = next_token(text)
        except AnnotationError:
            return False
        if not tmp:
            return True
        if tmp[0].isalpha():
            # Don't change case of what user types.  Fixes "show O".
            tmp = user_kw(tmp)
            if (any(kw.startswith(tmp) for kw in self._ci._keyword_map) or
                    any(kw.casefold().startswith(tmp) for kw in self._ci._keyword_map)):
                return True

        return False

    def _process_keyword_arguments(self, final, prev_annos):
        # side effects:
        #   updates amount_parsed
        #   updates possible completions
        #   if successful, updates self._kw_args
        session = self._session()  # resolve back reference
        m = _whitespace.match(self.current_text, self.amount_parsed)
        self.amount_parsed = m.end()
        text = self.current_text[self.amount_parsed:]
        if not text:
            return
        while 1:
            if _debugging:
                orig_text = text
            try:
                word, chars, text = next_token(text)
            except AnnotationError as err:
                self._error = str(err)
                return
            if _debugging:
                print('key next_token(%r) -> %r %r %r' % (
                    orig_text, word, chars, text))
            if not word or word == ';':
                break

            arg_name = user_kw(word)
            if arg_name not in self._ci._keyword_map:
                self.completion_prefix = word
                kw_map = self._ci._keyword_map
                # Don't change case of what user types.
                completions = [(kw_map[kw][1], kw) for kw in kw_map
                               if kw.startswith(arg_name) or kw.casefold().startswith(arg_name)]
                completions.sort()
                if (final or len(text) > len(chars)) and completions:
                    # require shortened keywords to be unambiguous
                    if len(completions) == 1:
                        unambiguous = True
                    elif len(completions[0][1]) == len(arg_name):
                        unambiguous = True
                    elif 1 == len([cnt for cnt, kw in completions if cnt == completions[0][0]]):
                        unambiguous = True
                    else:
                        unambiguous = False
                    if unambiguous:
                        c = completions[0][1]
                        self._replace(chars, c)
                        text = self.current_text[self.amount_parsed:]
                        self.completions = []
                        continue
                    self.completions = list(c[1] for c in completions)
                    self._error = "Expected keyword " + commas('"%s"' % x for x in self.completions)
                    return
                expected = []
                if isinstance(prev_annos[0], Aggregate):
                    expected.append("'%s'" % prev_annos[0].separator)
                if prev_annos[1] is not None:
                    expected.append(prev_annos[1].name)
                if len(self._ci._keyword_map) > 0:
                    expected.append("a keyword")
                else:
                    expected.append("fewer arguments")
                self._error = "Expected " + commas(expected)
                return
            self._replace(chars, arg_name)
            self.amount_parsed += len(arg_name)
            m = _whitespace.match(text)
            start = m.end()
            if start:
                self.amount_parsed += start
                text = text[start:]

            kw_name = self._ci._keyword_map[arg_name][0]
            anno = self._ci._keyword[kw_name]
            if not text and anno != NoArg:
                self._error = 'Missing "%s" keyword\'s argument' % user_kw(kw_name)
                break

            self.completion_prefix = ''
            self.completions = []
            try:
                value, text = self._parse_arg(anno, text, session, final)
                kwn = '%s_' % kw_name if is_python_keyword(kw_name) else kw_name
                if hasattr(anno, 'allow_repeat') and anno.allow_repeat:
                    if kwn in self._kw_args:
                        self._kw_args[kwn].append(value)
                    else:
                        self._kw_args[kwn] = [value]
                else:
                    if kwn in self._kw_args:
                        self._error = 'Repeated keyword argument "%s"' % user_kw(kw_name)
                        return
                    self._kw_args[kwn] = value
                prev_annos = (anno, None)
            except ValueError as err:
                if isinstance(err, AnnotationError) and err.offset is not None:
                    self.amount_parsed += err.offset
                self._error = 'Invalid "%s" argument: %s' % (
                    user_kw(kw_name), err)
                return
            m = _whitespace.match(text)
            start = m.end()
            if start:
                self.amount_parsed += start
                text = text[start:]
            if not text:
                break

    def run(self, text, *, log=True, log_only=False, return_json=False, _used_aliases=None):
        """Parse and execute commands in the text

        :param text: The text to be parsed.
        :param log: True (default) if commands are logged.

        There are a couple side effects:

        * The automatically completed text is put in self.current_text.
        * Possible completions are in self.completions.
        * The prefix of the completions is in self.completion_prefix.
        """
        session = self._session()  # resolve back reference
        if isinstance(session, _FakeSession):
            log = False

        self._reset()
        self.current_text = text
        final = True    # TODO: support partial parsing for cmd/arg completion
        results = []
        while True:
            self._find_command_name(final, used_aliases=_used_aliases)
            if self._error:
                save_error = self._error
                if self.registry == _command_info:
                    # See if this command is available in the toolshed
                    self._error = ""
                    _compute_available_commands(session)
                    self._find_command_name(final, used_aliases=_used_aliases,
                                            parent_info=_available_commands.commands)
                if self._error or not self._ci:
                    # Nope, give the original error message
                    self._error = save_error
                    if log:
                        self.log_parse_error()
                    raise UserError(self._error)
            if not self._ci:
                if len(self.current_text) > self.amount_parsed and self.current_text[self.amount_parsed] == ';':
                    # allow for leading and empty semicolon-separated commands
                    self.amount_parsed += 1  # skip semicolon
                    continue
                return results
            prev_annos = self._process_positional_arguments()
            if self._error:
                if log:
                    self.log_parse_error()
                raise UserError(self._error)
            self._process_keyword_arguments(final, prev_annos)
            if self._error:
                if log:
                    self.log_parse_error()
                raise UserError(self._error)
            missing = [kw for kw in self._ci._required_arguments if kw not in self._kw_args]
            if missing:
                arg_names = ['"%s"' % m for m in missing]
                msg = commas(arg_names, 'and')
                noun = plural_form(arg_names, 'argument')
                self._error = "Missing required %s %s" % (msg, noun)
                if log:
                    self.log_parse_error()
                raise UserError(self._error)
            for cond in self._ci._postconditions:
                if not cond.check(self._kw_args):
                    self._error = cond.error_message()
                    if log:
                        self.log_parse_error()
                    raise UserError(self._error)

            if not final:
                return results

            ci = self._ci
            kw_args = self._kw_args
            really_log = log and _used_aliases is None and not self._ci.self_logging
            if really_log or log_only:
                self.log()
            cmd_text = self.current_text[self.start:self.amount_parsed]
            with command_trigger(session, really_log, cmd_text):
                from chimerax.core.errors import CancelOperation
                if not isinstance(ci.function, Alias):
                    if not log_only:
                        if ci.can_return_json:
                            kw_args['return_json'] = return_json
                        if self._ci.self_logging:
                            kw_args['log'] = log
                        try:
                            result = ci.function(session, **kw_args)
                        except CancelOperation:
                            if not self._ci.self_logging:
                                session.logger.info("Command cancelled by user")
                            raise
                        results.append(result)
                else:
                    arg_names = [k for k in kw_args.keys() if isinstance(k, int)]
                    arg_names.sort()
                    args = [kw_args[k] for k in arg_names]
                    if 'optional' in kw_args:
                        optional = kw_args['optional']
                    else:
                        optional = ''
                    if _used_aliases is None:
                        used_aliases = {self.command_name}
                    else:
                        used_aliases = _used_aliases.copy()
                        used_aliases.add(self.command_name)
                    if not log_only:
                        try:
                            result = ci.function(session, *args, optional=optional,
                                                 _used_aliases=used_aliases, log=log)
                        except CancelOperation:
                            if not self._ci.self_logging:
                                session.logger.info("Command cancelled by user")
                            raise
                        results.extend(result)

            self.command_name = None
            self._ci = None
            self._kw_args = {}
            m = _whitespace.match(self.current_text, self.amount_parsed)
            self.amount_parsed = m.end()
            if self.amount_parsed == len(self.current_text):
                return results
            self.amount_parsed += 1  # skip semicolon

    def log(self):
        session = self._session()  # resolve back reference
        cmd_text = self.current_text[self.start:self.amount_parsed]
        url = self._ci.url if self._ci is not None else None
        log_command(session, self.command_name, cmd_text, url=url)

    def log_parse_error(self):
        session = self._session()  # resolve back reference
        rest = self.current_text[self.amount_parsed:]
        spaces = len(rest) - len(rest.lstrip())
        error_at = self.amount_parsed + spaces
        syntax_error = error_at < len(self.current_text)
        if session is None:
            # for testing purposes
            print(self.current_text[self.start:])
            if syntax_error:
                error_at -= self.start
                if error_at:
                    print("%s^" % ('.' * error_at))
            print(self._error)
        elif not session.ui.is_gui:
            session.logger.error(self.current_text[self.start:])
            if syntax_error:
                error_at -= self.start
                if error_at:
                    session.logger.error("%s^" % ('.' * error_at))
        else:
            from html import escape
            ci = self._ci
            err_color = 'crimson'
            msg = '<div class="cxcmd">'
            if ci is None or ci.url is None:
                offset = 0
            else:
                offset = len(self.command_name)
                msg += '<a href="%s">%s</a>' % (
                    ci.url, escape(self.command_name))
            if not syntax_error:
                msg += escape(self.current_text[self.start + offset:self.amount_parsed])
            else:
                msg += '%s<span style="color:white; background-color:%s;">%s</span>' % (
                    escape(self.current_text[self.start + offset:error_at]),
                    err_color,
                    escape(self.current_text[error_at:]))
            msg += '</div>'
            session.logger.info(msg, is_html=True, add_newline=False)


@contextmanager
def command_trigger(session, log, cmd_text):
    if session is not None and log:
        session.triggers.activate_trigger("command started", cmd_text)
    try:
        yield
    except Exception:
        if session is not None and log:
            session.triggers.activate_trigger("command failed", cmd_text)
        raise
    if session is not None and log:
        session.triggers.activate_trigger("command finished", cmd_text)


def command_function(name, no_aliases=False, *, registry=None):
    """Return callable for given command name

    :param name: the name of the command
    :param no_aliases: True if aliases should not be considered.
    :returns: the callable that implements the command
    """
    cmd = Command(None, registry=registry)
    cmd.current_text = name
    cmd._find_command_name(no_aliases=no_aliases)
    if not cmd._ci or cmd.amount_parsed != len(cmd.current_text):
        raise ValueError('"%s" is not a command name' % name)
    return cmd._ci.function


def command_url(name, no_aliases=False, *, registry=None):
    """Return help URL for given command name

    :param name: the name of the command
    :param no_aliases: True if aliases should not be considered.
    :returns: the URL registered with the command
    """
    cmd = Command(None, registry=registry)
    cmd.current_text = name
    cmd._find_command_name(no_aliases=no_aliases)
    if cmd.amount_parsed == 0:
        raise ValueError('"%s" is not a command name' % name)
    if cmd._ci:
        return cmd._ci.url
    return _get_help_url(cmd.command_name.split())


def usage(session, name, no_aliases=False, show_subcommands=5, expand_alias=True, show_hidden=False, *,
          registry=None):
    try:
        text = _usage(name, no_aliases, show_subcommands, expand_alias, show_hidden, registry=registry)
    except ValueError as e:
        _compute_available_commands(session)
        if _available_commands is None:
            raise e
        try:
            text = _usage(name, no_aliases, show_subcommands, expand_alias, show_hidden,
                          registry=_available_commands if registry is None else registry)
        except ValueError:
            raise e
    return text


def _usage(name, no_aliases=False, show_subcommands=5, expand_alias=True,
           show_hidden=False, *, registry=None, _shown_cmds=None):
    """Return usage string for given command name

    :param name: the name of the command
    :param no_aliases: True if aliases should not be considered.
    :param show_subcommands: number of subcommands that should be shown.
    :param show_hidden: True if hidden keywords should be shown.
    :returns: a usage string for the command
    """
    if _shown_cmds is None:
        _shown_cmds = set()
    name = name.strip()
    cmd = Command(None, registry=registry)
    cmd.current_text = name
    cmd._find_command_name(no_aliases=no_aliases)
    if cmd.amount_parsed == 0:
        raise ValueError('"%s" is not a command name' % name)
    if cmd.command_name in _shown_cmds:
        return ''

    syntax = ''
    ci = cmd._ci
    if ci:
        arg_syntax = []
        syntax = cmd.command_name
        for arg_name in ci._required:
            arg = ci._required[arg_name]
            arg_name = user_kw(arg_name)
            type = arg.name
            if can_be_empty_arg(arg):
                syntax += ' [%s]' % arg_name
            else:
                syntax += ' %s' % arg_name
            arg_syntax.append('  %s: %s' % (arg_name, type))
        num_opt = 0
        for arg_name in ci._optional:
            if not show_hidden and arg_name in ci._hidden:
                continue
            arg = ci._optional[arg_name]
            arg_name = user_kw(arg_name)
            type = arg.name
            if can_be_empty_arg(arg):
                syntax += ' [%s]' % arg_name
            else:
                syntax += ' [%s' % arg_name
                num_opt += 1
            arg_syntax.append('  %s: %s' % (arg_name, type))
        syntax += ']' * num_opt
        for arg_name in ci._keyword:
            if not show_hidden and (arg_name in ci._hidden or arg_name in ci._optional):
                continue
            arg_type = ci._keyword[arg_name]
            uarg_name = user_kw(arg_name)
            if arg_type is NoArg:
                syntax += ' [%s]' % uarg_name
                continue
            if arg_name in ci._required_arguments:
                syntax += ' %s _%s_' % (uarg_name, arg_type.name)
            else:
                syntax += ' [%s _%s_]' % (uarg_name, arg_type.name)
        if registry is not None and registry is _available_commands:
            uninstalled = ' (uninstalled)'
        else:
            uninstalled = ''
        if ci.synopsis:
            syntax += ' --%s %s' % (uninstalled, ci.synopsis)
        else:
            syntax += ' --%s no synopsis available' % uninstalled
        if arg_syntax:
            syntax += '\n%s' % '\n'.join(arg_syntax)
        _shown_cmds.add(cmd.command_name)
        if expand_alias and ci.is_alias():
            alias = ci.function
            arg_text = cmd.current_text[cmd.amount_parsed:]
            args = arg_text.split(maxsplit=alias.num_args)
            if len(args) > alias.num_args:
                optional = args[-1]
                del args[-1]
            else:
                optional = ''
            try:
                name = alias.expand(*args, optional=optional, partial_ok=True)
                if name not in _shown_cmds:
                    syntax += '\n' + _usage(name, registry=registry, _shown_cmds=_shown_cmds)
                    _shown_cmds.add(name)
            except Exception as e:
                print(e)
                pass

    if (show_subcommands and cmd.word_info is not None and
            cmd.word_info.has_subcommands()):
        sub_cmds = registered_commands(multiword=True, _start=cmd.word_info)
        name = cmd.current_text[:cmd.amount_parsed]
        if len(sub_cmds) <= show_subcommands:
            for w in sub_cmds:
                subcmd = '%s %s' % (name, w)
                if subcmd in _shown_cmds:
                    continue
                syntax += '\n\n' + _usage(subcmd, show_subcommands=0, registry=registry, _shown_cmds=_shown_cmds)
                _shown_cmds.add(subcmd)
        else:
            if syntax:
                syntax += '\n'
            syntax += 'Subcommands are:\n' + '\n'.join(
                '  %s %s' % (name, w) for w in sub_cmds)

    return syntax


def can_be_empty_arg(arg):
    return isinstance(arg, Or) and EmptyArg in arg.annotations


def html_usage(session, name, no_aliases=False, show_subcommands=5, expand_alias=True,
               show_hidden=False, *, registry=None):
    try:
        text = _html_usage(name, no_aliases, show_subcommands, expand_alias, show_hidden, registry=registry)
    except ValueError as e:
        _compute_available_commands(session)
        if _available_commands is None:
            raise e
        try:
            text = _html_usage(name, no_aliases, show_subcommands, expand_alias, show_hidden,
                               registry=_available_commands if registry is None else registry)
        except ValueError:
            raise e
    return text


def _html_usage(name, no_aliases=False, show_subcommands=5, expand_alias=True,
                show_hidden=False, *, registry=None, _shown_cmds=None):
    """Return usage string in HTML for given command name

    :param name: the name of the command
    :param no_aliases: True if aliases should not be considered.
    :param show_subcommands: number of subcommands that should be shown.
    :param show_hidden: True if hidden keywords should be shown.
    :returns: a HTML usage string for the command
    """
    if _shown_cmds is None:
        _shown_cmds = set()
    cmd = Command(None, registry=registry)
    cmd.current_text = name
    cmd._find_command_name(no_aliases=no_aliases)
    if cmd.amount_parsed == 0:
        raise ValueError('"%s" is not a command name' % name)
    if cmd.command_name in _shown_cmds:
        return ''
    from html import escape

    syntax = ''
    ci = cmd._ci
    if ci:
        arg_syntax = []
        if cmd._ci.url is None:
            syntax += '<b>%s</b>' % escape(cmd.command_name)
        else:
            syntax += '<b><a href="%s">%s</a></b>' % (
                ci.url, escape(cmd.command_name))
        for arg_name in ci._required:
            arg_type = ci._required[arg_name]
            arg_name = user_kw(arg_name)
            if arg_type.url is not None:
                arg_name = arg_type.html_name(arg_name)
            else:
                arg_name = escape(arg_name)
            if can_be_empty_arg(arg_type):
                syntax += ' [<i>%s</i>]' % arg_name
            else:
                syntax += ' <i>%s</i>' % arg_name
            if arg_type.url is None:
                arg_syntax.append('<i>%s</i>: %s' % (arg_name, arg_type.html_name()))
        num_opt = 0
        for arg_name in ci._optional:
            if not show_hidden and arg_name in ci._hidden:
                continue
            arg_type = ci._optional[arg_name]
            arg_name = escape(user_kw(arg_name))
            if arg_type.url is not None:
                arg_name = arg_type.html_name(arg_name)
            else:
                arg_name = escape(arg_name)
            if can_be_empty_arg(arg_type):
                syntax += ' [<i>%s</i>]' % arg_name
            else:
                syntax += ' [<i>%s</i>' % arg_name
                num_opt += 1
            if arg_type.url is None:
                arg_syntax.append('<i>%s</i>: %s' % (arg_name, arg_type.html_name()))
        syntax += ']' * num_opt
        for arg_name in ci._keyword:
            if not show_hidden and (arg_name in ci._hidden or arg_name in ci._optional):
                continue
            arg_type = ci._keyword[arg_name]
            uarg_name = escape(user_kw(arg_name))
            if arg_type is NoArg:
                type_info = ""
            elif isinstance(arg_type, type):
                type_info = " <i>%s</i>" % arg_type.html_name()
            else:
                type_info = " <i>%s</i>" % uarg_name
                if arg_name not in ci._optional:
                    arg_syntax.append('<i>%s</i>: %s' % (uarg_name, arg_type.html_name()))
            if arg_name in ci._required_arguments:
                syntax += ' <nobr><b>%s</b>%s</nobr>' % (uarg_name, type_info)
            else:
                syntax += ' <nobr>[<b>%s</b>%s]</nobr>' % (uarg_name, type_info)
        syntax += "<br>\n&nbsp;&nbsp;&nbsp;&nbsp;&mdash; "  # synopsis prefix
        if registry is not None and registry is _available_commands:
            syntax += '(uninstalled) '
        if ci.synopsis:
            syntax += "<i>%s</i>\n" % escape(ci.synopsis)
        else:
            syntax += "<i>[no synopsis available]</i>\n"
        if arg_syntax:
            syntax += '<br>\n&nbsp;&nbsp;%s' % '<br>\n&nbsp;&nbsp;'.join(arg_syntax)
        _shown_cmds.add(cmd.command_name)
        if expand_alias and ci.is_alias():
            alias = ci.function
            arg_text = cmd.current_text[cmd.amount_parsed:]
            args = arg_text.split(maxsplit=alias.num_args)
            if len(args) > alias.num_args:
                optional = args[-1]
                del args[-1]
            else:
                optional = ''
            try:
                name = alias.expand(*args, optional=optional, partial_ok=True)
                if name not in _shown_cmds:
                    syntax += '<br>' + _html_usage(name, registry=registry, _shown_cmds=_shown_cmds)
                    _shown_cmds.add(name)
            except Exception:
                pass

    if (show_subcommands and cmd.word_info is not None and
            cmd.word_info.has_subcommands()):
        sub_cmds = registered_commands(multiword=True, _start=cmd.word_info)
        name = cmd.current_text[:cmd.amount_parsed]
        if len(sub_cmds) <= show_subcommands:
            for w in sub_cmds:
                subcmd = '%s %s' % (name, w)
                if subcmd in _shown_cmds:
                    continue
                syntax += '<p>\n' + _html_usage(subcmd, show_subcommands=0, registry=registry, _shown_cmds=_shown_cmds)
                _shown_cmds.add(subcmd)
        else:
            if syntax:
                syntax += '<br>\n'
            syntax += 'Subcommands are:\n<ul>'
            for word in sub_cmds:
                subcmd = '%s %s' % (name, word)
                cmd = Command(None, registry=registry)
                cmd.current_text = subcmd
                cmd._find_command_name(no_aliases=no_aliases)
                if cmd.amount_parsed != len(cmd.current_text):
                    url = None
                elif cmd._ci is None or cmd._ci.url is None:
                    url = None
                else:
                    url = cmd._ci.url
                if url is not None:
                    syntax += '<li> <b><a href="%s">%s</a></b>\n' % (
                        url, escape(subcmd))
                else:
                    syntax += '<li> <b>%s</b>\n' % escape(subcmd)
            syntax += '</ul>\n'

    return syntax


def registered_commands(multiword=False, _start=None):
    """Return a sorted list of the currently registered commands"""

    if _start:
        parent_info = _start
    else:
        parent_info = _command_info.commands

    if not multiword:
        words = list(parent_info.subcommands.keys())
        words.sort(key=lambda x: x[x[0] == '~':])
        return words

    def cmds(parent_cmd, parent_info):
        skip_list = []
        for word, word_info in list(parent_info.subcommands.items()):
            if word_info.is_deferred():
                try:
                    if parent_cmd:
                        word_info.lazy_register('%s %s' % (parent_cmd, word))
                    else:
                        word_info.lazy_register(word)
                except RuntimeError:
                    skip_list.append(word)
        words = [word for word in parent_info.subcommands.keys()
                 if word not in skip_list]
        words.sort(key=lambda x: x[x[0] == '~':].lower())
        for word in words:
            word_info = parent_info.subcommands[word]
            if word_info.is_deferred():
                continue
            if word_info.cmd_desc:
                if parent_cmd:
                    yield '%s %s' % (parent_cmd, word)
                else:
                    yield word
            if word_info.has_subcommands():
                if parent_cmd:
                    yield from cmds('%s %s' % (parent_cmd, word), word_info)
                else:
                    yield from cmds(word, word_info)
    return list(cmds('', parent_info))


class Alias:
    """alias a command

    Returns a callable unnamed command alias.

    :param text: parameterized command text
    :param user: true if alias was generated by user

    The text is scanned for $n, where n is the n-th argument, $* for the rest
    of the line, and $$ for a single $.
    """

    def __init__(self, text, *, user=False, registry=None):
        text = text.lstrip()
        self.original_text = text
        self.user_generated = user
        self.num_args = 0
        self.parts = []  # list of strings and integer argument numbers
        self.optional_rest_of_line = False
        self.registry = registry
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

    def cmd_desc(self, **kw):
        """Return CmdDesc instance for alias

        The :py:class:`CmdDesc` keyword arguments other than 'required',
        'optional', and 'keyword' can be used.
        """
        if kw.pop('required', None) is not None:
            raise ValueError('can not override required arguments')
        if kw.pop('optional', None) is not None:
            raise ValueError('can not override optional arguments')
        if kw.pop('keyword', None) is not None:
            raise ValueError('can not override keyword arguments')
        required = [((i + 1), StringArg) for i in range(self.num_args)]
        if not self.optional_rest_of_line:
            return CmdDesc(required=required, **kw)
        return CmdDesc(required=required, optional=[('optional', RestOfLine)],
                       non_keyword=['optional'], **kw)

    def expand(self, *args, optional='', partial_ok=False):
        if not partial_ok and len(args) < self.num_args:
            raise UserError("Not enough arguments")
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
        return text

    def __call__(self, session, *args, optional='', echo_tag=None,
                 _used_aliases=None, log=True):
        # when echo_tag is not None, echo the substitued alias with
        # the given tag
        text = self.expand(*args, optional=optional)
        if echo_tag is not None:
            session.logger.info('%s%s' % (echo_tag, text))
        # save Command object so error reporting can give underlying error
        self.cmd = Command(session, registry=self.registry)
        return self.cmd.run(text, _used_aliases=_used_aliases, log=log)


def list_aliases(all=False, logger=None):
    """List all aliases

    :param all: if True, then list all aliases, not just user ones

    Return in depth-first order.
    """
    def find_aliases(partial_name, parent_info):
        for word, word_info in list(parent_info.subcommands.items()):
            if all and word_info.is_deferred():
                try:
                    if partial_name:
                        word_info.lazy_register('%s %s' % (partial_name, word))
                    else:
                        word_info.lazy_register(word)
                except RuntimeError as e:
                    if logger:
                        logger.warning(str(e))
                    continue
            if partial_name:
                yield from find_aliases('%s %s' % (partial_name, word), word_info)
            else:
                yield from find_aliases('%s' % word, word_info)
        if all:
            if parent_info.is_alias():
                yield partial_name
        elif parent_info.is_user_alias():
            yield partial_name
    return list(find_aliases('', _command_info.commands))


def expand_alias(name, *, registry=None):
    """Return text of named alias

    :param name: name of the alias
    """
    cmd = Command(None, registry=registry)
    cmd.current_text = name
    cmd._find_command_name(no_aliases=False)
    if cmd.amount_parsed != len(cmd.current_text):
        return None
    if not cmd.word_info.is_alias():
        return None
    return cmd.word_info.alias().original_text


def create_alias(name, text, *, user=False, logger=None, url=None, registry=None):
    """Create command alias

    :param name: name of the alias
    :param text: text of the alias
    :param user: boolean, true if user created alias
    :param logger: optional logger
    """
    name = ' '.join(name.split())   # canonicalize
    alias = Alias(text, user=user, registry=registry)
    try:
        register(name, alias.cmd_desc(synopsis='alias of "%s"' % text, url=url),
                 alias, logger=logger, registry=registry)
    except Exception:
        raise


def remove_alias(name=None, user=False, logger=None, *, registry=None):
    """Remove command alias

    :param name: name of the alias
    :param user: boolean, true if user created alias

    If no name is given, then all user generated aliases are removed.
    """
    if name is None:
        for name in list_aliases(logger=logger):
            deregister(name, is_user_alias=True, registry=registry)
        return

    deregister(name, is_user_alias=user, registry=registry)


if __name__ == '__main__':
    from ..utils import flattened

    alias_desc = CmdDesc(required=[('name', StringArg), ('text', WholeRestOfLine)], url='skip')

    @register('alias', alias_desc)
    def alias(session, name, text):
        create_alias(name, text, url='skip')

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
                    raise ValueError('Invalid color name: "%s"' % color_name)
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
        keyword=[('color', ColorArg)],
        url='skip'
    )

    @register('test1', test1_desc)
    def test1(session, a, b, color=None):
        print('test1 a: %s %s' % (type(a), a))
        print('test1 b: %s %s' % (type(b), b))
        print('test1 color: %s %s' % (type(color), color))

    test2_desc = CmdDesc(
        keyword=[('color', ColorArg), ('radius', FloatArg)],
        url='skip'
    )

    @register('test2', test2_desc)
    def test2(session, color=None, radius=0):
        print('test2 color: %s %s' % (type(color), color))
        print('test2 radius: %s %s' % (type(radius), radius))

    register('mw test1', test1_desc.copy(), test1)
    register('mw test2', test2_desc.copy(), test2)

    test3_desc = CmdDesc(
        required=[('name', StringArg)],
        optional=[('value', FloatArg)],
        url='skip'
    )

    @register('test3', test3_desc)
    def test3(session, name, value=None):
        print('test3 name: %s %s' % (type(name), name))
        print('test3 value: %s %s' % (type(value), value))

    test4_desc = CmdDesc(
        optional=[('draw', PositiveIntArg)],
        url='skip'
    )

    @register('test4', test4_desc)
    def test4(session, draw=None):
        print('test4 draw: %s %s' % (type(draw), draw))

    test4b_desc = CmdDesc(
        optional=[('draw', IntArg)],
        postconditions=[Limited('draw', min=1)],
        url='skip'
    )

    @register('test4b', test4b_desc)
    def test4b(session, draw=None):
        print('test4b draw: %s %s' % (type(draw), draw))

    test5_desc = CmdDesc(
        optional=[('ints', IntsArg)],
        url='skip'
    )

    @register('test5', test5_desc)
    def test5(session, ints=None):
        print('test5 ints: %s %s' % (type(ints), ints))

    test6_desc = CmdDesc(
        required=[('center', Float3Arg)],
        url='skip'
    )

    @register('test6', test6_desc)
    def test6(session, center):
        print('test6 center:', center)

    test7_desc = CmdDesc(
        optional=[('center', Float3Arg)],
        url='skip'
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
        url='skip'
    )

    @register('test8', test8_desc)
    def test8(session, always=True, target="all", names=[None]):
        print('test8 always, target, names:', always, target, names)

    test9_desc = CmdDesc(
        optional=(
            ("target", StringArg),
            ("names", ListOf(StringArg))
        ),
        keyword=(("full", BoolArg),),
        url='skip'
    )

    @register('test9', test9_desc)
    def test9(session, target="all", names=[None], full=False):
        print('test9 full, target, names: %r, %r, %r' % (full, target, names))

    test10_desc = CmdDesc(
        optional=(
            ("colors", ListOf(ColorArg)),
            ("offsets", ListOf(FloatArg)),
        ),
        required_arguments=("colors", "offsets"),
        postconditions=(
            SameSize('colors', 'offsets'),
        ),
        url='skip'
    )

    @register('test10', test10_desc)
    def test10(session, colors=[], offsets=[]):
        print('test10 colors, offsets:', colors, offsets)

    def lazy_reg():
        test11_desc = CmdDesc(url='skip')

        def test11(session):
            print('delayed')
        register('xyzzy subcmd', test11_desc, test11)
    delay_registration('xyzzy', lazy_reg)

    @register('echo', CmdDesc(optional=[('text', RestOfLine)], url='skip'))
    def echo(session, text=''):
        return text

    if len(sys.argv) > 1:
        _debugging = 'd' in sys.argv[1]

        @register('exit')
        def exit(session):
            raise SystemExit(0)

        register('usage', CmdDesc(required=[('name', RestOfLine)], url='skip'), usage)

        register('html_usage', CmdDesc(required=[('name', RestOfLine)], url='skip'),
                 html_usage)
        prompt = 'cmd> '
        cmd = Command(None)
        while True:
            try:
                text = input(prompt)
                results = cmd.run(text)
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
        (True, True, 'test5 ints 5 ints 6'),
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
        (False, True, 'echo hi there'),
        (False, True, 'alias plugh echo $* $*'),
        (False, True, 'plugh who'),
    ]
    sys.stdout = sys.stderr
    successes = 0
    failures = 0
    cmd = Command(None)
    for t in tests:
        fail, final, text = t
        try:
            print("\nTEST: '%s'" % text)
            results = cmd.run(text)
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


def log_command(session, command_name, command_text, *, url=None):
    if session is None:
        # for testing purposes
        print("Executing: %s" % command_text)
    elif not session.ui.is_gui:
        session.logger.info("Executing: %s" % command_text)
    else:
        from html import escape
        msg = '<div class="cxcmd"><div class="cxcmd_as_doc">'
        if url is None:
            msg += escape(command_text)
        else:
            cargs = command_text[len(command_name):]
            msg += '<a href="%s">%s</a>%s' % (url, escape(command_name), escape(cargs))
        text = escape(command_text)
        msg += '</div><div class="cxcmd_as_cmd"><a href="cxcmd:%s">%s</a></div></div>' % (text, text)
        session.logger.info(msg, is_html=True, add_newline=False)
