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
atomspec: atom specifier cli annotation and evaluation
======================================================

The 'atomspec' module provides two classes:

- AtomSpecArg : command line argument annotation class.
- AtomSpec : atom specifier class.

AtomSpecArg is a cli type annotation and is used to describe an
argument of a function that is registered with the cli module.  When
the registered function is called, the argument corresponding to
the AtomSpecArg is an instance of AtomSpec, which contains a parsed
version of the input atom specifier.  The model elements (atoms,
bonds, models, etc) that match an AtomSpec may be found by calling
the 'evaluate' method which returns an instance of Objects.
Each type of model elements may be accessed as an attribute of the
Objects instance.

Selectors
---------

A (name, function) pair may be registered as a 'selector' in an
atom specifier.  The selectors may either be global (e.g., chemical
groups) or session-specific (e.g., active site).  The selector
name may appear wherever a model, residue or atom string does.
The selector function is called when the atom specifier is
evaluated and is expected to fill in an Objects instance.

Example
-------

Here is an example of a function that may be registered with cli::

    from chimerax.core.commands import cli, atomspec

    def move(session, by, modelspec=None):
        if modelspec is None:
            modelspec = atomspec.everything(session)
        spec = modelspec.evaluate(session)
        import numpy
        by_vector = numpy.array(by)
        from chimerax.geometry import place
        translation = place.translation(by_vector)
        for m in spec.models:
            m.position = translation * m.position
    move_desc = cli.CmdDesc(required=[("by", cli.Float3Arg)],
                            optional=[("modelspec", atomspec.AtomSpecArg)])

Notes
-----

AtomSpecArg arguments should always be optional because
not providing an atom specifier is the same as choosing
all atoms.

"""

import re
from .cli import Annotation
from contextlib import contextmanager

_double_quote = re.compile(r'"(.|\")*?"(\s|$)')
_terminator = re.compile(r"[;\s]")  # semicolon or whitespace

MAX_STACK_DEPTH = 10000000


@contextmanager
def maximum_stack(max_depth=MAX_STACK_DEPTH):
    # fix #2790 by increasing maximum stack depth
    import sys
    save_current_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max_depth)
    yield
    sys.setrecursionlimit(save_current_limit)


class AtomSpecArg(Annotation):
    """Command line type annotation for atom specifiers.

    See cli documentation for details on type annotations.

    """
    name = "an atom specifier"
    url = "help:user/commands/atomspec.html"

    @classmethod
    def parse(cls, text, session):
        """Parse text and return an atomspec parse tree"""
        if not text or _terminator.match(text[0]) is not None:
            from .cli import AnnotationError
            raise AnnotationError("empty atom specifier")
        if text[0] == '"':
            return cls._parse_quoted(text, session)
        else:
            return cls._parse_unquoted(text, session)

    @classmethod
    def _parse_quoted(cls, text, session):
        # Split out quoted argument
        start = 0
        m = _double_quote.match(text, start)
        if m is None:
            from .cli import AnnotationError
            raise AnnotationError("incomplete quoted text")
        end = m.end()
        consumed = text[start:end]
        if text[end - 1].isspace():
            end -= 1
        if text[1] == '=':
            add_implied = False
            start += 1
        else:
            add_implied = True
        # Quoted argument is consumed on success
        # Text after quote is unused
        rest = text[end:]
        # Convert quote contents to string
        from .cli import unescape_with_index_map
        token, index_map = unescape_with_index_map(text[start + 1:end - 1])
        # Create parser and parse converted token
        from ._atomspec import _atomspecParser
        parser = _atomspecParser(parseinfo=True)
        semantics = _AtomSpecSemantics(session, add_implied=add_implied)
        from grako.exceptions import FailedParse, FailedSemantics
        try:
            with maximum_stack():
                ast = parser.parse(token, "atom_specifier", semantics=semantics)
        except FailedSemantics as e:
            from .cli import AnnotationError
            raise AnnotationError(str(e), offset=e.pos)
        except FailedParse as e:
            from .cli import AnnotationError, discard_article
            # Add one to offset for leading quote
            offset = index_map[e.pos]
            message = 'invalid ' + discard_article(cls.name)
            if str(e.message) != 'no available options':
                message = '%s: %s' % (message, e.message)
            raise AnnotationError(message, offset=offset)
        # Must consume everything inside quotes
        if ast.parseinfo.endpos != len(token):
            from .cli import AnnotationError
            offset = index_map[ast.parseinfo.endpos] + 1
            raise AnnotationError("mangled atom specifier", offset=offset)
        # Success!
        return ast, consumed, rest

    @classmethod
    def _parse_unquoted(cls, text, session):
        # Try to parse the entire line.
        # If we get nothing, then raise AnnotationError.
        # Otherwise, consume what we can use and call it a success.
        if text.startswith('='):
            parse_text = text[1:]
            add_implied = False
            text_offset = 1
        else:
            parse_text = text
            add_implied = True
            text_offset = 0
        from ._atomspec import _atomspecParser
        parser = _atomspecParser(parseinfo=True)
        semantics = _AtomSpecSemantics(session, add_implied=add_implied)
        from grako.exceptions import FailedParse, FailedSemantics
        try:
            with maximum_stack():
                ast = parser.parse(parse_text, "atom_specifier", semantics=semantics)
        except FailedSemantics as e:
            from .cli import AnnotationError
            raise AnnotationError(str(e), offset=e.pos)
        except FailedParse as e:
            from .cli import AnnotationError, discard_article
            message = 'invalid ' + discard_article(cls.name)
            if str(e.message) != 'no available options':
                message = '%s: %s' % (message, e.message)
            raise AnnotationError(message, offset=e.pos)

        end = ast.parseinfo.endpos
        if end == 0:
            from .cli import AnnotationError
            raise AnnotationError("not an atom specifier")
        if end < len(parse_text) and _terminator.match(parse_text[end]) is None:
            # We got an error in the middle of a string (no whitespace or
            # semicolon).  We check if there IS whitespace between the
            # start of the string and the error location.  If so, we
            # assume that the atomspec successfully ended at the whitespace
            # and leave the rest as unconsumed input.
            blank = end
            while blank > 0:
                if parse_text[blank].isspace():
                    break
                else:
                    blank -= 1
            if blank == 0:
                # No whitespace found
                from .cli import AnnotationError
                raise AnnotationError('only initial part "%s" of atom specifier valid'
                                      % text[:end + text_offset])
            else:
                ast, used, rem = AtomSpecArg._parse_unquoted(text[:blank + text_offset], session)
                return ast, used, rem + text[blank + text_offset:]
        # Consume what we used and return the remainder
        return ast, text[:end + text_offset], text[end + text_offset:]


#
# Parsing functions and classes
#


class _AtomSpecSemantics:
    """Semantics class to convert basic ASTs into AtomSpec instances."""
    def __init__(self, session, *, add_implied=True):
        self._session = session
        self._add_implied = add_implied

    def atom_specifier(self, ast):
        # print("atom_specifier", ast)
        atom_spec = AtomSpec(ast.operator, ast.left, ast.right, add_implied=self._add_implied)
        try:
            atom_spec.parseinfo = ast.parseinfo
        except AttributeError:
            pass
        return atom_spec

    def as_term(self, ast):
        # print("as_term", ast)
        if ast.zone:
            if ast.atomspec is not None:
                ast.zone.model = ast.atomspec
            elif ast.tilde is not None:
                ast.zone.model = ast.tilde
            elif ast.selector is not None:
                ast.zone.model = ast.selector
            return _Term(ast.zone)
        elif ast.atomspec is not None:
            return ast.atomspec
        elif ast.tilde is not None:
            return _Invert(ast.tilde, add_implied=self._add_implied)
        elif ast.selector is not None:
            return _Term(ast.selector)
        else:
            return _Term(ast.models)

    def selector_name(self, ast):
        # print("selector_name", ast)
        if get_selector(ast.name) is None:
            from grako.exceptions import FailedSemantics
            e = FailedSemantics("\"%s\" is not a selector name" % ast.name)
            e.pos = ast.parseinfo.pos
            e.endpos = ast.parseinfo.endpos
            raise e
        return _SelectorName(ast.name)

    def model_list(self, ast):
        # print("model_list", ast)
        model_list = _ModelList()
        if ast.model:
            model_list.extend(ast.model)
        return model_list

    def model(self, ast):
        m = _Model(ast.exact_match == "#!", ast.hierarchy, ast.attrs)
        if ast.parts is not None:
            for p in ast.parts:
                m.add_part(p)
        if ast.zone is None:
            # print("model", m)
            return m
        ast.zone.model = m
        return ast.zone

    def model_hierarchy(self, ast):
        hierarchy = _ModelHierarchy(ast.range_list)
        if ast.hierarchy:
            for rl in ast.hierarchy:
                if rl:
                    hierarchy.append(rl)
        return hierarchy

    def model_range_list(self, ast):
        range_list = _ModelRangeList(ast.range)
        if ast.range_list:
            for rl in ast.range_list:
                if rl:
                    range_list.append(rl)
        return range_list

    def model_range(self, ast):
        return _ModelRange(ast.start, ast.end)

    def model_spec_any(self, ast):
        if ast.number is not None:
            return ast.number
        elif ast.word is not None:
            return ast.word
        else:
            return None
    # Start and end are the same as any, except a different word is used
    model_spec_start = model_spec_any
    model_spec_end = model_spec_any

    def model_spec(self, ast):
        if ast.number is not None:
            return int(ast.number)
        else:
            return None

    def model_parts(self, ast):
        return ast.chain

    def chain(self, ast):
        c = _Chain(ast.parts, ast.attrs)
        if ast.residue:
            for r in ast.residue:
                c.add_part(r)
        return c

    def residue(self, ast):
        r = _Residue(ast.parts, ast.attrs)
        if ast.atom:
            for a in ast.atom:
                r.add_part(a)
        return r

    def part_list(self, ast):
        if ast.part is None:
            return _PartList(ast.range)
        else:
            return ast.part.add_parts(ast.range)

    def part_range_list(self, ast):
        return _Part(ast.start, ast.end)

    def atom(self, ast):
        return _Atom(ast.parts, ast.attrs)

    def atom_list(self, ast):
        if ast.part is None:
            return _PartList(ast.name)
        else:
            return ast.part.add_parts(ast.name)

    def atom_name(self, ast):
        return _Part(ast.name, None)

    def attribute_list(self, ast):
        attr_test, attr_list = ast
        if not attr_list:
            return _AttrList([attr_test])
        else:
            return attr_list.insert(0, attr_test)

    def attr_test(self, ast):
        import operator
        if ast.no is not None:
            op = operator.not_
            v = None
        elif ast.value is None:
            op = operator.truth
            v = None
        else:
            if ast.op == "=":
                op = operator.eq
            elif ast.op == "!=":
                op = operator.ne
            elif ast.op == ">=":
                op = operator.ge
            elif ast.op == ">":
                op = operator.gt
            elif ast.op == "<=":
                op = operator.le
            elif ast.op == "<":
                op = operator.lt
            else:
                op = ast.op
            # Convert string to value for comparison
            av = ast.value
            quoted = False
            if (isinstance(av, list) and len(av) == 3 and av[0] in ('"', "'") and av[2] in ('"', "'")):
                # Quoted value stay as string
                av = av[1]
                quoted = True
            if ast.name.lower().endswith("color"):
                # if ast.name ends with color, convert to color
                from . import ColorArg, make_converter
                try:
                    c = make_converter(ColorArg)(self._session, av)
                except ValueError as e:
                    from ..errors import UserError
                    raise UserError("bad color: %s: %s" % (av, e))
                # convert to tinyarray because numpy equality comparison is dumb
                import tinyarray as ta
                v = ta.array(c.uint8x4())
            elif quoted or op in ["==", "!=="]:
                # case sensitive compare must be string
                v = av
            else:
                # convert to best matching common type
                try:
                    v = int(av)
                except ValueError:
                    try:
                        v = float(av)
                    except ValueError:
                        v = av
        return _AttrTest(ast.no, ast.name, op, v)

    def zone_selector(self, ast):
        operator, distance = ast
        return _ZoneSelector(operator, distance)

    def zone_operator(self, ast):
        return ast

    def real_number(self, ast):
        return float(ast)


class _ModelList(list):
    """Stores list of model hierarchies."""
    def __str__(self):
        if not self:
            return "<no model specifier>"
        return "".join(str(mr) for mr in self)

    def find_matches(self, session, model_list, results, ordered, *, add_implied=None):
        for model_spec in self:
            # "model_spec" should be an _Model instance
            model_spec.find_matches(session, model_list, results, ordered, add_implied=add_implied)


class _ModelHierarchy(list):
    """Stores list of model ranges in hierarchy order."""
    def __init__(self, mrl):
        super().__init__()
        self.append(mrl)

    def __str__(self):
        if not self:
            return "[empty]"
        return ".".join(str(mr) for mr in self)

    def find_matches(self, session, model_list, sub_parts, results, ordered, *, add_implied=None):
        self._check(session, model_list, sub_parts, results, 0, ordered)

    def _check(self, session, model_list, sub_parts, results, i, ordered):
        if i >= len(self):
            # Hit end of list.  Match
            for model in model_list:
                _add_model_parts(session, model, sub_parts, results, ordered)
            return
        match_list = []
        # "self[i]" is a _ModelRangeList instance
        # "mr" is a _ModelRange instance
        for mr in self[i]:
            for model in model_list:
                try:
                    mid = model.id[i]
                except IndexError:
                    continue
                if mr.matches(mid):
                    match_list.append(model)
        if match_list:
            self._check(session, match_list, sub_parts, results, i + 1, ordered)


class _ModelRangeList(list):
    """Stores a list of model ranges."""
    def __init__(self, mr):
        super().__init__()
        self.append(mr)

    def __str__(self):
        if not self:
            return "[empty]"
        return ",".join(str(mr) for mr in self)


class _ModelRange:
    """Stores a single model range."""
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        if self.end:
            return "%s-%s" % (self.start, self.end)
        else:
            return str(self.start)

    def matches(self, mid):
        if self.end is None:
            # Exact match
            return self.start == '*' or mid == self.start
        else:
            # Range match
            if self.start != 'start' and self.start != '*' and mid < self.start:
                return False
            return self.end == 'end' or self.end == '*' or mid <= self.end


class _SubPart:
    """Stores part list for one item and subparts of the item."""
    def __init__(self, my_parts, my_attrs):
        self.my_parts = my_parts
        self.my_attrs = my_attrs
        self.sub_parts = None

    def __str__(self):
        if self.sub_parts:
            sub_repr = "".join([str(r) for r in self.sub_parts])
        else:
            sub_repr = ""
        if self.my_parts is None:
            r = sub_repr
        else:
            r = "%s%s%s" % (self.Symbol, str(self.my_parts), sub_repr)
        if self.my_attrs is not None:
            r += "%s%s%s" % (self.Symbol, self.Symbol, str(self.my_attrs))
        # print("_SubPart.__str__", self.__class__, r)
        return r

    def add_part(self, subpart):
        if subpart is None:
            return
        if self.sub_parts is None:
            self.sub_parts = [subpart]
        else:
            self.sub_parts.append(subpart)

    def find_selected_parts(self, model, atoms, num_atoms, results):
        # Only filter if a spec for this level is present
        # TODO: account for my_attrs in addition to my_parts
        if self.my_parts or self.my_attrs:
            atoms = self._filter_parts(model, atoms, num_atoms)
            num_atoms = len(atoms)
        if len(atoms) == 0:
            return
        if self.sub_parts is None:
            results.add_model(model)
            results.add_atoms(atoms)
            return
        from ..objects import Objects
        sub_results = Objects()
        for subpart in self.sub_parts:
            subpart.find_selected_parts(model, atoms, num_atoms, sub_results)
        if sub_results.num_atoms > 0:
            results.add_model(model)
            results.add_atoms(sub_results.atoms)

    def _filter_parts(self, model, atoms, num_atoms):
        if not self.my_parts:
            mask = model.atomspec_filter(self.Symbol, atoms, num_atoms,
                                         None, self.my_attrs)
            return atoms.filter(mask)
        from ..objects import Objects
        results = Objects()
        for part in self.my_parts:
            my_part = self.my_parts.__class__(part)
            mask = model.atomspec_filter(self.Symbol, atoms, num_atoms,
                                         my_part, self.my_attrs)
            sub_atoms = atoms.filter(mask)
            if len(sub_atoms) > 0:
                results.add_atoms(sub_atoms)
        return results.atoms


class _Model(_SubPart):
    """Stores model part list and atom spec."""
    Symbol = '#'

    def __init__(self, exact_match, *args):
        super().__init__(*args)
        self.exact_match = exact_match

    def find_matches(self, session, model_list, results, ordered, *, add_implied=None):
        model_list = [model for model in model_list
                      if not self.my_attrs or
                      model.atomspec_model_attr(self.my_attrs)]
        if not model_list:
            return
        if self.my_parts:
            if self.exact_match:
                model_list = [model for model in model_list
                              if len(model.id) == len(self.my_parts)]
            self.my_parts.find_matches(session, model_list, self.sub_parts, results, ordered,
                                       add_implied=add_implied)
        else:
            # No model spec given, everything matches
            for model in model_list:
                _add_model_parts(session, model, self.sub_parts, results, ordered)


def _add_model_parts(session, model, sub_parts, results, ordered):
    if not model.atomspec_has_atoms():
        if not sub_parts:
            results.add_model(model)
            if model.atomspec_has_pseudobonds():
                results.add_pseudobonds(model.atomspec_pseudobonds())
        return
    atoms = model.atomspec_atoms(ordered=ordered)
    if not sub_parts:
        results.add_model(model)
        results.add_atoms(atoms)
    else:
        # Has sub-model selector, filter atoms
        from ..objects import Objects
        my_results = Objects()
        num_atoms = len(atoms)
        for chain_spec in sub_parts:
            chain_spec.find_selected_parts(model, atoms, num_atoms, my_results)
        if my_results.num_atoms > 0:
            results.add_model(model)
            results.add_atoms(my_results.atoms)


class _Chain(_SubPart):
    """Stores residue part list and atom spec."""
    Symbol = '/'


class _Residue(_SubPart):
    """Stores residue part list and atom spec."""
    Symbol = ':'


class _Atom(_SubPart):
    """Stores residue part list and atom spec."""
    Symbol = '@'


class _PartList(list):
    """Stores a part list (sub-parts of models)."""
    def __init__(self, part_range):
        super().__init__()
        self.append(part_range)

    def __str__(self):
        return ','.join([str(p) for p in self])

    def add_parts(self, part_range):
        self.insert(0, part_range)
        return self


def _has_wildcard(s):
    try:
        return any((c in s) for c in "*?[")
    except TypeError:
        return False


class _Part:
    """Stores one part of a part range."""
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        if self.end is None:
            return self.start
        else:
            return "%s-%s" % (self.start, self.end)

    def string_matcher(self, case_sensitive=False):
        # String matcher used for atom names, chain ids, residue names, etc.
        start_test = self.start if case_sensitive else self.start.lower()
        if self.end is None:
            from fnmatch import fnmatch
            if _has_wildcard(start_test):
                if case_sensitive:
                    def matcher(name):
                        return fnmatch(name, start_test)
                else:
                    def matcher(name):
                        return fnmatch(name.lower(), start_test)
            else:
                if case_sensitive:
                    def matcher(name):
                        return name == start_test
                else:
                    def matcher(name):
                        return name.lower() == start_test
        else:
            # Both start and end specified.  No wildcards allowed (for now).
            end_test = self.end if case_sensitive else self.end.lower()
            if case_sensitive:
                def matcher(name):
                    return name >= start_test and name <= end_test
            else:
                def matcher(name):
                    n = name.lower()
                    return n >= start_test and n <= end_test
        return matcher

    def res_id_matcher(self):
        # Residue id matcher used for (residue sequence, insert code) pairs
        # "ic" = insert code
        try:
            start_seq, start_ic = self._parse_as_res_id(self.start, True)
        except (ValueError, IndexError):
            return None
        if self.end is None:
            def matcher(seq, ic):
                return seq == start_seq and ic == start_ic
        else:
            try:
                end_seq, end_ic = self._parse_as_res_id(self.end, False)
            except (ValueError, IndexError):
                return None
            if start_seq is None and end_seq is None:
                # :start-end
                def matcher(seq, ic):
                    return True
            elif start_seq is None:
                # :start-N
                def matcher(seq, ic):
                    if seq > end_seq:
                        return False
                    elif seq < end_seq:
                        return True
                    else:
                        # seq == end_seq
                        # Blank insert code < any non-blank
                        if not ic and not end_ic:
                            return True
                        elif ic and not end_ic:
                            return False
                        elif not ic and end_ic:
                            return True
                        else:
                            return ic <= end_ic
            elif end_seq is None:
                # :N-end
                def matcher(seq, ic):
                    if seq < start_seq:
                        return False
                    elif seq > start_seq:
                        return True
                    else:
                        # seq == start_seq
                        # Blank insert code < any non-blank
                        if not ic and not start_ic:
                            return True
                        elif ic and not start_ic:
                            return True
                        elif not ic and start_ic:
                            return False
                        else:
                            return ic <= start_ic
            else:
                # :N-M
                def matcher(seq, ic):
                    if seq < start_seq or seq > end_seq:
                        return False
                    elif seq > start_seq and seq < end_seq:
                        return True
                    elif seq == start_seq:
                        if not ic and not start_ic:
                            return True
                        elif ic and not start_ic:
                            return True
                        elif not ic and start_ic:
                            return False
                        else:
                            return start_ic <= ic
                    else:   # seq == end_seq
                        if not ic and not end_ic:
                            return True
                        elif ic and not end_ic:
                            return False
                        elif not ic and end_ic:
                            return True
                        else:
                            return ic <= end_ic
        return matcher

    def _parse_as_res_id(self, n, at_start):
        if at_start:
            if n.lower() == "start":
                return None, None
        else:
            if n.lower() == "end":
                return None, None
        try:
            return int(n), ""
        except ValueError:
            return int(n[:-1]), n[-1]


class _AttrList(list):
    """Stores a part list (sub-parts of models)."""
    def __str__(self):
        return ','.join([str(p) for p in self])


class _AttrTest:
    """Stores one part of a part range."""
    def __init__(self, no, name, op, value):
        self.no = no
        self.name = name
        self.op = op
        self.value = value

    def __str__(self):
        if self.no is not None:
            return '~' + self.name
        elif self.value is None:
            return self.name
        else:
            import operator
            if isinstance(self.op, str):
                op = self.op
            elif self.op == operator.eq:
                op = "="
            elif self.op == operator.ne:
                op = "!="
            elif self.op == operator.ge:
                op = ">="
            elif self.op == operator.gt:
                op = ">"
            elif self.op == operator.le:
                op = "<="
            elif self.op == operator.lt:
                op = "<"
            else:
                op = "???"
            return "%s%s%s" % (self.name, op, self.value)

    def attr_matcher(self):
        import operator
        attr_name = self.name
        if self.value is None:
            def matcher(obj):
                try:
                    v = getattr(obj, attr_name)
                except AttributeError:
                    return False
                return bool(v)
        elif (self.op in (operator.eq, operator.ne, "==", "!==") and
                isinstance(self.value, str)):
            # Equality-comparison operators for strings handle wildcards
            case_sensitive = self.op in ["==", "!=="]
            attr_value = self.value if case_sensitive else self.value.lower()
            invert = self.op in (operator.ne, "!==")
            if _has_wildcard(self.value):
                from fnmatch import fnmatchcase

                def matcher(obj):
                    try:
                        v = getattr(obj, attr_name)
                        if v is None:
                            return False
                    except AttributeError:
                        return False
                    try:
                        v = str(v)
                    except TypeError:
                        # "fake" attribute, such as Chain.identity
                        pass
                    else:
                        if not case_sensitive:
                            v = v.lower()
                    matches = fnmatchcase(v, attr_value)
                    return not matches if invert else matches
            else:
                def matcher(obj):
                    try:
                        v = getattr(obj, attr_name)
                        if v is None:
                            return False
                    except AttributeError:
                        return False
                    try:
                        v = str(v)
                    except TypeError:
                        # "fake" attribute, such as Chain.identity
                        pass
                    else:
                        if not case_sensitive:
                            v = v.lower()
                    matches = v == attr_value
                    return not matches if invert else matches
        else:
            op = self.op
            attr_value = self.value

            def matcher(obj):
                try:
                    v = getattr(obj, attr_name)
                    if v is None:
                        return False
                except AttributeError:
                    return False
                return op(v, attr_value)
        return matcher


class _SelectorName:
    """Stores a single selector name."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def find_matches(self, session, models, results, ordered, *, add_implied=None):
        f = get_selector(self.name)
        if f:
            from ..objects import Objects
            if isinstance(f, Objects):
                results.combine(f)
            else:
                try:
                    f(session, models, results)
                except Exception:
                    session.logger.report_exception(preface="Error executing selector '%s'" % self.name)
                    from grako.exceptions import FailedSemantics
                    raise FailedSemantics("error evaluating selector %s" % self.name)


class _ZoneSelector:
    """Stores zone operator and distance information."""
    def __init__(self, operator, distance):
        self.distance = distance
        self.target_type = operator[0]  # '@', ':' or '#'
        self.operator = operator[1:]    # '<', '>'
        # We do not support <= or >= because distances are
        # computed as floating point and equality suggests
        # more control than we really have.  If two atoms
        # are _actually_ _exactly_ equal to the specified
        # distance, we put them in the < bucket (because
        # we are using closepoints).
        self.model = None

    def __str__(self):
        return "%s%s%.3f" % (self.target_type, self.operator, self.distance)

    def find_matches(self, session, models, results, ordered, *, add_implied=None):
        if self.model is None:
            # No reference atomspec, so do nothing
            return
        from ..objects import Objects
        my_results = Objects()
        zone_results = Objects()
        self.model.find_matches(session, models, my_results, ordered, add_implied=add_implied)
        if my_results.num_atoms > 0:
            # expand my_results before combining with results
            coords = my_results.atoms.scene_coords
            for m in session.models.list():
                m.atomspec_zone(session, coords, self.distance,
                                self.target_type, self.operator, zone_results)
        results.combine(zone_results)
        if '<' in self.operator:
            results.combine(my_results)

    def matches(self, session, model):
        if self.model is None:
            return False
        return self.model.matches(session, model)


class _Term:
    """A term in an atom specifier."""
    def __init__(self, spec):
        self._specifier = spec

    def __str__(self):
        return str(self._specifier)

    def evaluate(self, session, models, *, top=True, ordered=False, add_implied=None):
        """Return Objects for model elements that match."""
        from ..objects import Objects
        results = Objects()
        return self.find_matches(session, models, results, ordered, add_implied=add_implied)

    def find_matches(self, session, models, results, ordered, *, add_implied=None):
        self._specifier.find_matches(session, models, results, ordered, add_implied=add_implied)
        return results


class _Invert:
    """A "not" (~) term in an atom specifier."""
    def __init__(self, atomspec, *, add_implied=True):
        self._atomspec = atomspec
        self._add_implied = add_implied

    def __str__(self):
        return "~%s" % str(self._atomspec)

    def evaluate(self, session, models=None, *, ordered=False, add_implied=None, **kw):
        if add_implied is None:
            add_implied = self._add_implied
        if models is None:
            models = session.models.list(**kw)
        with maximum_stack():
            results = self._atomspec.evaluate(session, models, top=False, ordered=ordered)
        if add_implied:
            add_implied_bonds(results)
        results.invert(session, models)
        return results

    def find_matches(self, session, models, results, ordered, *, add_implied=None):
        if add_implied is None:
            add_implied = self._add_implied
        with maximum_stack():
            self._atomspec.find_matches(session, models, results, ordered, add_implied=add_implied)
        if add_implied:
            add_implied_bonds(results)
        results.invert(session, models)
        return results


class AtomSpec:
    """AtomSpec instances store and evaluate atom specifiers.

    An AtomSpec instance, returned by AtomSpecArg arguments in
    cli command functions, keeps track of an atom specifier.
    When evaluated, the model elements that match the specifier
    are returned.
    """

    def __init__(self, operator, left_spec, right_spec, *, add_implied=True):
        self._operator = operator
        self._left_spec = left_spec
        self._right_spec = right_spec
        self._add_implied = add_implied
        self.outermost_inversion = None

    def __str__(self):
        if self._operator is None:
            return str(self._left_spec)
        else:
            return "%s %s %s" % (str(self._left_spec), self._operator,
                                 str(self._right_spec))

    def evaluate(self, session, models=None, *, order_implicit_atoms=False, add_implied=None, **kw):
        """Return results of evaluating atom specifier for given models.

        Parameters
        ----------
        session : chimerax.core.session.Session instance
            The session in which to evaluate atom specifier.
        models : list of chimerax.core.models.Model instances
            Defaults to None, which uses all models in 'session'.
        order_implicit_atoms : whether to order atoms that aren't
            explicitly specified (e.g. ":5") [which can be costly]
        **kw : keyword arguments
            If 'models' is None, 'kw' is passed through to call to
            'session.models.list' to generate the model list.

        Returns
        -------
        Objects instance
            Instance containing data (atoms, bonds, etc) that match
            this atom specifier.
        """
        # print("evaluate:", str(self))
        if add_implied is None:
            add_implied = self._add_implied
        if models is None:
            models = session.models.list(**kw)
            models.sort(key=lambda m: m.id)
        if self._operator is None:
            results = self._left_spec.evaluate(
                session, models, top=False, ordered=order_implicit_atoms, add_implied=add_implied)
            if self.outermost_inversion is None:
                self.outermost_inversion = isinstance(self._left_spec, _Invert)
                if self.outermost_inversion:
                    only_fully_selected_bonds(results)
        elif self._operator == '|':
            left_results = self._left_spec.evaluate(
                session, models, top=False, ordered=order_implicit_atoms, add_implied=add_implied)
            right_results = self._right_spec.evaluate(
                session, models, top=False, ordered=order_implicit_atoms, add_implied=add_implied)
            from ..objects import Objects
            results = Objects.union(left_results, right_results)
            if self.outermost_inversion is None:
                self.outermost_inversion = False
        elif self._operator == '&':
            left_results = self._left_spec.evaluate(
                session, models, top=False, ordered=order_implicit_atoms, add_implied=add_implied)
            if add_implied:
                add_implied_bonds(left_results)
            right_results = self._right_spec.evaluate(
                session, models, top=False, ordered=order_implicit_atoms, add_implied=add_implied)
            if add_implied:
                add_implied_bonds(right_results)
            from ..objects import Objects
            results = Objects.intersect(left_results, right_results)
            if self.outermost_inversion is None:
                self.outermost_inversion = False
        else:
            raise RuntimeError("unknown operator: %s" % repr(self._operator))
        if add_implied:
            add_implied_bonds(results)
        return results

    def find_matches(self, session, models, results, ordered, *, add_implied=None):
        if add_implied is None:
            add_implied = self._add_implied
        my_results = self.evaluate(session, models, top=False, ordered=ordered, add_implied=add_implied)
        results.combine(my_results)
        return results


def add_implied_bonds(objects):
    atoms = objects.atoms
    objects.add_bonds(atoms.intra_bonds)
    objects.add_pseudobonds(atoms.intra_pseudobonds)


def only_fully_selected_bonds(objects):
    from chimerax.atomic import Bonds, Pseudobonds
    from numpy import array
    intra_bond_ptrs = set(objects.atoms.intra_bonds.pointers)
    objects.set_bonds(Bonds(array(list(set(objects.bonds.pointers) & intra_bond_ptrs))))
    intra_pbond_ptrs = set(objects.atoms.intra_pseudobonds.pointers)
    objects.set_pseudobonds(Pseudobonds(array(list(set(objects.pseudobonds.pointers) & intra_pbond_ptrs))))


#
# Selector registration and use
#
# TODO: Registered session-specific selectors should go into a
# state manager class, but I have not figured out how to save
# callable objects in states.
#
_selectors = {}


class _Selector:

    def __init__(self, name, value, user, desc, atomic):
        self.name = name
        self.value = value
        self.user_defined = user
        self.atomic = atomic
        self._description = desc

    def description(self, session):
        if self._description:
            return self._description
        from chimerax.core.objects import Objects
        sel = self.value
        if callable(sel):
            if self.user_defined:
                value = "[Function]"
            else:
                value = "[Built-in]"
        elif isinstance(sel, Objects):
            if sel.empty():
                deregister_selector(self.name, session.logger)
                return None
            title = []
            if sel.num_atoms:
                title.append("%d atoms" % sel.num_atoms)
            if sel.num_bonds:
                title.append("%d bonds" % sel.num_bonds)
            if len(sel.models) > 1:
                title.append("%d models" % len(sel.models))
            if not title:
                if sel.num_pseudobonds:
                    title.append("%d pseudobonds" % sel.num_pseudobonds)
            if not title:
                if sel.model_instances:
                    title.append("%d model instances" % len(sel.model_instances))
            value = "[%s]" % ', '.join(title)
        else:
            value = str(sel)
        return value


def register_selector(name, value, logger, *,
                      user=False, desc=None, atomic=True):
    """Register a (name, value) pair as an atom specifier selector.

    Parameters
    ----------
    name : str
        Selector name, preferably without whitespace.
    value : callable object or instance of Objects
        Selector value.  If a callable object, called as
        'value(session, models, results)' where 'models'
        are chimerax.core.models.Model instances and
        'results' is an Objects instance; the callable
        is expected to add selected items to 'results'.
        If an Objects instance, items in value are merged
        with already selected items.
    logger : instance of chimerax.core.logger.Logger
        Current logger.
    user : boolean
        Boolean value indicating whether name is considered
        user-defined or not.
    desc : string
        Selector description.  Returned by get_selector_description().
        If not supplied, a generic description will be generated.
    atomic : boolean
        Boolean value indicating atoms may be selected using selector.
        Non-atomic selectors will not appear in Basic Actions tool.
    """
    if not name[0].isalpha():
        logger.warning("Not registering illegal selector name \"%s\"" % name)
        return
    for c in name[1:]:
        if not c.isalnum() and c not in "-+_":
            logger.warning("Not registering illegal selector name \"%s\"" % name)
            return
    _selectors[name] = _Selector(name, value, user, desc, atomic)
    from ..toolshed import get_toolshed
    ts = get_toolshed()
    if ts:
        ts.triggers.activate_trigger("selector registered", name)


def deregister_selector(name, logger=None):
    """Deregister a name as an atom specifier selector.

    Parameters
    ----------
    name : str
        Previously registered selector name.
    logger : instance of chimerax.core.logger.Logger
        Current logger.

    Raises
    ------
    KeyError
        If name is not registered.
    """
    try:
        del _selectors[name]
    except KeyError:
        if logger:
            logger.warning("deregistering unregistered selector \"%s\"" % name)
    else:
        from ..toolshed import get_toolshed
        ts = get_toolshed()
        if ts:
            ts.triggers.activate_trigger("selector deregistered", name)


def check_selectors(trigger_name, model):
    # Called when models are closed so that selectors whose values
    # are Object instances can be cleared if they contain no models
    from ..objects import Objects
    empty = []
    for name, sel in _selectors.items():
        if isinstance(sel.value, Objects):
            for m in sel.value.models:
                if not m.deleted and m.id is not None:
                    break
            else:
                empty.append(name)
    for name in empty:
        deregister_selector(name)


def list_selectors():
    """Return a list of all registered selector names.

    Returns
    -------
    iterator yielding str
        Iterator that yields registered selector names.
    """
    return _selectors.keys()


def get_selector(name):
    """Return value associated with registered selector name.

    Parameters
    ----------
    name : str
        Previously registered selector name.

    Returns
    -------
    Callable object, Objects instance, or None.
        Callable object if name was registered; None, if not.
    """
    try:
        return _selectors[name].value
    except KeyError:
        return None


def is_selector_user_defined(name):
    """Return whether selector name is user-defined.

    Parameters
    ----------
    name : str
        Previously registered selector name.

    Returns
    -------
    Boolean
        Whether selector name is user-defined.
    """
    return _selectors[name].user_defined


def is_selector_atomic(name):
    """Return whether selector may select any atoms.

    Parameters
    ----------
    name : str
        Previously registered selector name.

    Returns
    -------
    Boolean
        Whether selector name may select any atoms.
    """
    return _selectors[name].atomic


def get_selector_description(name, session):
    """Return description for selector.

    Parameters
    ----------
    session : instance of chimerax.core.session.Session
        Session in which the name may be used.  If None, name is global.
    name : str
        Previously registered selector name.

    Returns
    -------
    string
        Description of selector.  Registered description is
        used when available; otherwise, description is generated
        from the selector value.
    """
    return _selectors[name].description(session)


def everything(session):
    """Return AtomSpec that matches everything.

    Parameters
    ----------
    session : instance of chimerax.core.session.Session
        Session in which the name may be used.  If None, name is global.

    Returns
    -------
    AtomSpec instance
        An AtomSpec instance that matches everything in session.
    """
    return AtomSpecArg.parse('#*', session)[0]


def all_objects(session):
    '''Return Objects that matches everything.'''
    return everything(session).evaluate(session)
