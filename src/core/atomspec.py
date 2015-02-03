# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
atomspec: atom specifier cli annotation and evaluation
======================================================

The 'atomspec' module provides three classes:

- AtomSpecArg : command line argument annotation class.
- AtomSpec : atom specifier class.
- AtomSpecResults : atom specifier evaluation results class.

AtomSpecArg is a cli type annotation and is used to
describe an argument of a function that is registered
with the cli module.
When the registered function is called, the argument
corresponding to the AtomSpecArg is an instance of
AtomSpec, which contains a parsed version of the
input atom specifier.
The model elements (atoms, bonds, models, etc)
that match an AtomSpec may be found by calling
the 'evaluate' method which returns an instance of
AtomSpecResults.
Each type of model elements may be accessed as an
attribute of the AtomSpecResults instance.

Example
-------

Here is an example of a function that may be registered with cli:

    from chimera.core import cli, atomspec

    def move(session, by, modelspec=None):
        spec = modelspec.evaluate(session)
        import numpy
        by_vector = numpy.array(by)
        from chimera.core.geometry import place
        translation = place.translation(by_vector)
        for m in spec.models:
            m.position = translation * m.position
            m.update_graphics()
    move_desc = cli.CmdDesc(required=[("by", cli.Float3Arg)],
                            optional=[("modelspec", atomspec.AtomSpecArg)])

Notes
-----

AtomSpecArg arguments should always be optional because
not providing and atom specifier is the same as choosing
all atoms.

"""

from .cli import Annotation


class AtomSpecArg(Annotation):
    """Annotation for atom specifiers"""
    name = "an atom specifier"
    _parser = None
    _semantics = None

    @staticmethod
    def parse(text, session):
        token, text, rest = _next_atomspec(text)
        parser, helper = AtomSpecArg._get_parser()
        from grako.exceptions import FailedParse
        try:
            ast = parser.parse(token, "atom_specifier", semantics=helper)
        except FailedParse as e:
            raise ValueError(str(e))
        if ast.parseinfo.endpos != len(token):
            # TODO: better error message on syntax error
            raise ValueError("mangled atom specifier")
        return ast, text, rest

    @classmethod
    def _get_parser(cls):
        if cls._parser is None:
            from ._atomspec import _atomspecParser
            cls._parser = _atomspecParser(parseinfo=True)
            cls._semantics = _AtomSpecParserSemantics()
        return cls._parser, cls._semantics


#
# Lexical analysis functions
#
import re

_double_quote = re.compile(r'"(.|\")*?"(\s|$)')
_operator = re.compile(r'\s+[&~|]+\s+')


def _next_atomspec(text):
    # Modeled after .cli._next_token()
    #
    # Return a 3-tuple of first argument in text, the actual text used,
    # and the rest of the text.
    #
    # Arguments may be quoted, in which case the text between
    # the quotes is returned.
    assert(text and not text[0].isspace())
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
            raise ValueError("incomplete quoted text")
        from .cli import unescape
        token = unescape(token)
    else:
        # Do space collapsing around operators
        eol = len(text)
        if text[0] == '~':
            token = "~"
            end = 1
            while end < eol:
                if not text[end].isspace():
                    break
                else:
                    end += 1
        else:
            token = ""
            end = 0
        while end < eol:
            # Look for the next operator.  If there is a space
            # between here and the next operator, then the space
            # terminates the atom specifier.
            m = _operator.search(text, end)
            if m:
                # An operator appears later in the line
                group_start = m.start()
                n = _find_intervening_space(text, end, group_start)
                if n is None:
                    # No intervening spaces, operator is part of atomspec
                    token += text[end:group_start] + m.group()
                    end = m.end()
                else:
                    # Add remainder of atomspec and terminate
                    token += text[end:n]
                    end = n
                    break
            else:
                # No operator appears in rest of line
                n = _find_intervening_space(text, end, eol)
                if n is None:
                    # No intervening spaces, rest of line is part of atomspec
                    token += text[end:]
                    end = eol
                else:
                    # Add remainder of atomspec and terminate
                    token += text[end:n]
                    end = n
                break
    return token, text[:end], text[end:]


def _find_intervening_space(text, start, end):
    for n in range(start, end):
        if text[n].isspace():
            return n
    else:
        return None


#
# Parsing functions and classes
#


class _AtomSpecParserSemantics:
    """Semantics class to convert basic ASTs into AtomSpec instances."""
    def atom_specifier(self, ast):
        atom_spec = AtomSpec(ast.operator, ast.left, ast.right)
        try:
            atom_spec.parseinfo = ast.parseinfo
        except AttributeError:
            pass
        return atom_spec

    def as_term(self, ast):
        if ast.term is not None:
            return ast.term
        elif ast.models is not None:
            return _Term(ast.models)
        else:
            return _Term(ast.selector)

    def model_list(self, ast):
        if ast.model_list is None:
            model_list = _ModelList(ast.model)
        else:
            model_list = ast.model_list
            model_list.insert(0, ast.model)
        return model_list

    def model(self, ast):
        if ast.hierarchy is not None:
            return ast.hierarchy
        else:
            return ast.name

    def model_hierarchy(self, ast):
        if ast.hierarchy is None:
            hierarchy = _ModelHierarchy(ast.range_list)
        else:
            hierarchy = ast.hierarchy
            hierarchy.insert(0, ast.range_list)
        return hierarchy

    def model_range_list(self, ast):
        if ast.range_list is None:
            range_list = _ModelRangeList(ast.range)
        else:
            range_list = ast.range_list
            range_list.insert(0, ast.range)
        return range_list

    def model_range(self, ast):
        return _ModelRange(ast.start, ast.end)

    def model_spec(self, ast):
        if ast.number is not None:
            return int(ast.number)
        elif ast.star is not None:
            return '*'
        else:
            return None

    def selector_name(self, ast):
        return _SelectorName(ast.name)


class _ModelList(list):
    """Stores list of model hierarchies."""
    def __init__(self, h):
        super().__init__(self)
        self.append(h)

    def __repr__(self):
        if not self:
            return "[empty]"
        return "#" + "".join(repr(mr) for mr in self)

    def find_matches(self, session, model_list, results):
        for m in model_list:
            for h in self:
                if h.matches(session, m):
                    results._add_model(m)
                    break


class _ModelHierarchy(list):
    """Stores list of model ranges in hierarchy order."""
    def __init__(self, mrl):
        super().__init__(self)
        self.append(mrl)

    def __repr__(self):
        if not self:
            return "[empty]"
        return ".".join(repr(mr) for mr in self)

    def matches(self, session, model):
        for i, mrl in enumerate(self):
            try:
                mid = model.id[i]
            except IndexError:
                mid = 1
            if mrl.matches(mid):
                return True
        return False


class _ModelRangeList(list):
    """Stores a list of model ranges."""
    def __init__(self, mr):
        super().__init__(self)
        self.append(mr)

    def __repr__(self):
        if not self:
            return "[empty]"
        return ",".join(repr(mr) for mr in self)

    def matches(self, mid):
        for mr in self:
            if mr.matches(mid):
                return True
        return False


class _ModelRange:
    """Stores a single model range."""
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
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
            if self.start != '*' and mid < self.start:
                return False
            return self.end == '*' or mid <= self.end


class _SelectorName:
    """Stores a single selector name."""
    def __init__(self, name):
        self.name = name

    def find_matches(self, session, models, results):
        results = AtomSpecResults()
        # TODO: implement
        print("_SelectorName.match", self.name)
        return results


class _Term:
    """A term in an atom specifier."""
    def __init__(self, spec):
        self._specifier = spec

    def evaluate(self, session, models):
        """Return AtomSpecResults for model elements that match."""
        results = AtomSpecResults()
        self._specifier.find_matches(session, models, results)
        return results


class AtomSpec:
    """AtomSpec instances store and evaluate atom specifiers.

    An AtomSpec instance, returned by AtomSpecArg arguments in
    cli command functions, keeps track of an atom specifier.
    When evaluated, the model elements that match the specifier
    are returned.
    """
    def __init__(self, operator, left_term, right_term):
        self._operator = operator
        self._left_term = left_term
        self._right_term = right_term

    def evaluate(self, session, models=None, **kw):
        """Return results of evaluating atom specifier for given models.

        Parameters
        ----------
        session : chimera.core.session.Session instance
            The session in which to evaluate atom specifier.
        models : list of chimera.core.models.Model instances
            Defaults to None, which uses all models in 'session'.
        **kw : keyword arguments
            If 'models' is None, 'kw' is passed through to call to
            'session.models.list' to generate the model list.

        Returns
        -------
        AtomSpecResults instance
            Instance containing data (atoms, bonds, etc) that match 
            this atom specifier.
        """
        if models is None:
            models = session.models.list(**kw)
        if self._operator is None:
            results = self._left_term.evaluate(session, models)
        elif self._operator == '|':
            left_results = self._left_term.evaluate(session, models)
            right_results = self._right_term.evaluate(session, models)
            results = AtomSpecResults._Union(left_results, right_results)
        elif self._operator == '&':
            left_results = self._left_term.evaluate(session, models)
            right_results = self._right_term.evaluate(session, models)
            results = AtomSpecResults._Intersect(left_results, right_results)
        else:
            raise RuntimeError("unknown operator: %s" % repr(self._operator))
        return results

class AtomSpecResults:
    """AtomSpecResults store evaluation results from AtomSpec.

    An AtomSpecResults instance, returned by calls to
    'AtomSpec.evaluate', keeps track of model elements that
    match the atom specifier.

    Parameters
    ----------
    models : readonly list of chimera.core.models.Model
        List of models that matches the atom specifier
    """
    def __init__(self):
        self._models = set()

    def _add_model(self, m):
        self._models.add(m)

    @property
    def models(self):
        return self._models

    @staticmethod
    def _Union(left, right):
        atom_spec = AtomSpecResults()
        atom_spec._models = left._models | right._models
        return atom_spec

    @staticmethod
    def _Intersect(left, right):
        atom_spec = AtomSpecResults()
        atom_spec._models = left._models & right._models
        return atom_spec
