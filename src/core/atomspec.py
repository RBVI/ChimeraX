# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
atomspec: atom specifier cli annotation and evaluation
======================================================

The 'atomspec' module provides three classes:

- AtomSpecArg : command line argument annotation class.
- AtomSpec : atom specifier class.
- AtomSpecResults : atom specifier evaluation results class.

AtomSpecArg is a cli type annotation and is used to describe an
argument of a function that is registered with the cli module.  When
the registered function is called, the argument corresponding to
the AtomSpecArg is an instance of AtomSpec, which contains a parsed
version of the input atom specifier.  The model elements (atoms,
bonds, models, etc) that match an AtomSpec may be found by calling
the 'evaluate' method which returns an instance of AtomSpecResults.
Each type of model elements may be accessed as an attribute of the
AtomSpecResults instance.

Selectors
---------

A (name, function) pair may be registered as a 'selector' in an
atom specifier.  The selectors may either be global (e.g., chemical
groups) or session-specific (e.g., active site).  The selector
name may appear wherever a model, residue or atom string does.
The selector function is called when the atom specifier is
evaluated and is expected to fill in an AtomSpecResults instance.

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
    """Command line type annotation for atom specifiers.

    See cli documentation for details on type annotations.

    """
    name = "an atom specifier"

    @staticmethod
    def parse(text, session):
        token, text, rest = _next_atomspec(text)
        from ._atomspec import _atomspecParser
        parser = _atomspecParser(parseinfo=True)
        semantics = _AtomSpecSemantics(session)
        from grako.exceptions import FailedParse
        try:
            ast = parser.parse(token, "atom_specifier", semantics=semantics)
        except FailedParse as e:
            raise ValueError(str(e))
        if ast.parseinfo.endpos != len(token):
            # TODO: better error message on syntax error
            raise ValueError("mangled atom specifier")
        return ast, text, rest


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


class _AtomSpecSemantics:
    """Semantics class to convert basic ASTs into AtomSpec instances."""
    def __init__(self, session):
        self._session = session

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

    def selector_name(self, ast):
        if (get_selector(self._session, ast.name) is None
            and get_selector(None, ast.name) is None):
                # TODO: generate better error message in cli
                raise ValueError("\"%s\" is not a selector name" % ast.name)
        return _SelectorName(ast.name)

    def model_list(self, ast):
        if ast.model_list is None:
            model_list = _ModelList(ast.model)
        else:
            model_list = ast.model_list
            model_list.append(ast.model)
        return model_list

    def model(self, ast):
        m = _Model(ast.hierarchy)
        if ast.parts is not None:
            for p in ast.parts:
                m.add(p)
        return m

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

    def model_parts(self, ast):
        if ast.parts is None:
            if ast.chain is None:
                return None
            else:
                return [ ast.chain ]
        else:
            parts = ast.parts
            if ast.chain is not None:
                parts.append(ast.chain)
            return parts

    def chain(self, ast):
        if ast.chain is None and ast.parts is None and ast.residue is None:
            return None
        if ast.chain is not None:
            c = ast.chain
        else:
            c = _Chain(ast.parts)
        if ast.residue is not None:
            c.add(ast.residue)
        return c

    def residue(self, ast):
        if ast.residue is None and ast.parts is None and ast.atom is None:
            return None
        if ast.residue is not None:
            r = ast.residue
        else:
            r = _Residue(ast.parts)
        if ast.atom is not None:
            r.add(ast.atom)
        return r

    def atom(self, ast):
        if ast.parts is None:
            return None
        else:
            return _Atom(ast.parts)

    def part_list(self, ast):
        if ast.part is None:
            return _PartList(ast.range)
        else:
            return ast.part.add(ast.range)

    def part_range_list(self, ast):
        return _Part(ast.start, ast.end)


class _ModelList(list):
    """Stores list of model hierarchies."""
    def __init__(self, h):
        super().__init__()
        self.append(h)

    def __str__(self):
        if not self:
            return "[empty]"
        return "".join(str(mr) for mr in self)

    def find_matches(self, session, model_list, results):
        for m in model_list:
            for h in self:
                if h.matches(session, m):
                    results.add_model(m)
                    break


class _ModelHierarchy(list):
    """Stores list of model ranges in hierarchy order."""
    def __init__(self, mrl):
        super().__init__()
        self.append(mrl)

    def __str__(self):
        if not self:
            return "[empty]"
        return ".".join(str(mr) for mr in self)

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
        super().__init__()
        self.append(mr)

    def __str__(self):
        if not self:
            return "[empty]"
        return ",".join(str(mr) for mr in self)

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
            if self.start != '*' and mid < self.start:
                return False
            return self.end == '*' or mid <= self.end


class _SubPart:
    """Stores part list for one item and subparts of the item."""
    def __init__(self, my_parts):
        self.my_parts = my_parts
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
        # print("_SubPart.__str__", self.__class__, r)
        return r

    def add(self, subpart):
        if self.sub_parts is None:
            self.sub_parts = [subpart]
        else:
            self.sub_parts.append(subpart)


class _Model(_SubPart):
    """Stores model part list and atom spec."""
    Symbol = '#'

    def matches(self, session, model):
        return self.my_parts.matches(session, model)


class _Chain(_SubPart):
    """Stores residue part list and atom spec."""
    Symbol = '/'


class _Residue(_SubPart):
    """Stores residue part list and atom spec."""
    Symbol = ':'


class _Atom(_SubPart):
    """Stores residue part list and atom spec."""
    Symbol = '@'


class _PartList:
    """Stores a part list (sub-parts of models)."""
    def __init__(self, part_range):
        self.parts = [part_range]

    def __str__(self):
        return ','.join([str(p) for p in self.parts])

    def add(self, part_range):
        self.parts.append(part_range)


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


class _SelectorName:
    """Stores a single selector name."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def find_matches(self, session, models, results):
        f = get_selector(session, self.name) or get_selector(None, self.name)
        if f:
            f(session, models, results)


class _Term:
    """A term in an atom specifier."""
    def __init__(self, spec):
        self._specifier = spec

    def __str__(self):
        return str(self._specifier)

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

    def __str__(self):
        if self._operator is None:
            return str(self._left_term)
        else:
            return "%s %s %s" % (str(self._left_term), self_operator,
                                 str(self._right_term))

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
        print("evaluate:", str(self))
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
        self._atoms = None

    def add_model(self, m):
        """Add model to atom spec results."""
        self._models.add(m)

    def add_atoms(self, atom_blob):
        """Add atoms to atom spec results."""
        if self._atoms is None:
            self._atoms = atom_blob
        else:
            self._atoms.merge(atom_blob)

    @property
    def models(self):
        return self._models

    @staticmethod
    def _Union(left, right):
        atom_spec = AtomSpecResults()
        atom_spec._models = left._models | right._models
        if left._atoms is None:
            atom_spec._atoms = right._atoms
        elif right._atoms is None:
            atom_spec._atoms = left._atoms
        else:
            atoms_spec._atoms = right._atoms.merge(left._atoms)
        return atom_spec

    @staticmethod
    def _Intersect(left, right):
        atom_spec = AtomSpecResults()
        atom_spec._models = left._models & right._models
        if left._atoms is None:
            atom_spec._atoms = right._atoms
        elif right._atoms is None:
            atom_spec._atoms = left._atoms
        else:
            # TODO: implement
            raise RuntimeError("Atom spec intersection not implemented yet")
        return atom_spec

#
# Selector registration and use
#
# TODO: Registered session-specific selectors should go into a
# state manager class, but I have not figured out how to save
# callable objects in states.
#
from .session import State
_selectors = {}


def _get_selector_map(session):
    if session is None:
        return _selectors
    else:
        try:
            return session.atomspec_selectors
        except AttributeError:
            d = {}
            session.atomspec_selectors = d
            return d


def register_selector(session, name, func):
    """Register a (name, func) pair as an atom specifier selector.

    Parameters
    ----------
    session : instance of chimera.core.session.Session
        Session in which the name may be used.  If None, name is global.
    name : str
        Selector name, preferably without whitespace.
    func : callable object
        Selector evaluation function, called as 'func(session, models, results)'
        where 'models' are chimera.core.models.Model instances and
        'results' is an AtomSpecResults instance.

    """
    _get_selector_map(session)[name] = func


def deregister_selector(session, name):
    """Deregister a name as an atom specifier selector.

    Parameters
    ----------
    session : instance of chimera.core.session.Session
        Session in which the name may be used.  If None, name is global.
    name : str
        Previously registered selector name.

    Raises
    ------
    KeyError
        If name is not registered.

    """
    del _get_selector_map(session)[name]


def list_selectors(session):
    """Return a list of all registered selector names.

    Parameters
    ----------
    session : instance of chimera.core.session.Session
        Session in which the name may be used.  If None, name is global.

    Returns
    -------
    iterator yielding str
        Iterator that yields registered selector names.

    """
    return _get_selector_map(session).keys()


def get_selector(session, name):
    """Return function associated with registered selector name.

    Parameters
    ----------
    session : instance of chimera.core.session.Session
        Session in which the name may be used.  If None, name is global.
    name : str
        Previously registered selector name.

    Returns
    -------
    Callable object or None.
        Callable object if name was registered; None, if not.

    """
    return _get_selector_map(session).get(name, None)
