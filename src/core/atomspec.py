# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
atomspec: atom specifier cli annotation and evaluation
======================================================

TODO: Stubs for now.

AtomSpecArg arguments should always be optional because
not providing and atom specifier is the same as choosing
all atoms.

"""

from .cli import Annotation
import re

_double_quote = re.compile(r'"(.|\")*?"(\s|$)')
_operator = re.compile(r'\s+[&~|]+\s+')
_parser = None
_parser_helper = None


class AtomSpecArg(Annotation):
    """Annotation for atom specifiers"""
    name = "an atom specifier"

    @staticmethod
    def parse(text, session):
        token, text, rest = _next_atomspec(text)
        parser, helper = _get_parser()
        from grako.exceptions import FailedParse
        try:
            ast = parser.parse(token, "model_list", semantics=helper)
        except FailedParse as e:
            raise ValueError(str(e))
        if ast.parseinfo.endpos != len(token):
            # TODO: better error message on syntax error
            raise ValueError("mangled atom specifier")
        return ast, text, rest


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


def _get_parser():
    global _parser, _parser_helper
    if _parser is None:
        from ._atomspec import _atomspecParser
        _parser = _atomspecParser(parseinfo=True)
        _parser_helper = _AtomSpecParserHelper()
    return _parser, _parser_helper


class _AtomSpecParserHelper:
    """Helper class to convert basic ASTs into AtomSpec instances."""
    def model_list(self, ast):
        if ast.model_list is None:
            model_list = _ModelList(ast.model)
        else:
            model_list = ast.model_list
            model_list.insert(0, ast.model)
        try:
            model_list.parseinfo = ast.parseinfo
        except AttributeError:
            pass
        return model_list

    def model(self, ast):
        return ast[1]

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


class _ModelList(list):
    """Stores list of model hierarchies."""
    def __init__(self, h):
        super().__init__(self)
        self.append(h)

    def __repr__(self):
        if not self:
            return "[empty]"
        return "#" + "".join(repr(mr) for mr in self)

    def evaluate(self, model_list, wanted):
        for m in model_list:
            for h in self:
                if h.matches(m):
                    wanted.add(m)
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

    def matches(self, model):
        for i, mrl in enumerate(self):
            try:
                mid = model.id[i]
            except IndexError:
                mid = 1
            if mrl.matches(mid):
                return True
        return False


class _ModelRangeList(list):
    """Stores list of model ranges and evaluates against a list of models."""
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
    """Stores models of ranges and evaluates against a list of models."""
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
            return self.end == '*' or mid <= self.start
