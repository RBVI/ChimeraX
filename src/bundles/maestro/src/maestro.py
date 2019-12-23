# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Read Maestro ASCII format file"""


IndexAttribute = "i_chimera_index"


class MaestroString:

    import re
    REIdent = re.compile(r"(?P<value>[A-Za-z_][^\s{}[\]]+)")
    REString = re.compile(r'\"(?P<value>([^"\\]|\\.)*)\"')
    REString2 = re.compile(r"(?P<value>[^\s{}[\]]+)")
    REValue = re.compile(r"(?P<value>\S+)")
    REComment = re.compile(r"#.*")
    REFloat = re.compile(r"^(?P<value>\d+(?=[.eE])(\.\d*)?([eE]\d+))$")
    REInteger = re.compile(r"^(?P<value>\d+)$")
    del re

    TokenEOF = 0
    TokenEndList = 1
    TokenIdent = 2
    TokenString = 3
    TokenValue = 3

    MatchTokenList = [
        ( TokenIdent,  REIdent   ),
        ( TokenString, REString  ),
        ( TokenString, REString2 ),
    ]
    MatchValueList = [
        ( TokenValue, REString ),
        ( TokenValue, REValue  ),
    ]

    def __init__(self, data, parse_contents=True):
        """Read Maestro format data from string"""

        self.data = data
        self.parse_contents = parse_contents

    def __iter__(self):
        self.length = len(self.data)
        self.index = 0
        self.lineno = 1
        self.eof = False
        self._next_token()
        # Always parse first block since it has the
        # version information
        block = self._read_block()
        if block:
            yield block
        if not self.parse_contents:
            # Reset index to start of token so
            # _read_unparsed_block starts at the right place
            self.index = self._tokenIndex
        while not self.eof:
            if self.parse_contents:
                block = self._read_block()
            else:
                block = self._read_unparsed_block()
            if block:
                yield block

    def estimate_block_count(self):
        """Return approximate number of "Molecule" blocks."""
        count = 0
        start = 0
        while True:
            start = self.data.find("f_m_ct", start)
            if start == -1:
                return count
            count += 1
            start += 6

    def _read_block(self, depth=0):
        """Read a single block and assign all attribute values."""
        #print "_read_block", depth

        token_type, token_value = self.token
        if token_type is self.TokenEOF:
            return None

        # Read block name and size if present
        if token_type is self.TokenIdent:
            name = token_value
            token_type, token_value = self._next_token()
            if token_type != '[':
                size = 1
                multivalued = False
            else:
                token_type, token_value = self._next_token()
                size = self._get_integer(token_value)
                multivalued = True
                token_type, token_value = self._next_token()
                if token_type != ']':
                    self._syntax_error("unclosed block size")
                token_type, token_value = self._next_token()
        else:
            name = None
            size = 1
            multivalued = False
        block = Block(name, size)

        # Open block
        if token_type != '{':
            if name is not None:
                self._syntax_error("missing block open brace")
            else:
                return None

        # Read block attribute names
        attrNames = list()
        while not self.eof:
            token_type, token_value = self._next_token()
            if token_type == self.TokenEndList:
                break
            attrNames.append(token_value)
            #print "attribute name:", token_value
        if multivalued:
            # For multivalued blocks, the first attribute
            # column is always the index
            attrNames.insert(0, IndexAttribute)
            #print "insert attribute name:", IndexAttribute
        #print "number of rows:", size

        # Read block attribute values
        for row in range(size):
            for i in range(len(attrNames)):
                token_type, token_value = self._next_value()
                if token_type in (self.TokenIdent,
                            self.TokenString):
                    #print "set", row, attrNames[i], token_value
                    block.set_attribute(self, row,
                                attrNames[i],
                                token_value)
                else:
                    self._syntax_error("data value expected")
        token_type, token_value = self._next_token()
        if token_type == self.TokenEndList:
            self._next_token()

        # Read subblocks
        while not self.eof:
            subblock = self._read_block(depth + 1)
            if subblock is None:
                break
            block.add_sub_block(subblock)

        # Close block
        token_type, token_value = self.token
        if token_type != '}':
            self._syntax_error("missing block close brace")
        self._next_token()
        return block

    def _next_token(self):
        self._skip_whitespace()
        self._tokenIndex = self.index
        if self.index >= self.length:
            self.eof = True
            self.token = (self.TokenEOF, "<EOF>")
            return self.token
        if self.data[self.index] in "{}[]":
            c = self.data[self.index]
            self.index += 1
            self.token = (c, c)
            return self.token
        if self.data[self.index:self.index+3] == ":::":
            self.index += 3
            self.token = (self.TokenEndList, ":::")
            return self.token
        m = self.REComment.match(self.data, self.index)
        if m is not None:
            self.index = m.end()
            return self._next_token()
        for token_type, pattern in self.MatchTokenList:
            m = pattern.match(self.data, self.index)
            if m is not None:
                self.index = m.end()
                self.token = (token_type, m.group("value"))
                return self.token
        self._syntax_error("unrecognized token")

    def _next_value(self):
        self._skip_whitespace()
        if self.index >= self.length:
            self.eof = True
            self.token = (self.TokenEOF, "<EOF>")
            return self.token
        if self.data[self.index:self.index+3] == ":::":
            self.index += 3
            self.token = (self.TokenEndList, ":::")
            return self.token
        for token_type, pattern in self.MatchValueList:
            m = pattern.match(self.data, self.index)
            if m is not None:
                self.index = m.end()
                self.token = (token_type, m.group("value"))
                return self.token
        self._syntax_error("unrecognized value")

    def _skip_whitespace(self):
        while self.index < self.length:
            c = self.data[self.index]
            if c == '\n':
                self.lineno += 1
            if not c.isspace():
                break
            self.index += 1

    def _read_unparsed_block(self):
        """Read a single block as a chunk of text."""

        #print "_read_unparsed_block"
        self._skip_whitespace()
        if self.index >= self.length:
            self.eof = True
            return None
        start = n = self.index
        depth = 0
        while n < self.length:
            if self.data[n] == '{':
                n += 1
                depth += 1
            elif self.data[n] == '}':
                n += 1
                depth -= 1
                if depth == 0:
                    break
            elif self.data[n] in "'\"":
                quote = self.data[n]
                n += 1
                while n < self.length:
                    if self.data[n] == '\\':
                        n += 2
                    elif self.data[n] == quote:
                        n += 1
                        break
                    else:
                        n += 1
            elif self.data[n] == '\n':
                self.lineno += 1
                n += 1
            else:
                n += 1
        else:
            self._syntax_error("unclosed block")
        self.index = n
        self.eof = self.index >= self.length
        return UnparsedBlock(self.data[start:n])

    def _get_value(self, name, value):
        try:
            return get_value(name, value)
        except ValueError:
            raise ValueError(
                "value (%s) does not match attribute (%s) type"
                % (value, name))

    def _get_integer(self, value):
        try:
            return int(value)
        except ValueError:
            raise ValueError(
                "expected integer and got \"%s\"" % value)

    def _syntax_error(self, msg):
        raise SyntaxError("line %d: %s (current token: %s %s)" %
                    (self.lineno, msg, self.token[0],
                        self.token[1]))


class Block:

    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.attribute_rows = [ dict() for i in range(size) ]
        self.sub_blocks = list()
        self.attribute_names = list()

    def add_sub_block(self, block):
        self.sub_blocks.append(block)

    def get_sub_block(self, name):
        for sb in self.sub_blocks:
            if sb.name == name:
                return sb
        return None

    def set_attribute(self, mb, row, name, value):
        if row == 0:
            self.attribute_names.append(name)
        attrs = self.attribute_rows[row]
        # Use "mb._get_value" instead of "get_value" to 
        # generate better error message
        attrs[name] = mb._get_value(name, value)

    def get_attribute(self, name, row=0):
        return self.attribute_rows[row][name]

    def get_attribute_map(self, row=0):
        return self.attribute_rows[row]

    def write(self, f, indent=0):
        prefix = " " * indent
        contentPrefix = " " * (indent + 2)
        if self.name:
            name = self.name
            sep = " "
        else:
            name = ""
            sep = ""
        if self.size > 1:
            size = "[%d]" % self.size
        else:
            size = ""
        print("%s%s%s%s{" % (prefix, name, size, sep), file=f)

        for name in self.attribute_names:
            if name == IndexAttribute:
                continue
            print("%s%s" % (contentPrefix, name), file=f)
        print("%s:::" % contentPrefix, file=f)
        for row in self.attribute_rows:
            f.write(contentPrefix)
            print(' '.join([printable_value(name, row[name])
                            for name in self.attribute_names]), file=f)

        for block in self.sub_blocks:
            block.write(f, indent + 2)

        print("%s}" % prefix, file=f)
        print(file=f)


class UnparsedBlock:

    def __init__(self, text):
        self.text = text

    def write(self, f, indent=0):
        print(self.text, file=f)
        print(file=f)


class MaestroFile(MaestroString):

    def __init__(self, f, *args, **kw):
        if isinstance(f, str):
            # Assume string is filename
            with open(f) as fi:
                MaestroString.__init__(self, fi.read(), *args, **kw)
        else:
            MaestroString.__init__(self, f.read(), *args, **kw)


def get_value(name, value):
    """Convert text string into value based on attribute name"""
    if value == "<>":
        return None
    if name[0] == 'i':
        return int(value)
    elif name[0] == 'r':
        return float(value)
    elif name[0] == 's':
        return value
    elif name[0] == 'b':
        return int(value) != 0
    else:
        raise ValueError("unknown attribute type: %s" % name)


def printable_value(name, value):
    """Convert value into text string based on attribute name"""
    if value is None:
        return "<>"
    if name[0] == 'i':
        return "%d" % value
    elif name[0] == 'r':
        return "%g" % value
    elif name[0] == 's':
        v = value.replace('\\', '\\\\').replace('"', '\\"')
        if v != value:
            return '"%s"' % v
        for c in v:
            if c.isspace():
                need_quotes = True
                break
        else:
            need_quotes = len(v) == 0
        if not need_quotes:
            return v
        else:
            return '"%s"' % v
    elif name[0] == 'b':
        return "%d" % value
    else:
        raise ValueError("unknown attribute type: %s" % name)


if __name__ == "__main__":
    print(MaestroFile("../test-data/kegg_dock5.mae"))
