from chimerax.log.parse import tokenize, Html, Command, Disclosure, select_command_variant

_cmd = ('<div class="cxcmd"><div class="cxcmd_as_doc"><a href="help:user/commands/open.html">open</a>'
        ' 3fx2</div><div class="cxcmd_as_cmd"><a href="cxcmd:open 3fx2">open 3fx2</a></div></div>')


def test_tokenize_plain():
    assert tokenize("Opened 3fx2<br>\n", base=7) == [Html("Opened 3fx2<br>\n", 7)]


def test_tokenize_command():
    pieces = tokenize("before" + _cmd + "after")
    assert [type(p) for p in pieces] == [Html, Command, Html]
    assert pieces[1].offset == len("before")
    assert pieces[1].html.startswith('<div class="cxcmd_as_doc">')
    assert pieces[1].html.endswith('</a></div>')
    assert pieces[2] == Html("after", len("before" + _cmd))


def test_tokenize_disclosure():
    html = ('<details style="background-color: ButtonFace"><summary>Log from Mon</summary>'
            + _cmd + '<details open><summary>inner</summary>x</details>end</details>tail')
    pieces = tokenize(html, base=100)
    assert len(pieces) == 2
    d = pieces[0]
    assert isinstance(d, Disclosure)
    assert d.offset == 100
    assert not d.open
    assert d.background == "ButtonFace"
    assert d.summary == "Log from Mon"
    assert html[d.content_offset - 100:].startswith(_cmd)
    assert d.content.endswith("x</details>end")
    assert pieces[1] == Html("tail", 100 + len(html) - len("tail"))

    inner = tokenize(d.content, d.content_offset)
    assert [type(p) for p in inner] == [Command, Disclosure, Html]
    assert inner[1].open
    assert inner[1].summary == "inner"
    assert inner[1].content == "x"
    assert html[inner[1].offset - 100:].startswith("<details open>")


def test_tokenize_unclosed():
    pieces = tokenize('<details><summary>s</summary>never closed')
    assert len(pieces) == 1
    assert pieces[0].content == "never closed"


def test_select_command_variant():
    assert select_command_variant(_cmd, exec_links=True) == (
        '<div class="cxcmd"><a href="cxcmd:open 3fx2">open 3fx2</a></div>')
    assert select_command_variant(_cmd, exec_links=False) == (
        '<div class="cxcmd"><a href="help:user/commands/open.html">open</a> 3fx2</div>')
    # syntax error echoes have no variants
    plain = '<a href="help:x">foo</a> bar'
    assert select_command_variant(plain, exec_links=True) == plain
