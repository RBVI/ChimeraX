from chimerax.log.parse import (
    tokenize, Html, Command, Disclosure, select_command_variant,
    starts_block, starts_table, strip_line_end,
    parse_style, replace_inline_blocks, css_length, css_border, drop_redundant_breaks,
)

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


def test_line_ends():
    assert strip_line_end("text<br>\n") == ("text", True)
    assert strip_line_end("text<BR/>") == ("text", True)
    assert strip_line_end("a<br>b<br>\n") == ("a<br>b", True)
    assert strip_line_end('<p style="color:red">warn</p>') == ('<p style="color:red">warn</p>', True)
    assert strip_line_end("<table><tr><td>x</td></tr></table>\n")[1]
    assert strip_line_end("partial ") == ("partial ", False)
    assert strip_line_end('<font color="red"><b>err</b></font>') == ('<font color="red"><b>err</b></font>', False)


def test_starts_block():
    assert starts_block("\n<p>x</p>")
    assert starts_block("<pre>x</pre>")
    assert starts_block("<table>")
    assert not starts_block("<b>x</b>")
    assert not starts_block("plain")
    assert not starts_block("<pref>")
    assert starts_table(" <table border=1>")
    assert not starts_table("<p><table>")


def test_parse_style():
    assert parse_style("width:1em; height:.6em; display:inline-block;") == {
        "width": "1em", "height": ".6em", "display": "inline-block"}
    assert parse_style("BACKGROUND-COLOR: #fff") == {"background-color": "#fff"}
    assert parse_style("") == {}


def test_replace_inline_blocks():
    # std_commands color swatch
    swatch = ('&nbsp;<div style="width:1em; height:.6em; display:inline-block;'
              ' border:1px solid #000; background-color:#ff0000"></div>')
    seen = []

    def replace(props):
        seen.append(props)
        return "[swatch]"
    assert replace_inline_blocks("hex #ff0000" + swatch + "<br>", replace) == "hex #ff0000&nbsp;[swatch]<br>"
    assert seen[0]["background-color"] == "#ff0000"
    assert seen[0]["border"] == "1px solid #000"
    # ordinary and non-empty divs are left alone
    for html in ('<div style="color:red"></div>', '<div style="display:inline-block">x</div>'):
        assert replace_inline_blocks(html, replace) == html


def test_css_lengths():
    assert css_length("10px", em=16) == 10
    assert css_length(".5em", em=16) == 8
    assert css_length("12", em=16) == 12
    assert css_length("auto", em=16) is None
    assert css_length(None, em=16) is None
    assert css_border("1px solid #000") == (1, "#000")
    assert css_border("solid red") == (3, "red")
    assert css_border("none") == (0, None)


def test_drop_redundant_breaks():
    assert drop_redundant_breaks("old line<br>\n<p>para</p>") == "old line<p>para</p>"
    assert drop_redundant_breaks("<td>a<br>b<br></td>") == "<td>a<br>b</td>"
    assert drop_redundant_breaks("a<br><br>\n<table>") == "a<br><table>"
    assert drop_redundant_breaks("a<br>b<br>") == "a<br>b<br>"
    assert drop_redundant_breaks("a<br><b>bold</b>") == "a<br><b>bold</b>"
    assert drop_redundant_breaks("a<br><pref>") == "a<br><pref>"
