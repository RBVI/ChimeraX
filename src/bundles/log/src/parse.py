# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
Split log HTML into the pieces the native log view renders specially.

Qt's rich text engine does not support CSS 'display', <details>, or
reliably keep block formatting when HTML fragments are appended, so the
log view handles command echoes and disclosures itself and hands
everything else to Qt as ordinary HTML.
"""

import re
from dataclasses import dataclass

_opener_re = re.compile(r'<div class="cxcmd">|<details\b[^>]*>', re.IGNORECASE)
_div_re = re.compile(r'<div\b|</div\s*>', re.IGNORECASE)
_details_re = re.compile(r'<details\b|</details\s*>', re.IGNORECASE)
_summary_re = re.compile(r'\s*<summary\b[^>]*>(.*?)</summary\s*>', re.IGNORECASE | re.DOTALL)
_variant_re = re.compile(r'<div class="cxcmd_as_(doc|cmd)">(.*?)</div\s*>', re.IGNORECASE | re.DOTALL)
_bg_re = re.compile(r'background-color\s*:\s*([^;"\']+)', re.IGNORECASE)

_block_tags = r'(?:p|div|table|pre|ul|ol|dl|h[1-6]|blockquote|details|center)'
_starts_block_re = re.compile(r'\s*<(?:%s\b|hr\b)' % _block_tags, re.IGNORECASE)
_inline_block_re = re.compile(
    r'<div\s+style\s*=\s*(["\'])(?P<style>.*?)\1\s*>\s*</div\s*>', re.IGNORECASE | re.DOTALL)
_redundant_br_re = re.compile(
    r'<br\s*/?>\s*(?=<(?:%s|li|hr)\b|</(?:%s|td|th|li|dd|dt|summary)\s*>)' % (_block_tags, _block_tags),
    re.IGNORECASE)
_starts_table_re = re.compile(r'\s*<table\b', re.IGNORECASE)
_trailing_br_re = re.compile(r'\s*<br\s*/?>\s*$', re.IGNORECASE)
_ends_block_re = re.compile(r'(?:</%s\s*>|<hr\b[^>]*>)\s*$' % _block_tags, re.IGNORECASE)


@dataclass
class Html:
    """Ordinary HTML handed to Qt as-is."""
    html: str
    offset: int


@dataclass
class Command:
    """Contents of a <div class="cxcmd"> command echo."""
    html: str
    offset: int


@dataclass
class Disclosure:
    """A <details> element.  'offset' is where the <details> tag starts;
    'content_offset' is where 'content' starts."""
    open: bool
    background: str | None
    summary: str | None
    content: str
    offset: int
    content_offset: int


def _matching_close(html, start, tag_re):
    """Return (end of contents, end of closing tag) for the element whose
    opening tag ends at 'start', allowing for nested elements of the same type."""
    depth = 1
    for m in tag_re.finditer(html, start):
        if m.group(0)[1] == '/':
            depth -= 1
            if depth == 0:
                return m.start(), m.end()
        else:
            depth += 1
    return len(html), len(html)


def tokenize(html, base=0):
    """Split 'html' into Html, Command, and Disclosure pieces.

    Offsets are relative to the start of 'html' plus 'base', so that
    they can serve as stable identifiers within the whole log.
    """
    pieces = []
    pos = 0
    while True:
        m = _opener_re.search(html, pos)
        if m is None:
            break
        if m.start() > pos:
            pieces.append(Html(html[pos:m.start()], base + pos))
        tag = m.group(0)
        if tag.lower().startswith('<div'):
            content_end, end = _matching_close(html, m.end(), _div_re)
            pieces.append(Command(html[m.end():content_end], base + m.start()))
        else:
            content_end, end = _matching_close(html, m.end(), _details_re)
            content_start = m.end()
            summary = None
            sm = _summary_re.match(html, content_start, content_end)
            if sm:
                summary = sm.group(1)
                content_start = sm.end()
            bg = _bg_re.search(tag)
            pieces.append(Disclosure(
                open=re.search(r'\sopen\b', tag[len('<details'):], re.IGNORECASE) is not None,
                background=bg.group(1).strip() if bg else None,
                summary=summary,
                content=html[content_start:content_end],
                offset=base + m.start(),
                content_offset=base + content_start,
            ))
        pos = end
    if pos < len(html):
        pieces.append(Html(html[pos:], base + pos))
    return pieces


def select_command_variant(html, exec_links):
    """Keep only the cxcmd_as_cmd (exec_links True) or cxcmd_as_doc variant
    of a command echo, since Qt doesn't support 'display: none'."""
    want = 'cmd' if exec_links else 'doc'
    return _variant_re.sub(lambda m: m.group(2) if m.group(1).lower() == want else '', html)


def parse_style(style):
    """Convert a CSS declaration list into a dict of lowercase property names to values."""
    props = {}
    for decl in style.split(';'):
        name, sep, value = decl.partition(':')
        if sep:
            props[name.strip().lower()] = value.strip()
    return props


def replace_inline_blocks(html, replace):
    """Replace empty inline-block divs, such as color swatches, whose
    style dict is passed to 'replace' and which returns the replacement HTML.
    Qt treats all divs as blocks, so these would otherwise vanish."""
    if 'inline-block' not in html:
        return html

    def sub(m):
        props = parse_style(m.group('style'))
        if props.get('display', '').lower() != 'inline-block':
            return m.group(0)
        return replace(props)
    return _inline_block_re.sub(sub, html)


def css_length(value, em):
    """Convert a CSS length in px or em ('em' gives pixels per em) to pixels,
    or None if it can't be interpreted."""
    m = re.fullmatch(r'\s*([0-9]*\.?[0-9]+)\s*(px|em)?\s*', value or '')
    if m is None:
        return None
    size = float(m.group(1))
    return size * em if m.group(2) == 'em' else size


_border_styles = {'solid', 'dashed', 'dotted', 'double', 'groove', 'ridge', 'inset', 'outset'}


def css_border(value):
    """Return (width in px, color or None) from a CSS 'border' shorthand
    such as '1px solid #000'.  A color of None means the text color."""
    width, color = 3, None  # CSS 'medium' width
    for part in (value or '').split():
        p = part.lower()
        if p in ('none', 'hidden'):
            return 0, None
        if p in _border_styles:
            continue
        w = css_length(p, em=None) if not p.endswith('em') else None
        if w is not None:
            width = w
        else:
            color = part
    return width, color


def drop_redundant_breaks(html):
    """Remove <br>s that immediately precede the start or end of a block.
    Browsers don't show these as an empty line, but Qt does."""
    return _redundant_br_re.sub('', html)


def starts_block(html):
    """Whether 'html' starts with a block-level element."""
    return _starts_block_re.match(html) is not None


def starts_table(html):
    """Whether 'html' starts with a table, which Qt always puts in its own block."""
    return _starts_table_re.match(html) is not None


def strip_line_end(html):
    """Return (html, ends_line).  A trailing <br> is removed since the
    view starts a new block for following content instead."""
    m = _trailing_br_re.search(html)
    if m:
        return html[:m.start()], True
    return html, _ends_block_re.search(html) is not None
