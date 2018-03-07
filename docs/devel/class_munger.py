"""Sphinx extension class for munging output for autodoc subclass links.

Only output the last component of the class name instead of the full path.
"""

from docutils.nodes import GenericNodeVisitor

Replacements = [
    (".commands.cli", ".commands"),
    (".commands.atomspec", ".commands"),
]


def setup(app):
    app.connect("doctree-read", doctree_read)
    return {"version":"0.1"}


def doctree_read(app, doctree):
    doctree.walk(DoctreeNodeVisitor(doctree))


class DoctreeNodeVisitor(GenericNodeVisitor):

    def visit_Text(self, node):
        text = orig_text = node.astext()
        for old, new in Replacements:
            text = text.replace(old, new)
        if text != orig_text:
            from docutils.nodes import Text
            node.parent.replace(node, Text(text, node.rawsource))

    def default_visit(self, node):
        return
