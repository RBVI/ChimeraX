# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Sphinx extension class for munging output for autodoc subclass links.

Only output the last component of the class name instead of the full path.
"""

from docutils.nodes import GenericNodeVisitor

Replacements = [
    (".commands.cli", ".commands"),
    (".commands.atomspec", ".commands"),
]


def setup(app):
    munge_autodoc()
    app.connect("doctree-read", doctree_read)
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    return {"version":"0.1"}


def munge_autodoc():
    # Munge autodoc ClassDocumenter so that base classes do
    # not have full path
    from sphinx.ext import autodoc
    # Code below is copied from Sphinx 1.6.8, with the '~' inserted
    # for non-builtin classes
    def my_add_directive_header(self, sig):
        if self.doc_as_attr:
            self.directivetype = 'attribute'
        autodoc.Documenter.add_directive_header(self, sig)

        if not self.doc_as_attr and self.options.show_inheritance:
            sourcename = self.get_sourcename()
            self.add_line(u'', sourcename)
            if hasattr(self.object, '__bases__') and len(self.object.__bases__):
                bases = [b.__module__ in ('__builtin__', 'builtins') and
                         u':class:`%s`' % b.__name__ or
                         u':class:`~%s.%s`' % (b.__module__, b.__name__)
                         for b in self.object.__bases__]
                self.add_line(u'   ' + autodoc._(u'Bases: %s') % ', '.join(bases),
                              sourcename)
    autodoc.ClassDocumenter.add_directive_header = my_add_directive_header


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


def autodoc_process_docstring(app, what, name, obj, options, lines):
    api_status = "unknown"
    while lines:
        if not lines[0].strip():
            lines.pop(0)
        else:
            break
    if what in ["method"]:
        for n, line in enumerate(lines):
            if "Supported API" in line:
                lines[n] = line.replace("Supported API", "*Supported API*")
                api_status = "supported"
            elif "Experimental API" in line:
                api_status = "experimental"
            elif "Private API" in line:
                api_status = "private"
        if api_status == "unknown":
            lines.insert(0, "*Experimental API*.")


def autodoc_skip_member(app, what, name, obj, skip, options):
    if skip:
        return None
    try:
        doc = obj.__doc__
    except AttributeError:
        return None
    else:
        if doc and "Private API" in doc:
            return True
        return None
