# vim: set expandtab shiftwidth=4 softtabstop=4:

"""Sphinx extension class for munging output for autodoc subclass links.

Only output the last component of the class name instead of the full path.
"""

from docutils.nodes import GenericNodeVisitor

Replacements = [
    # hide chimerax.core.commands submodules
    (r"\.commands\.cli\b", ".commands"),
    (r"\.commands\.atomspec\b", ".commands"),
    (r"\.commands\.run.run\b", ".commands.run"),
    (r"\.commands\.run.concise_model_spec\b", ".commands.concise_model_spec"),
    (r"\.commands\.run.quote_if_necessary\b", ".commands.quote_if_necessary"),
    (r"\.commands\.runscript.runscript\b", ".commands.runscript"),
    (r"\.commands\.runscript\b", ".commands"),
    (r"\.commands\.selectors\b", ".commands"),
    (r"\.commands\.symarg\b", ".commands"),

    # hide chimerax.ui submodules
    # TODO: what about mousemodes?
    (r"\.ui\.gui\b", ".ui"),
    (r"\.ui\.htmltool\b", ".ui"),
    (r"\.ui\.cmd\b", ".ui"),
    (r"\.ui\.font\b", ".ui"),
    (r"\.ui\.mousemodes\b", ".ui"),
    (r"\.ui\.widgets.htmlview\b", ".ui.widgets"),

    # hide chimerax.atomic submodules
    (r"\.atomic\.molobject\b", ".atomic"),
    (r"\.atomic\.pbgroup\b", ".atomic"),
    (r"\.atomic\.molarray\b", ".atomic"),
    (r"\.atomic\.structure\b", ".atomic"),
    (r"\.atomic\.molsurf\b", ".atomic"),
    (r"\.atomic\.changes\b", ".atomic"),
    (r"\.atomic\.pdbmatrices\b", ".atomic"),
    (r"\.atomic\.triggers\b", ".atomic"),
    (r"\.atomic\.mmcif\b", ".atomic"),
    (r"\.atomic\.pdb\b", ".atomic"),
    (r"\.atomic\.search\b", ".atomic"),
    (r"\.atomic\.shapedrawing\b", ".atomic"),
]


def setup(app):
    munge_autodoc()
    app.add_role("raw-html", raw_html_role)
    app.connect("doctree-read", doctree_read)
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    return {"version":"0.1"}


def munge_autodoc():
    # Munge autodoc ClassDocumenter so that base classes do
    # not have full path
    from sphinx.ext.autodoc import ClassDocumenter, Documenter, _
    # Code below is copied from Sphinx 1.6.8, with the '~' inserted
    # for non-builtin classes
    def my_add_directive_header(self, sig):
        if self.doc_as_attr:
            self.directivetype = 'attribute'
        Documenter.add_directive_header(self, sig)

        if not self.doc_as_attr and self.options.show_inheritance:
            sourcename = self.get_sourcename()
            self.add_line(u'', sourcename)
            if hasattr(self.object, '__bases__') and len(self.object.__bases__):
                bases = [b.__module__ in ('__builtin__', 'builtins') and
                         u':class:`%s`' % b.__name__ or
                         u':class:`~%s.%s`' % (b.__module__, b.__name__)
                         for b in self.object.__bases__]
                self.add_line(u'   ' + _(u'Bases: %s') % ', '.join(bases),
                              sourcename)
    ClassDocumenter.add_directive_header = my_add_directive_header


def doctree_read(app, doctree):
    doctree.walk(DoctreeNodeVisitor(doctree))


class DoctreeNodeVisitor(GenericNodeVisitor):

    def visit_Text(self, node):
        import re
        text = orig_text = node.astext()
        for old, new in Replacements:
            text = re.sub(old, new, text)
        # for old, new in Replacements:
        #     text = text.replace(old, new)
        if text != orig_text:
            rawsource = str(node).replace('\x00', '\\')
            from docutils.nodes import Text
            node.parent.replace(node, Text(text, rawsource))

    def default_visit(self, node):
        return


def autodoc_process_docstring(app, what, name, obj, options, lines):
    api_status = "unknown"
    while lines:
        if not lines[0].strip():
            lines.pop(0)
        else:
            break
    if what in ["method", "attribute", "function"]:
        for n, line in enumerate(lines):
            if "Supported API" in line:
                lines[n] = line.replace("Supported API",
                                        ":raw-html:`<i>` "
                                        ":ref:`supported-api` "
                                        ":raw-html:`</i>` ")
                api_status = "supported"
            elif "Experimental API" in line:
                lines[n] = line.replace("Experimental API",
                                        ":raw-html:`<i>` "
                                        ":ref:`experimental-api` "
                                        ":raw-html:`</i>` ")
                api_status = "experimental"
            elif "Private API" in line:
                api_status = "private"
        if api_status == "unknown":
            lines.insert(0, ":raw-html:`<i>` "
                            ":ref:`experimental-api` "
                            ":raw-html:`</i>`.")


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


def raw_html_role(*args, options={}, **kw):
    options["format"] = "html"
    from docutils.parsers.rst.roles import raw_role
    return raw_role(*args, options=options, **kw)
