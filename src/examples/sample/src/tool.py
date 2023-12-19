# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.ui import HtmlToolInstance
from chimerax.core.logger import PlainTextLog


class SampleTool(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False         # No session saving for now
    display_name = "Sample Tool"

    CUSTOM_SCHEME = "sample"    # HTML scheme for custom links

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(575, 200))
        self.update_models()

    def update_models(self, trigger=None, trigger_data=None):
        # Called to update page with current list of models
        from chimerax.atomic import AtomicStructure
        html = ["<h2>Sample Tool</h2>", "<ul>"]
        from urllib.parse import quote
        for m in self.session.models.list(type=AtomicStructure):
            html.append("<li><a href=\"%s:%s\">%s - %s</a></li>" %
                        (self.CUSTOM_SCHEME, quote(m.atomspec),
                         m.id_string, m.name))
        html.extend(["</ul>",
                     "<h3>Output:</h3>",
                     '<div id="output">Counts appear here</div>'])
        self.html_view.setHtml('\n'.join(html))

    def handle_scheme(self, url):
        # "url" is an instance of QUrl
        # Execute "sample count" command for given atomspec
        atomspec = url.path()
        from chimerax.core.commands import run
        from chimerax.core.logger import StringPlainTextLog
        with StringPlainTextLog(self.session.logger) as log:
            try:
                run(self.session, "sample count " + atomspec)
            finally:
                html = "<pre>\n%s</pre>" % log.getvalue()
                js = ('document.getElementById("output").innerHTML = %s'
                      % repr(html))
                self.html_view.page().runJavaScript(js)
