# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.core.atomic import AtomicStructure
from chimerax.core.tools import ToolInstance


class ViewDockTool(ToolInstance):

    SESSION_ENDURING = False
    SESSION_SKIP = True         # No session saving for now
    CUSTOM_SCHEME = "viewdockx"    # HTML scheme for custom links
    display_name = "ViewDockX"

    def __init__(self, session, tool_name, structures=None):
        # Standard template stuff for intializing tool
        super().__init__(session, tool_name)
        from chimerax.core.ui.gui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        self.tool_window.manage(placement="side")
        if structures is None:
            structures = session.models.list(type=AtomicStructure)
        self.structures = structures
        parent = self.tool_window.ui_area

        # Create an HTML viewer for our user interface.
        # We can include other Qt widgets if we want to.
        from PyQt5.QtWidgets import QGridLayout
        from chimerax.core.ui.widgets import HtmlView
        layout = QGridLayout()
        self.html_view = HtmlView(parent, size_hint=(575, 200),
                                  interceptor=self._navigate,
                                  schemes=[self.CUSTOM_SCHEME])
        layout.addWidget(self.html_view, 0, 0)  # row 0, column 0
        parent.setLayout(layout)

        # Register for model addition/removal so we can update model list
        from chimerax.core.models import REMOVE_MODELS
        t = session.triggers
        # self._add_handler = t.add_handler(ADD_MODELS, self._update_models)
        self._remove_handler = t.add_handler(
            REMOVE_MODELS, self._update_models)

        # Go!
        self._update_models()

    def delete(self):
        t = self.session.triggers
        if self._remove_handler:
            t.remove_handler(self._remove_handler)
            self._remove_handler = None
        super().delete()

    def _update_models(self, trigger=None, trigger_data=None):
        """ Called to update page with current list of models"""
        from urllib.parse import urlunparse, urlencode
        if trigger_data is not None:

            for struct in self.structures:
                if struct in trigger_data:
                    self.structures.remove(struct)
            if not self.structures:
                self.delete()
                return

        # TRANSFERS ALL KEYS INTO A SET, THEN A LIST
        category_set = set()
        for struct in self.structures:
            try:
                category_set.update({key for key in struct.viewdock_comment})
            except AttributeError:
                pass
        category_list = sorted(list(category_set), key=str.lower)

        ####################
        ####    TABLE   ####
        ####################

        table = []
        table.append(
            '<table id="viewdockx_table" class="tablesorter" style="width:100%">')

        ###########################
        ###    COLUMN HEADERS   ###
        ###########################

        #   COLUMN HEADER    | ID |
        table.append('<thead><tr>')
        table.append('<th class="id">ID</th>')

        #   COLUMN HEADERS    | NAME |...|...|...
        table.append('<th>NAME</th>')
        for category in category_list:
            if category.upper() == "NAME":
                pass
            else:
                table.append('<th>{}</th>'.format(category.upper()))
        table.append("</tr></thead>")

        ########################
        ###    COLUMN DATA   ###
        ########################
        table.append('<tbody>')
        for struct in self.structures:
            try:
                comment_dict = struct.viewdock_comment
            except AttributeError:  # for files with empty comment sections
                comment_dict = {}

            # MAKES THE URL FOR EACH STRUCTURE
            args = [("atomspec", struct.atomspec())]
            query = urlencode(args)

            #url = urlunparse((self.CUSTOM_SCHEME, "", "", "", query, ""))
            checkbox_url = urlunparse((self.CUSTOM_SCHEME, "", "checkbox", "", query, ""))
            link_url = urlunparse((self.CUSTOM_SCHEME, "", "link", "", query, ""))

            # ADDING ID VALUE
            table.append("<tr>")
            table.extend(['<td class="id">',
                          # for checkbox + atomspec string
                          '<span class="checkbox">'
                          '<input class="checkbox, struct" type="checkbox" href="{}"/>'

                          '{}</span>'.format(checkbox_url, struct.atomspec()[1:]),


                          # for atomspec links only
                          '<span class="link"><a href="{}">{}</a></span>'
                          .format(link_url, struct.atomspec()[1:]),
                          '</td>'])

            # ADDING VALUE FOR NAME
            for category in category_list:
                if category.upper() == "NAME":
                    try:
                        table.append(
                            '<td>{}</td>'.format(comment_dict[category]))
                    except KeyError:
                        table.append('<td>missing</td>')

            # ADDING THE REST
            for category in category_list:
                try:
                    if category.upper() != "NAME":
                        table.append('<td>{}</td>'
                                     .format(comment_dict[category]))
                except KeyError:
                    table.append('<td>missing</td>')
            table.append("</tr>")
        table.append("</tbody>")
        table.append("</table>")



        import os
        from PyQt5.QtCore import QUrl

        # os.path.join()

        dir_path = os.path.dirname(os.path.abspath(__file__))
        # lib_path = os.path.join(dir_path, "lib")

        qurl = QUrl.fromLocalFile(os.path.join(dir_path, "viewdockx.html"))

        with open(os.path.join(dir_path, "viewdockx_frame.html"), "r") as file:
            template = file.read()
        output = template.replace("TABLE", ('\n'.join(table)))\
                         .replace("URLBASE", qurl.url())
        with open("viewdock.html", "w") as f:
            print(output, file=f)
        self.html_view.setHtml(output, qurl)


        # output_file = os.path.join(
        #     "C:/Users/hannahku/Desktop/RBVIInternship/GitHub/UCSF-RBVI-Internship/ViewDockX/src/output-test.html")
        # print(output_file)
        # with open(output_file, "w") as file2:
        #     file2.write(output)
        # print("TEST SUCCESS")

    def _navigate(self, info):
        # Called when link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        url = info.requestUrl()
        scheme = url.scheme()

        if scheme == self.CUSTOM_SCHEME:
            # Intercept our custom scheme.
            # Method may be invoked in a different thread than
            # the main thread where Qt calls may be made.
            query = parse_qs(url.query())
            path = url.path()

            function_map = {
                "check_all": self.check_all,
                "checkbox": self.checkbox,
                "link": self.link,
                "graph": self.graph
            }

            function_map[path](query)
            # self.session.ui.thread_safe(function_map[path](query))

    def check_all(self, query):
        """shows or hides all structures"""
        show_all = query["show_all"][0]
        self.session.ui.thread_safe(self._run_checkall, show_all)


    def checkbox(self, query):
        """shows or hides individual structure"""
        from chimerax.core.commands.cli import StructuresArg
        print(query)
        try:
            atomspec = query["atomspec"][0]
            disp = query["display"][0]
        except (KeyError, ValueError):
            atomspec = "missing"
        structures = StructuresArg.parse(atomspec, self.session)[0]

        self.session.ui.thread_safe(self._run_cb, structures, disp)

    def link(self, query):
        """shows only selected structure"""
        from chimerax.core.commands.cli import StructuresArg
        try:
            atomspec = query["atomspec"][0]
        except (KeyError, ValueError):
            atomspec = "missing"
        structures = StructuresArg.parse(atomspec, self.session)[0]

        self.session.ui.thread_safe(self._run_link, structures)

    def graph(self, query):
        """open new window to render graph"""
        pass

    def _run_cb(self, structures, disp):
        if disp == "0":
            for struct in self.structures:
                if structures[0] == struct:
                    struct.display = False
        else:
            for struct in self.structures:
                if structures[0] == struct:
                    struct.display = True

    def _run_link(self, structures):
        for struct in self.structures:
            struct.display = struct in structures

    def _run_checkall(self, show_all):
        if show_all == "true":
            for struct in self.structures:
                struct.display = True

        else:
            for struct in self.structures:
                struct.display = False

        # run(self.session, "select " + text)
        # from chimerax.core.logger import StringPlainTextLog
        # with StringPlainTextLog(self.session.logger) as log:
        #     try:
        #     finally:
        #         html = "<pre>\n%s</pre>" % log.getvalue()
        #         js = ('document.getElementById("output").innerHTML = %s'
        #               % repr(html))
        #         self.html_view.page().runJavaScript(js)
