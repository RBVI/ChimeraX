# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.core.ui import HtmlToolInstance


class ViewDockTool(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "viewdockx"

    def __init__(self, session, tool_name, structures=None):
        self.display_name = "ViewDockX"
        super().__init__(session, tool_name, size_hint=(575,200))
        if structures is None:
            from chimerax.core.atomic import AtomicStructure
            structures = session.models.list(type=AtomicStructure)
        self.structures = structures
        from chimerax.core.models import REMOVE_MODELS
        self._remove_handler = session.triggers.add_handler(REMOVE_MODELS,
                                                            self._update_models)
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
        table.append('<table id="viewdockx_table" class="tablesorter" '
                     'style="width:100%">')

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
            checkbox_url = urlunparse((self.CUSTOM_SCHEME, "",
                                       "checkbox", "", query, ""))
            link_url = urlunparse((self.CUSTOM_SCHEME, "",
                                   "link", "", query, ""))

            # ADDING ID VALUE
            table.append("<tr>")
            table.extend(['<td class="id">',
                          # for checkbox + atomspec string
                          '<span class="checkbox">'
                          '<input class="checkbox, struct" '
                          'type="checkbox" href="{}"/>'
                          '{}</span>'.format(checkbox_url,
                                             struct.atomspec()[1:]),
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
        dir_path = os.path.dirname(os.path.abspath(__file__))
        qurl = QUrl.fromLocalFile(os.path.join(dir_path, "viewdockx.html"))
        with open(os.path.join(dir_path, "viewdockx_frame.html"), "r") as file:
            template = file.read()
        output = template.replace("TABLE", ('\n'.join(table)))\
                         .replace("URLBASE", qurl.url())
        self.html_view.setHtml(output, qurl)
        # Debug
        with open("viewdock.html", "w") as f:
            print(output, file=f)

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_check_all(self, query):
        """shows or hides all structures"""
        show_all = query["show_all"][0]
        if show_all == "true":
            for struct in self.structures:
                struct.display = True
        else:
            for struct in self.structures:
                struct.display = False

    def _cb_checkbox(self, query):
        """shows or hides individual structure"""
        from chimerax.core.commands.cli import StructuresArg
        try:
            atomspec = query["atomspec"][0]
            disp = query["display"][0]
        except (KeyError, ValueError):
            atomspec = "missing"
        structures = StructuresArg.parse(atomspec, self.session)[0]
        if disp == "0":
            for struct in self.structures:
                if structures[0] == struct:
                    struct.display = False
        else:
            for struct in self.structures:
                if structures[0] == struct:
                    struct.display = True

    def _cb_link(self, query):
        """shows only selected structure"""
        from chimerax.core.commands.cli import StructuresArg
        try:
            atomspec = query["atomspec"][0]
        except (KeyError, ValueError):
            atomspec = "missing"
        structures = StructuresArg.parse(atomspec, self.session)[0]
        for struct in self.structures:
            struct.display = struct in structures

    def _cb_graph(self, query):
        pass

    def _cb_histogram(self, query):
        pass
