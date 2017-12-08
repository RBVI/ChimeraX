# vim: set expandtab shiftwidth=4 softtabstop=4:
from chimerax.core.ui import HtmlToolInstance


class _BaseTool:

    def setup(self, session, structures):
        #
        # Set attributes that may be examined during delete.
        # Must be done before raising exceptions.
        #
        self._remove_handler = None
        self._display_handler = None
        self.category_name = None
        self.category_list = []
        self.structures = []

        #
        # Get list of structures that we are displaying
        #
        if structures is None:
            # Include structures only if they have viewdock data
            from chimerax.core.atomic import AtomicStructure
            structures = [s for s in session.models.list(type=AtomicStructure)
                          if hasattr(s, "viewdock_comment")]
        else:
            structures = [s for s in structures
                          if hasattr(s, "viewdock_comment")]

        if not structures:
            raise ValueError("No suitable models found for ViewDockX")
        self.structures = structures
        t = session.triggers
        from chimerax.core.models import REMOVE_MODELS, MODEL_DISPLAY_CHANGED
        self._remove_handler = t.add_handler(REMOVE_MODELS, self._update_models)
        self._display_handler = t.add_handler(MODEL_DISPLAY_CHANGED,
                                              self._update_display)

        #
        # Get union of categories found in all viewdock_comment attributes
        #
        category_set = set()
        for s in self.structures:
            try:
                category_set.update([key for key in s.viewdock_comment])
            except AttributeError:
                pass
        # "name" category is a special case that we separate out
        for category in category_set:
            if category.lower() == "name":
                self.category_name = category
                category_set.remove(category)
                break
        else:
            self.category_name = None
        self.category_list = sorted(list(category_set), key=str.lower)

    def _update_models(self, trigger=None, trigger_data=None):
        """ Called to update page with current list of models"""
        if trigger_data is not None:
            for s in self.structures:
                if s in trigger_data:
                    self.structures.remove(s)

    def _update_display(self, trigger, trigger_data):
        pass

    def delete_handlers(self):
        t = self.session.triggers
        if self._remove_handler:
            t.remove_handler(self._remove_handler)
            self._remove_handler = None
        if self._display_handler:
            t.remove_handler(self._display_handler)
            self._display_handler = None

    def make_data_arrays(self):
        # Construct separate dictionaries for numeric and text data
        numeric_data = {}
        text_data = {}
        # First make the id and name columns
        id_list = []
        name_list = []
        name_attr = self.category_name
        for s in self.structures:
            id_list.append(s.id_string())
            if name_attr:
                name_list.append(s.viewdock_comment.get(name_attr, "unnamed"))
        text_data["Id"] = id_list
        if name_attr:
            text_data[name_attr] = name_list
        # Now make numeric and text versions for each category
        # If there are more numbers than text, then assume numeric
        for category in self.category_list:
            numeric_list = []
            text_list = []
            num_numeric = 0
            num_text = 0
            for s in self.structures:
                datum = s.viewdock_comment.get(category, None)
                if not datum:
                    numeric_list.append(None)
                else:
                    try:
                        numeric_list.append(int(datum))
                        num_numeric += 1
                    except ValueError:
                        try:
                            numeric_list.append(float(datum))
                            num_numeric += 1
                        except ValueError:
                            numeric_list.append(None)
                            num_text += 1
                text_list.append(datum)
            if num_numeric > num_text:
                numeric_data[category] = numeric_list
            else:
                text_data[category] = text_list
        return numeric_data, text_data

    def show_only(self, atomspec):
        from chimerax.core.commands.cli import StructuresArg
        structures = StructuresArg.parse(atomspec, self.session)[0]
        for s in self.structures:
            s.display = s in structures

    def show_toggle(self, atomspec):
        from chimerax.core.commands.cli import StructuresArg
        structures = StructuresArg.parse(atomspec, self.session)[0]
        for s in structures:
            if s in self.structures:
                s.display = not s.display

    def show_set(self, atomspec, onoff):
        from chimerax.core.commands.cli import StructuresArg
        structures = StructuresArg.parse(atomspec, self.session)[0]
        for s in structures:
            if s in self.structures:
                s.display = onoff


class ViewDockTool(HtmlToolInstance, _BaseTool):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "viewdockx"

    def __init__(self, session, tool_name, structures=None):
        self.display_name = "ViewDockX"
        super().__init__(session, tool_name, size_hint=(575,200))
        try:
            self.setup(session, structures)
        except ValueError as e:
            session.logger.error(str(e))
            self.delete()
            return
        from chimerax.core.models import REMOVE_MODELS
        self._remove_handler = session.triggers.add_handler(REMOVE_MODELS,
                                                            self._update_models)
        self._update_models()
        self.html_view.loadFinished.connect(self._load_finished)

    def _load_finished(self, success):
        # First time through, we need to wait for the page to load
        # before trying to update data.  Afterwards, we don't care.
        if success:
            self._update_display()
            self.html_view.loadFinished.disconnect(self._load_finished)

    def delete(self):
        self.delete_handlers()
        super().delete()

    def _update_models(self, trigger=None, trigger_data=None):
        """ Called to update page with current list of models"""
        super()._update_models(trigger, trigger_data)
        if not self.structures:
            self.delete()
            return

        # Table container
        table = []
        table.append('<table id="viewdockx_table" class="tablesorter" '
                     'style="width:100%">')

        # Table column headers: checkbox model_id [name] all_other_columns
        table.append('<thead><tr>')
        table.append('<th/>')
        table.append('<th class="id">ID</th>')
        if self.category_name:
            table.append('<th>NAME</th>')
        for category in self.category_list:
            table.append('<th>{}</th>'.format(category.upper()))
        table.append("</tr></thead>")

        # Table cell data
        from urllib.parse import urlunparse, urlencode
        table.append('<tbody>')
        for s in self.structures:
            try:
                comment_dict = s.viewdock_comment
            except AttributeError:  # for files with empty comment sections
                comment_dict = {}

            # MAKES THE URL FOR EACH STRUCTURE
            atomspec = s.atomspec()
            query = urlencode([("atomspec", atomspec)])

            checkbox_url = urlunparse((self.CUSTOM_SCHEME, "",
                                       "checkbox", "", query, ""))
            link_url = urlunparse((self.CUSTOM_SCHEME, "",
                                   "link", "", query, ""))

            table.append("<tr>")
            # First column is always checkbox
            table.extend(['<td>',
                          '<input class="structure" '
                          'type="checkbox" id="cb_{}" href="{}"/>'
                          .format(atomspec[1:], checkbox_url),
                          '</td>'])

            # Second column is always model id
            table.extend(['<td class="id">',
                          '<a href="{}">{}</a>'
                          .format(link_url, atomspec[1:]),
                          '</td>'])

            # If there is a name column, it is next
            if self.category_name:
                v = comment_dict.get(self.category_name, "-")
                table.append('<td>{}</td>'.format(v))

            # All other columns in alphabetical order
            for category in self.category_list:
                v = comment_dict.get(category, "-")
                table.append('<td>{}</td>'.format(v))
            table.append("</tr>")
        table.append("</tbody>")
        table.append("</table>")

        import os.path
        template_path = os.path.join(os.path.dirname(__file__),
                                     "viewdockx_table.html")
        with open(template_path, "r") as f:
            template = f.read()
        # template path also serves as the <base> tag value for relative links
        from PyQt5.QtCore import QUrl
        qurl = QUrl.fromLocalFile(template_path)
        output = template.replace("TABLE", ('\n'.join(table)))\
                         .replace("URLBASE", qurl.url())
        self.html_view.setHtml(output, qurl)
        # Debug
        #with open("vtable.html", "w") as f:
        #    print(output, file=f)

    def _update_display(self, trigger=None, trigger_data=None):
        if trigger_data is not None:
            structures = [trigger_data]
        else:
            structures = self.structures
        import json
        onoff = [(s.atomspec()[1:], s.display) for s in structures]
        js = "update_display(%s);" % json.dumps(onoff)
        self.html_view.runJavaScript(js)

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_check_all(self, query):
        """shows or hides all structures"""
        self.show_set("#", query["show_all"][0] == True)

    def _cb_checkbox(self, query):
        """shows or hides individual structure"""
        self.show_set(query["atomspec"][0], query["display"][0] != "0")

    def _cb_link(self, query):
        """shows only selected structure"""
        self.show_only(query["atomspec"][0])

    def _cb_graph(self, query):
        ChartTool(self.session, "ViewDock Chart", structures=self.structures)

    def _cb_histogram(self, query):
        pass


class ChartTool(HtmlToolInstance, _BaseTool):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "viewdockx"

    def __init__(self, session, tool_name, structures=None):
        self.display_name = "ViewDockX"
        super().__init__(session, tool_name, size_hint=(575,400))
        try:
            self.setup(session, structures)
        except ValueError as e:
            session.logger.error(str(e))
            self.delete()
            return
        self._setup_page()

    def _setup_page(self):
        import os.path
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(os.path.dirname(__file__),
                                     "viewdockx_chart.html")
        with open(template_path, "r") as f:
            template = f.read()
        from PyQt5.QtCore import QUrl
        qurl = QUrl.fromLocalFile(template_path)
        output = template.replace("URLBASE", qurl.url())
        self.html_view.setHtml(output, qurl)
        self.html_view.loadFinished.connect(self._load_finished)
        # Debug
        #with open("vchart.html", "w") as f:
        #    print(output, file=f)

    def _load_finished(self, success):
        # First time through, we need to wait for the page to load
        # before trying to update data.  Afterwards, we don't care.
        if success:
            self._update_models()
            self.html_view.loadFinished.disconnect(self._load_finished)

    def _update_models(self, trigger=None, trigger_data=None):
        super()._update_models(trigger, trigger_data)
        if not self.structures:
            self.delete()
            return
        import json
        js = "update_columns(%s);" % json.dumps(self.make_data_arrays())
        self.html_view.runJavaScript(js)

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_show_only(self, query):
        """shows or hides all structures"""
        self.show_only("#" + query["id"][0])

    def _cb_show_toggle(self, query):
        """shows or hides all structures"""
        self.show_toggle("#" + query["id"][0])
