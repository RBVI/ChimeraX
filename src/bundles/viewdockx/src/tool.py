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

    def delete(self):
        t = self.session.triggers
        if self._remove_handler:
            t.remove_handler(self._remove_handler)
            self._remove_handler = None
        if self._display_handler:
            t.remove_handler(self._display_handler)
            self._display_handler = None
        super().delete()

    def _update_models(self, trigger=None, trigger_data=None):
        """ Called to update page with current list of models"""
        if trigger_data is not None:
            for s in self.structures:
                if s in trigger_data:
                    self.structures.remove(s)
        if not self.structures:
            self.delete()
            return
        import json
        columns = json.dumps(self._make_columns())
        js = "%s.update_columns(%s);" % (self.CUSTOM_SCHEME, columns)
        self.html_view.runJavaScript(js)
        self._update_display()

    def _make_columns(self):
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
        text_data["id"] = id_list
        if name_attr:
            text_data["name"] = name_list
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
        return { "numeric": numeric_data, "text": text_data }

    def _update_display(self, trigger=None, trigger_data=None):
        import json
        onoff = json.dumps(self._make_display(trigger_data))
        js = "%s.update_display(%s);" % (self.CUSTOM_SCHEME, onoff)
        self.html_view.runJavaScript(js)

    def _make_display(self, s=None):
        if s:
            structures = [s]
        else:
            structures = self.structures
        return [(s.atomspec()[1:], s.display) for s in structures]

    def setup_page(self, html_file):
        import os.path
        dir_path = os.path.dirname(__file__)
        template_path = os.path.join(os.path.dirname(__file__), html_file)
        with open(template_path, "r") as f:
            template = f.read()
        from PyQt5.QtCore import QUrl
        qurl = QUrl.fromLocalFile(template_path)
        output = template.replace("URLBASE", qurl.url())
        self.html_view.setHtml(output, qurl)
        self.html_view.loadFinished.connect(self._load_finished)

    def _load_finished(self, success):
        # First time through, we need to wait for the page to load
        # before trying to update data.  Afterwards, we don't care.
        if success:
            self._update_models()
            self.html_view.loadFinished.disconnect(self._load_finished)

    def _get_structures(self, model_id):
        if model_id:
            from chimerax.core.commands.cli import StructuresArg
            return StructuresArg.parse('#' + model_id, self.session)[0]
        else:
            return self.structures

    def show_only(self, model_id):
        structures = self._get_structures(model_id)
        for s in self.structures:
            onoff = s in structures
            if s.display != onoff:
                s.display = onoff

    def show_toggle(self, model_id):
        structures = self._get_structures(model_id)
        for s in structures:
            if s in self.structures:
                s.display = not s.display

    def show_set(self, model_id, onoff):
        structures = self._get_structures(model_id)
        for s in structures:
            if s.display != onoff and s in self.structures:
                s.display = onoff


class TableTool(HtmlToolInstance, _BaseTool):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "vdxtable"

    def __init__(self, session, tool_name, structures=None):
        self.display_name = "ViewDockX Table"
        super().__init__(session, tool_name, size_hint=(575,200))
        try:
            self.setup(session, structures)
        except ValueError as e:
            session.logger.error(str(e))
            self.delete()
            return
        self.setup_page("viewdockx_table.html")

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_check_all(self, query):
        """shows or hides all structures"""
        self.show_set(None, query["show_all"][0] == "true")

    def _cb_checkbox(self, query):
        """shows or hides individual structure"""
        self.show_set(query["id"][0], query["display"][0] != "0")

    def _cb_link(self, query):
        """shows only selected structure"""
        self.show_only(query["id"][0])

    def _cb_chart(self, query):
        ChartTool(self.session, "ViewDockX Chart", structures=self.structures)

    def _cb_plot(self, query):
        pass

    def _cb_histogram(self, query):
        pass


class ChartTool(HtmlToolInstance, _BaseTool):

    SESSION_ENDURING = False
    SESSION_SAVE = False
    CUSTOM_SCHEME = "vdxchart"

    def __init__(self, session, tool_name, structures=None):
        self.display_name = "ViewDockX Chart"
        super().__init__(session, tool_name, size_hint=(575,400))
        try:
            self.setup(session, structures)
        except ValueError as e:
            session.logger.error(str(e))
            self.delete()
            return
        self.setup_page("viewdockx_chart.html")

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_show_only(self, query):
        """shows or hides all structures"""
        self.show_only(query["id"][0])

    def _cb_show_toggle(self, query):
        """shows or hides all structures"""
        self.show_toggle(query["id"][0])
