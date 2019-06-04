# vim: set expandtab shiftwidth=4 softtabstop=4:
from io import StringIO
from chimerax.ui import HtmlToolInstance


class _BaseTool(HtmlToolInstance):

    SESSION_ENDURING = False
    SESSION_SAVE = True

    help = "help:user/tools/viewdockx.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(575,400), log_errors=True)
        self.structures = []
        self._html_state = None
        self._loaded_page = False

    def setup(self, structures=None, html_state=None):
        self._html_state = html_state
        try:
            self._setup(structures)
        except ValueError as e:
            self.delete()
            raise

    def _setup(self, structures):
        #
        # Set attributes that may be examined during delete.
        # Must be done before raising exceptions.
        #
        self._remove_handler = None
        self._display_handler = None
        self.category_name = None
        self.category_rating = "viewdockx_rating"
        self.category_list = []
        self.structures = []

        #
        # Get list of structures that we are displaying
        #
        session = self.session
        if structures is None:
            # Include structures only if they have viewdock data
            from chimerax.atomic import AtomicStructure
            structures = [s for s in session.models.list(type=AtomicStructure)
                          if hasattr(s, "viewdockx_data") and s.viewdockx_data]
        else:
            structures = [s for s in structures
                          if hasattr(s, "viewdockx_data") and s.viewdockx_data]

        if not structures:
            from chimerax.core.errors import UserError
            raise UserError("No suitable models found for ViewDockX")
        self.structures = structures
        t = session.triggers
        from chimerax.core.models import REMOVE_MODELS, MODEL_DISPLAY_CHANGED
        self._remove_handler = t.add_handler(REMOVE_MODELS, self._update_models)
        self._display_handler = t.add_handler(MODEL_DISPLAY_CHANGED,
                                              self._update_display)

        #
        # Make sure every structure has a rating
        #
        for s in structures:
            if self.category_rating not in s.viewdockx_data:
                s.viewdockx_data[self.category_rating] = "3"
            else:
                try:
                    r = int(s.viewdockx_data[self.category_rating])
                    if r < 0 or r > 5:
                        raise ValueError("out of range")
                except ValueError:
                    s.viewdockx_data[self.category_rating] = "3"

        #
        # Get union of categories found in all viewdockx_data attributes
        #
        category_set = set()
        for s in self.structures:
            try:
                category_set.update([key for key in s.viewdockx_data])
            except AttributeError:
                pass
        # "name" and "rating" categories are special cases that we separate out
        for category in category_set:
            if category.lower() == "name":
                self.category_name = category
                category_set.remove(category)
                break
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
            self.structures = [s for s in self.structures
                               if s not in trigger_data]
        if not self.structures:
            self.delete()
            return
        import json
        columns = json.dumps(self._make_columns())
        js = "%s.update_columns(%s);" % (self.CUSTOM_SCHEME, columns)
        self.html_view.runJavaScript(js)

    def _make_columns(self):
        # Construct separate dictionaries for numeric and text data
        numeric_data = {}
        text_data = {}
        # First make the id and name columns
        id_list = []
        name_list = []
        name_attr = self.category_name
        for s in self.structures:
            id_list.append(s.id_string)
            if name_attr:
                name_list.append(s.viewdockx_data.get(name_attr, "unnamed"))
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
                datum = s.viewdockx_data.get(category, None)
                if datum is None:
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
        return [(s.atomspec[1:], True if s.display else False)
                for s in structures]

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
            self._loaded_page = True
            self._update_models()
            self._set_html_state()
            self.html_view.loadFinished.disconnect(self._load_finished)

    def get_structures(self, model_id):
        if model_id:
            from chimerax.atomic import StructuresArg
            atomspec = ''.join(['#' + mid for mid in model_id.split(',')])
            return StructuresArg.parse(atomspec, self.session)[0]
        else:
            return self.structures

    def show_only(self, model_id):
        structures = self.get_structures(model_id)
        for s in self.structures:
            onoff = s in structures
            if s.display != onoff:
                s.display = onoff

    def show_toggle(self, model_id):
        structures = self.get_structures(model_id)
        for s in structures:
            if s in self.structures:
                s.display = not s.display

    def show_set(self, model_id, onoff):
        structures = self.get_structures(model_id)
        for s in structures:
            if s.display != onoff and s in self.structures:
                s.display = onoff

    # Session stuff

    html_state = "_html_state"

    def take_snapshot(self, session, flags):
        data = {
            "version": 2,
            "_super": super().take_snapshot(session, flags),
            "structures": self.structures,
        }
        self.add_webview_state(data)
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data["_super"])
        structures = data["structures"]
        if data.get("version", 1) == 1:
            classes = set()
            for s, vdx_data in structures:
                s.viewdockx_data = vdx_data
                classes.add(s.__class__)
            structures = list([sd[0] for sd in structures])
            for c in classes:
                c.register_attr(session, "viewdockx_data", "ViewDockX")
        inst.setup(structures, data.get(cls.html_state, None))
        return inst

    def add_webview_state(self, data):
        # Add webview state to data dictionary, synchronously.
        #
        # You have got to be kidding me - Johnny Mac
        # JavaScript callbacks are executed asynchronously,
        # and it looks like (in Qt 5.9) it is handled as
        # part of event processing.  So we cannot simply
        # use a semaphore and wait for the callback to
        # happen, since it will never happen because we
        # are not processing events.  So we use a busy
        # wait until the data we expect to get shows up.
        # Using a semaphore is overkill, since we can just
        # check for the presence of the key to be added,
        # but it does generalize if we want to call other
        # JS functions and get the value back synchronously.
        from PyQt5.QtCore import QEventLoop
        from threading import Semaphore
        event_loop = QEventLoop()
        js = "%s.get_state();" % self.CUSTOM_SCHEME
        def add(state):
            data[self.html_state] = state
            event_loop.quit()
        self.html_view.runJavaScript(js, add)
        while self.html_state not in data:
            event_loop.exec_()

    def _set_html_state(self):
        if self._html_state:
            import json
            js = "%s.set_state(%s);" % (self.CUSTOM_SCHEME,
                                        json.dumps(self._html_state))
            self.html_view.runJavaScript(js)
            self._html_state = None


class TableTool(_BaseTool):

    CUSTOM_SCHEME = "vdxtable"

    def __init__(self, session, tool_name, structures=None, html_state=None):
        self.display_name = "ViewDockX Table"
        super().__init__(session, tool_name)
        self.setup_page("viewdockx_table.html")

    def _update_ratings(self, trigger=None, trigger_data=None):
        if trigger_data is None:
            trigger_data = self.structures
        ratings = [(s.atomspec[1:], s.viewdockx_data[self.category_rating])
                   for s in trigger_data]
        import json
        js = "%s.update_ratings(%s);" % (self.CUSTOM_SCHEME,
                                         json.dumps(ratings))
        self.html_view.runJavaScript(js)

    def handle_scheme(self, url):
        # Called when custom link is clicked.
        # "info" is an instance of QWebEngineUrlRequestInfo
        from urllib.parse import parse_qs
        method = getattr(self, "_cb_" + url.path())
        query = parse_qs(url.query())
        method(query)

    def _cb_show_all(self, query):
        """shows or hides all structures"""
        self.show_set(None, True)

    def _cb_show_only(self, query):
        """shows only selected structure"""
        try:
            models = query["id"][0]
        except KeyError:
            self.show_set(None, False)
        else:
            self.show_only(models)

    def _cb_rating(self, query):
        """update rating for structure"""
        # May need to fire trigger for notification later
        try:
            model_id = query["id"][0]
            rating = int(query["rating"][0])
        except (KeyError, ValueError):
            return
        structures = self.get_structures(model_id)
        any_change = False
        for s in structures:
            v = str(rating)
            if s.viewdockx_data[self.category_rating] != v:
                s.viewdockx_data[self.category_rating] = v
                any_change = True
        if any_change:
            self._update_ratings(trigger_data=structures)

    def _cb_graph(self, query):
        tool = ChartTool(self.session, "ViewDockX Graph")
        tool.setup(self.structures)

    def _cb_plot(self, query):
        tool = PlotTool(self.session, "ViewDockX Plot")
        tool.setup(self.structures)

    def _cb_hb(self, query):
        # Create hydrogen bonds between receptor(s) and ligands
        from chimerax.core.commands import concise_model_spec, run
        cmd = "hbonds %s restrict cross reveal true" % concise_model_spec(
                                                            self.session,
                                                            self.structures)
        run(self.session, cmd)
        self._count_pb("hydrogen bonds", "HBonds")

    def _count_pb(self, group_name, key):
        # Count up the hydrogen bonds for each structure
        pbg = self.session.pb_manager.get_group(group_name)
        pa1, pa2 = pbg.pseudobonds.atoms
        for s in self.structures:
            atoms = s.atoms
            ma1 = pa1.mask(atoms)
            ma2 = pa2.mask(atoms)
            s.viewdockx_data[key] = (ma1 ^ ma2).sum()
        # Make sure HBonds is in our list of columns
        if key not in self.category_list:
            self.category_list.append(key)
            self.category_list.sort(key=str.lower)
        self._update_models()

    def _cb_clash(self, query):
        # Compute clashes between receptor(s) and ligands
        from chimerax.core.commands import concise_model_spec, run
        cmd = "clashes %s test others reveal true" % concise_model_spec(
                                                            self.session,
                                                            self.structures)
        run(self.session, cmd)
        self._count_pb("clashes", "Clashes")

    def _cb_export(self, query):
        from chimerax.ui.open_save import SaveDialog
        sd = SaveDialog(add_extension="mol2")
        if not sd.exec():
            return
        path = sd.get_path()
        if path is None:
            return
        prefix = "##########"
        from chimerax.mol2.io import write_mol2
        with open(path, "w") as outf:
            for s in self.structures:
                with OutputCache() as sf:
                    write_mol2(self.session, sf, models=[s])
                    for item in s.viewdockx_data.items():
                        print(prefix, "%s: %s\n" % item, end='', file=outf)
                    print("\n", end='', file=outf)
                    print(sf.saved_output, end='', file=outf)
                    print("\n\n", end='', file=outf)

    def _cb_prune(self, query):
        stars = int(query["stars"][0])
        structures = [s for s in self.structures
                      if int(s.viewdockx_data[self.category_rating]) <= stars]
        if not structures:
            print("No structures closed")
            return
        self.session.models.close(structures)

    def _cb_columns_updated(self, query):
        self._update_display()
        self._update_ratings()


class OutputCache(StringIO):

    def close(self, *args, **kw):
        if not self.closed:
            self.saved_output = self.getvalue()
        super().close(*args, **kw)


class ChartTool(_BaseTool):

    CUSTOM_SCHEME = "vdxchart"

    help = "help:user/tools/viewdockx.html#plots"

    def __init__(self, session, tool_name, structures=None, html_state=None):
        self.display_name = "ViewDockX Chart"
        super().__init__(session, tool_name)
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


class PlotTool(_BaseTool):

    CUSTOM_SCHEME = "vdxplot"

    help = "help:user/tools/viewdockx.html#plots"

    def __init__(self, session, tool_name, structures=None, html_state=None):
        self.display_name = "ViewDockX Plot"
        super().__init__(session, tool_name)
        self.setup_page("viewdockx_plot.html")

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
