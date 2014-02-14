class Model_Panel:

    def __init__(self, session):

        self.session = session
        self.image_size = (128,128)

        from .qt import QtWidgets, QtCore
        self.dock_widget = dw = QtWidgets.QDockWidget('Scenes', session.main_window)
        dw.setTitleBarWidget(QtWidgets.QWidget(dw))   # No title bar
        dw.setFeatures(dw.NoDockWidgetFeatures)       # No close button

        class Thumbnail_Viewer(QtWidgets.QTextBrowser):
            height = 140
            def sizeHint(self):
                return QtCore.QSize(600,self.height)
        self.text = e = Thumbnail_Viewer(dw)
        e.setOpenLinks(False)
        dw.setWidget(e)
        dw.setVisible(False)

        e.setReadOnly(True)
        e.anchorClicked.connect(self.anchor_callback)          # Handle clicks on anchors

    def show(self):
        from .qt import QtGui, QtCore
        d = self.text.document()
        img_lines = []
        mlist = list(self.session.model_list())
        mlist.sort(key = lambda m: m.id)
        for m in mlist:
            qi = self.model_image(m)
            uri = "file://image%d" % m.id
            d.addResource(QtGui.QTextDocument.ImageResource, QtCore.QUrl(uri), qi)
            img_lines.append('<a href="%d"><img src="%s"></a>' % (m.id, uri))
        self.html = html = '\n'.join(img_lines)
        self.text.setHtml(html)

        from .qt import QtCore
        dw = self.dock_widget
        mw = self.session.main_window
        mw.addDockWidget(QtCore.Qt.TopDockWidgetArea, dw)
        dw.setVisible(True)

    def model_image(self, model):

        if hasattr(model, 'thumbnail_image'):
            return model.thumbnail_image
        v = self.session.view
        w,h = self.image_size
        from . import camera
        c = camera.camera_framing_models(w, h, [model])
        qi = v.image(w,h,c,[model])
        from . import qt
        qt.draw_image_text(qi, str(model.id), bgcolor = (0,0,0), font_size = 24)
        model.thumbnail_image = qi
        return qi

    def shown(self):
        return self.dock_widget.isVisible()

    def hide(self):
        mw = self.session.main_window
        mw.removeDockWidget(self.dock_widget)

    def anchor_callback(self, qurl):
        url = qurl.toString()
        id = int(url)
        for m in self.session.model_list():
            if m.id == id:
                m.display = not m.display
                m.redraw_needed = True

def show_model_panel(session):
    if not hasattr(session, 'model_panel'):
        session.model_panel = Model_Panel(session)
    mp = session.model_panel
    if mp.shown():
        mp.hide()
    else:
        mp.show()
