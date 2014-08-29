# vim: set expandtab ts=4 sw=4:

class Log:
    '''
    Log window for command output.
    '''
    def __init__(self, main_window):
        self._main_window = main_window
        self._html_text = ''
        self.thumbnail_size = 128       # Pixels
        self._image_count = 0

    def show(self, toggle = True):
        mw = self._main_window
        if mw.showing_text() and mw.text_id == 'log' and toggle:
            mw.show_graphics()
        else:
#            mw.show_text(self._html_text, html = True, id = "log", open_links = True)
            mw.show_text(self._html_text, html = True, id = "log", scroll_to_end = True)
    def log_message(self, text, color = None, html = False):
        if html:
            htext = text
        else:
            style = '' if color is None else ' style="color:%s;"' % color
            import cgi
            etext = cgi.escape(text)
            htext = '<pre%s>%s</pre>\n' % (style,etext)
        self._html_text += htext

    def insert_graphics_image(self, format = 'JPG'):
        mw = self._main_window
        v = mw.view
        s = self.thumbnail_size
        i = v.image(s,s)
        self._image_count += 1
        uri = "file://image%d" % (self._image_count,)
        mw.register_html_image_identifier(uri, i)
        htext = '<br><img src="%s"><br>\n' % (uri,)
        self._html_text += htext

    def exceptions_to_log(self):
        import sys
        sys.excepthook = self._log_exception

    def _log_exception(self, type, value, traceback):
        from traceback import format_exception
        lines = format_exception(type, value, traceback)
        import cgi
        elines = tuple(cgi.escape(line) for line in lines)
        tb = '<p style="color:#A00000;">\n%s</p>' % '<br><br>'.join(elines)
        self.log_message(tb, html = True)
        self.show(toggle = False)

    def stdout_to_log(self):
        import sys
        sys.stdout_orig = sys.stdout
        sys.stdout = self._output_stream()

    def _output_stream(self):
        class Log_Output_Stream:
            def __init__(self, log):
                self.log = log
                self.text = ''
            def write(self, text):
                self.text += text
                if text.endswith('\n'):
                    self.log.log_message(self.text.rstrip())
                    self.text = ''
            def flush(self):
                if self.text:
                    self.log.log_message(self.text.rstrip())
                    self.text = ''
        return Log_Output_Stream(self)

