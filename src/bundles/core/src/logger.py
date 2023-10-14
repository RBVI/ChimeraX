# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
logger: Application log support
===============================

Support classes for logging messages.
"""


class Log:
    """Base class for the "real" log classes: :py:class:`HtmlLog` and :py:class:`PlainTextLog`.

    Attributes
    ----------
    LEVEL_BUG : for bugs
    LEVEL_ERROR : for other error messages
    LEVEL_INFO : for informational messages
    LEVEL_WARNING : for warning messages
    """

    # log levels
    LEVEL_INFO = 0
    LEVEL_WARNING = 1
    LEVEL_ERROR = 2
    LEVEL_BUG = 3

    LEVEL_DESCRIPTS = ["note", "warning", "error", "bug"]

    # if excludes_other_logs is True, then if this log consumed the
    # message (log() returned True) downstream logs will not get
    # the message
    excludes_other_logs = False

    def status(self, msg, color, secondary):
        """Supported API. Show a status message.

        Parameters
        ----------
        msg : plain (non-HTML) text
            The message to display
        color : text or (r,g,b) tuple, r/g/b in range 0-1
            Color to display text in.  If log cannot understand color
            text string, use black instead.
        secondary : boolean
            Whether to show the status in the secondary status area.
            If the log doesn't support a secondary status area it should
            either drop the status or combine it with the last known
            primary status message.

        Returns
        -------
        True if the routine displayed/handled the status, False otherwise.

        This method is not abstract because a log is free to totally
        ignore/drop status messages.

        Note that this method may be called from a thread (due to the
        use of timers to get proper time delays) and that therefore
        special window toolkit handling may be necessary to get your
        code executed in the main thread (*e.g.*, session.ui.thread_safe()).
        """
        return False

    def log(self, level, msg):
        """Supported API. Log a message.

        Must be overriden by subclass.

        Parameters
        ----------
        level : LEVEL_XXX constant from :py:class:`Log` base class
            How important the message is (*e.g.*, error, warning, info)
        msg : text, possibly HTML
            Message to log

        """
        raise NotImplemented


# note: HtmlLog and PlainTextLog were originally abstract classes, but
# multiply inheriting from C++ wrapped classes (like Qt) is _very_
# problematic with metaclasses
class HtmlLog(Log):
    """Base class for logs that support HTML output"""

    def log(self, level, msg, image_info, is_html):
        """Supported API. Log a message.

        Parameters
        ----------
        level : LEVEL_XXX constant from :py:class:`Log` base class
            How important the message is (*e.g.*, error, warning, info)
        msg : text, possibly HTML
            Message to log
        image_info : a (image, boolean) 2-tuple
            *image* is either a PIL image or None (if there is no image
            to log).  The boolean indicates whether there should be a
            line break after the image.  When there is an image to log,
            *msg* param is alt text to use
        is_html : boolean
            Is the message text HTML or not

        Returns
        -------
        True if the routine displayed/handled the log message, False otherwise.
        """
        return False


class PlainTextLog(Log):
    """Base class for logs that support only plain text output"""

    def log(self, level, msg):
        """Supported API. Log a message.

        Parameters
        ----------
        level : LOG_XXX constant from Log base class
            How important the message is (*e.g.*, error, warning, info)
        msg : text
            Message to log

        Returns
        -------
        True if the routine displayed/handled the log message, False otherwise.
        """
        return False


class StringPlainTextLog(PlainTextLog):
    """Capture plain text messages in a string (similar to StringIO)"""

    excludes_other_logs = True

    def __init__(self, logger):
        super().__init__()
        self._msgs = []
        self.logger = logger

    def __enter__(self):
        self.logger.add_log(self)
        return self

    def __exit__(self, *exc_info):
        self.logger.remove_log(self)

    def log(self, level, msg):
        self._msgs.append(msg)
        return True

    def getvalue(self):
        return ''.join(self._msgs)


class StatusLogger:
    """Base class for classes that offer 'status' method."""

    def __init__(self, session):
        self.session = session
        self._status_timer1 = self._status_timer2 = None
        self._follow_timer1 = self._follow_timer2 = None

    def clear(self):
        if self._status_timer1:
            self._status_timer1.cancel()
            self._status_timer1 = None
        if self._status_timer2:
            self._status_timer2.cancel()
            self._status_timer2 = None
        if self._follow_timer1:
            self._follow_timer1.cancel()
            self._follow_timer1 = None
        if self._follow_timer2:
            self._follow_timer2.cancel()
            self._follow_timer2 = None

    def status(self, msg, color="black", log=False, secondary=False,
               blank_after=None, follow_with="", follow_time=20, follow_log=None, is_html=False):
        """Supported API. Show status."""
        if log:
            self.session.logger.info(msg, is_html=is_html)

        if is_html:
            msg = html_to_plain(msg)

        for l in self._prioritized_logs():
            if l.status(msg, color, secondary) and getattr(l, "excludes_other_logs", True):
                break
        if secondary:
            status_timer = self._status_timer2
            follow_timer = self._follow_timer2
            blank_default = 0
        else:
            status_timer = self._status_timer1
            follow_timer = self._follow_timer1
            blank_default = 15

        if status_timer:
            status_timer.cancel()
            status_timer = None
        if follow_timer:
            follow_timer.cancel()
            follow_timer = None

        from threading import Timer
        if follow_with:
            follow_timer = Timer(follow_time,
                lambda fw=follow_with, clr=color, log=log, sec=secondary,
                fl=follow_log: self._follow_timeout(fw, clr, log, sec, fl))
            follow_timer.daemon = True
            follow_timer.start()
        elif msg:
            if blank_after is None:
                blank_after = blank_default
            if blank_after:
                from threading import Timer
                status_timer = Timer(blank_after, lambda sec=secondary:
                                     self._status_timeout(sec))
                status_timer.daemon = True
                status_timer.start()

        if secondary:
            self._status_timer2 = status_timer
            self._follow_timer2 = follow_timer
        else:
            self._status_timer1 = status_timer
            self._follow_timer1 = follow_timer

    def _follow_timeout(self, follow_with, color, log, secondary, follow_log):
        if secondary:
            self._follow_timer2 = None
        else:
            self._follow_timer1 = None
        if follow_log is None:
            follow_log = log
        self.session.ui.thread_safe(self.status, follow_with, color=color,
                                    log=follow_log, secondary=secondary)

    def _status_timeout(self, secondary):
        if secondary:
            self._status_timer2 = None
        else:
            self._status_timer1 = None
        self.session.ui.thread_safe(self.status, "", secondary=secondary)


class Logger(StatusLogger):
    """Log/status message dispatcher

    Log/status message producers use the
    :py:meth:`error`/
    :py:meth:`warning`/
    :py:meth:`info`/
    :py:meth:`status` methods
    to send messages to a log.  The message will be sent to the log at the
    top of the log stack and then each other log in order.

    Message consumers must inherit from :py:class:`HtmlLog` or
    :py:class:`PlainTextLog` and register themselves with the Logger's
    :py:meth:`add_log` method, which will put them at the top of the log
    stack.  When quitting or otherwise no longer interested in receiving
    log messages they should deregister themselves with the
    :py:meth:`remove_log` method.  Consumers need to override their
    :py:meth:`Log.log` abstract method,
    but need not override the :py:meth:`Log.status` method
    if they are not interested in showing status.
    """

    def __init__(self, session):
        StatusLogger.__init__(self, session)
        from .orderedset import OrderedSet
        self._prev_newline = True
        self.logs = OrderedSet()
        self.method_map = {
            Log.LEVEL_BUG: self.bug,
            Log.LEVEL_ERROR: self.error,
            Log.LEVEL_WARNING: self.warning,
            Log.LEVEL_INFO: self.info
        }
        # only put in an excepthook if we're the first session:
        import sys
        if sys.excepthook == sys.__excepthook__:
            def ehook(*exc_info):
                if self.session.debug or not hasattr(self.session, "ui"):
                    from traceback import print_exception
                    print_exception(*exc_info, file=sys.__stderr__)
                else:
                    self.session.ui.thread_safe(self.report_exception, exc_info=exc_info)
            sys.excepthook = ehook

        # non-exclusively collate any early log messages, so that they
        # can also be sent to the first "real" log to hit the stack
        self.add_log(_EarlyCollator())

    def add_log(self, log):
        """Supported API. Add a logger"""
        if not isinstance(log, (HtmlLog, PlainTextLog)):
            raise ValueError("Cannot add log that is not instance of"
                             " HtmlLog or PlainTextLog")
        if isinstance(log, _EarlyCollator):
            self._early_collation = True
        elif self._early_collation:
            # main window only handles status messages, so in that case keep collating...
            if not hasattr(self.session, 'ui') or not self.session.ui.is_gui:
                log_is_main_window = False
            else:
                from chimerax.ui.gui import MainWindow
                log_is_main_window = isinstance(log, MainWindow)
            if not log_is_main_window:
                self._early_collation = None
                for cur_log in self.logs:
                    if isinstance(cur_log, _EarlyCollator):
                        early_collator = cur_log
                        break
                self.logs.discard(early_collator)
        elif log in self.logs:
            # move to top
            self.logs.discard(log)
        self.logs.add(log)
        if self._early_collation == None:
            self._early_collation = False
            early_collator.log_summary(self)

    def bug(self, msg, add_newline=True, image=None, is_html=False):
        """Supported API. Log a bug

        Parameters
        ----------
        msg : text
            Message to log, either plain text or HTML
        add_newline : boolean
            Whether to add a newline to the message before logging it
            (also whether there is a line break after an image)
        image : PIL image or None
            If not None, an image to log.  If an image is provided, then
            the :param:msg parameter is alt text to show for logs than
            cannot display images
        is_html : boolean
            Is the :param:msg text HTML or plain text
        """
        import sys
        self._log(Log.LEVEL_BUG, msg, add_newline, image, is_html, 
                  last_resort=sys.__stderr__)

    def clear(self):
        """Supported API. Clear all loggers"""
        StatusLogger.clear(self)
        self.logs.clear()

    def error(self, msg, add_newline=True, image=None, is_html=False):
        """Supported API. Log an error message

        Parameters
        ----------
        msg : text
            Message to log, either plain text or HTML
        add_newline : boolean
            Whether to add a newline to the message before logging it
            (also whether there is a line break after an image)
        image : PIL image or None
            If not None, an image to log.  If an image is provided, then
            the :param:msg parameter is alt text to show for logs than
            cannot display images
        is_html : boolean
            Is the :param:msg text HTML or plain text
        """
        import sys
        self._log(Log.LEVEL_ERROR, msg, add_newline, image, is_html, 
                  last_resort=sys.__stderr__)

    def info(self, msg, add_newline=True, image=None, is_html=False):
        """Supported API. Log an info message

        The parameters are the same as for the :py:meth:`error` method.
        """
        if self.session.silent:
            return
        import sys
        self._log(Log.LEVEL_INFO, msg, add_newline, image, is_html,
                  last_resort=sys.__stdout__)

    def remove_log(self, log):
        """Supported API. Remove a logger"""
        self.logs.discard(log)

    def report_exception(self, preface=None, error_description=None,
                         exc_info=None):
        """Supported API. Report the current exception (without changing execution context)

        Parameters
        ----------
        preface : text
            Prepend this text to the report
        error_description : text
            Replace any traceback information with this text
        """
        from .errors import NotABug, CancelOperation
        from traceback import format_exception_only, format_exception, format_tb
        if exc_info is not None:
            ei = exc_info
        else:
            import sys
            ei = sys.exc_info()
        if preface:
            preface = "%s:\n" % preface
        else:
            preface = ""

        exception_value = ei[1]

        if isinstance(exception_value, KeyboardInterrupt):
            self.session.ui.quit()

        if isinstance(exception_value, NotABug):
            self.error("%s%s" % (preface, exception_value))
        elif isinstance(exception_value, CancelOperation):
            pass  # Cancelled operations are not reported
        else:
            from html import escape
            if error_description:
                tb_msg = escape(error_description)
            else:
                tb = format_exception(ei[0], ei[1], ei[2])
                tb_msg = "".join(tb)
                # preserve exception traceback's indentation
                tmp = []
                for line in tb_msg.split('\n'):
                    text = line.lstrip()
                    num_spaces = len(line) - len(text)
                    tmp.append('&nbsp;' * num_spaces + escape(text))
                tb_msg = "<br>\n".join(tmp)
            if self.session.silent:
                self.error(tb_msg, is_html=True)
                return
            self.info(tb_msg, is_html=True)

            err = "".join(format_exception_only(ei[0], ei[1]))
            loc = "".join(format_tb(ei[2])[-1:])
            err_msg = "%s%s\n%s\n" % (preface, err, loc) + \
                "<i>See log for complete Python traceback.</i>\n"
            self.bug(err_msg.replace("\n", "<br>"), is_html=True)

    def status(self, msg, **kw):
        """Supported API. Show status."""
        if self.session.silent:
            return
        StatusLogger.status(self, msg, **kw)

    def warning(self, msg, add_newline=True, image=None, is_html=False):
        """Supported API. Log a warning message

        The parameters are the same as for the :py:meth:`error` method.
        """
        if self.session.silent:
            return
        import sys
        self._log(Log.LEVEL_WARNING, msg, add_newline, image, is_html,
                  last_resort=sys.__stderr__)

    def _html_to_plain(self, msg, image, is_html):
        if image:
            if msg:
                if is_html:
                    msg = html_to_plain(msg)
                if msg[0].isalnum() and msg[-1].isalnum():
                    msg = "[" + msg + "]"
            else:
                msg = "[image]"
        elif is_html:
            msg = html_to_plain(msg)
        return msg

    def _log(self, level, msg, add_newline, image, is_html, last_resort=None):
        prev_newline = self._prev_newline
        self._prev_newline = add_newline

        if add_newline:
            if is_html:
                msg += "<br/>"
            else:
                msg += "\n"

        msg_handled = False
        # "highest prority" log is last added, so:
        for log in reversed(list(self.logs)):
            if isinstance(log, HtmlLog):
                args = (level, msg, (image, add_newline), is_html)
            else:
                args = (level, self._html_to_plain(msg, image, is_html))
            if log.log(*args):
                # message displayed
                msg_handled = not self._early_collation
                if log.excludes_other_logs:
                    break

        if not msg_handled:
            if last_resort:
                msg = self._html_to_plain(msg, image, is_html)
                if prev_newline:
                    output = "{}: {}".format(Log.LEVEL_DESCRIPTS[level].upper(), msg)
                else:
                    output = msg
                print(output, end="", file=last_resort)

    def _prioritized_logs(self):
        # "highest priority" log is last added, so:
        return reversed(list(self.logs))


html_table_params = 'border=1 cellpadding=4 cellspacing=0'

class CollatingLog(HtmlLog):
    """Collates log messages

    This class is designed to be used via the :py:class:`Collator` context manager.
    Please see that class for documentation.
    """

    excludes_other_logs = True

    sim_test_size = 10
    sim_collapse_after = 5

    MAX_COLLATION_LEVEL = HtmlLog.LEVEL_ERROR

    def __init__(self):
        self.msgs = []
        for _ in range(self.MAX_COLLATION_LEVEL+1):
            self.msgs.append([])

    def log(self, level, msg, image_info, is_html):
        if level <= self.MAX_COLLATION_LEVEL:
            self.msgs[level].append((msg, image_info, is_html))
            return True
        return False

    def log_summary(self, logger, summary_title, collapse_similar=True):
        # note that this handling of the summary (only calling logger,info
        # at the end and not calling the individual log-level functions)
        # will not raise an error dialog except for 'bug'-level log entries
        summary = '\n<table %s>\n' % html_table_params
        summary += '  <thead>\n'
        summary += '    <tr>\n'
        summary += '      <th colspan="2">%s</th>\n' % summary_title
        summary += '    </tr>\n'
        summary += '  </thead>\n'
        summary += '  <tbody>\n'
        some_msgs = False
        colors = ["#ffffff", "#ffb961", "#ff7882", "#dc1436"]
        for level, msgs in reversed(list(enumerate(self.msgs))):
            if not msgs:
                continue
            some_msgs = True
            summary += '    <tr>\n'
            summary += '      <td><i>%s%s</i></td>' % (
                self.LEVEL_DESCRIPTS[level], 's' if len(msgs) > 1 else '')
            summary += '      <td style="background-color:%s">%s</td>' % (colors[level], self.summarize_msgs(msgs, collapse_similar))
            summary += '    </tr>\n'
        if some_msgs:
            summary += '  </tbody>\n'
            summary += '</table>'
            logger.info(summary, is_html=True)

    def summarize_msgs(self, msgs, collapse_similar):
        # For plain text messages, escape < and > otherwise <stuff between angle brackets>
        # disappears in html output.
        import html
        msgs = [(m if is_html else html.escape(m), image_info) for m, image_info, is_html in msgs]

        import sys
        if collapse_similar:
            summarized = []
            prev_msg = sim_info = None
            for msg, image_info in msgs:
                # Judge similarity to preceding messages and perhaps collapse...
                if image_info[0] is not None:
                    # always log images
                    if sim_info:
                        sim_reps = sim_info[0]
                        if sim_reps > self.sim_collapse_after:
                            summarized.append("{} messages similar to the above omitted\n".format(
                                sim_reps - self.sim_collapse_after))
                    prev_msg = sim_info = None
                    summarized.append(image_info_to_html(msg, image_info))
                elif sim_info:
                    sim_reps, sim_type, sim_data = sim_info
                    st = self.sim_test_size
                    if sim_type == "front":
                        similar = msg[:2*st] == sim_data
                    elif sim_type == "back":
                        similar = msg[-2*st:] == sim_data
                    else:
                        similar = msg[st:] == sim_data[0] \
                            and msg[-st:] == sim_data[1]
                    if similar:
                        sim_reps += 1
                        sim_info = (sim_reps, sim_type, sim_data)
                        if sim_reps >= self.sim_collapse_after+1:
                            continue
                        # let first few reps get logged individually...
                    else:
                        if sim_reps > self.sim_collapse_after:
                            summarized.append("{} messages similar to the above omitted\n".format(
                                sim_reps - self.sim_collapse_after))
                        sim_info = None
                elif prev_msg is not None:
                    st = self.sim_test_size
                    similar = True
                    if msg[:2*st] == prev_msg[:2*st]:
                        sim_type = "front"
                        sim_data = msg[:2*st]
                    elif msg[-2*st:] == prev_msg[-2*st:]:
                        sim_type = "back"
                        sim_data = msg[-2*st:]
                    elif msg[:st] == prev_msg[:st] \
                    and msg[-st:] == prev_msg[-st:]:
                        sim_type = "ends"
                        sim_data = (msg[:st], msg[-st:])
                    else:
                        similar = False
                    if similar:
                        sim_info = (2, sim_type, sim_data)
                summarized.append(msg)
                prev_msg = msg
            if sim_info:
                sim_reps = sim_info[0]
                if sim_reps > self.sim_collapse_after:
                    summarized.append("{} messages similar to the above omitted\n".format(
                        sim_reps - self.sim_collapse_after))
        else:
            summarized = msgs
        return "".join(summarized).strip().replace('\n', '<br>')

class Collator:
    """Context manager for a CollatingLog

    This class is designed to be used when some operation may produce
    many log messages that would be more convenient to present as one
    combined message.  Since this class is a context manager, you simply
    surround the code whose messages you want collated with a 'with' context
    using an instance of this class, e.g.::

        from chimerax.core.logger import Collator
        with Collator(session.logger, "Problems found while doing X"):
            ...code to collate...

    Parameters
    ----------
    logger : the session's :py:class:`Logger`
        The logger to use.
    summary_title : string
        What to title the log summary.
    log_messages : boolean
        Whether or not to actually log the collated messages, defaults True.
    """

    def __init__(self, logger, summary_title, log_messages=True):
        self.logger = logger
        self.summary_title = summary_title
        self.log_messages = log_messages
        self.collater = CollatingLog()

    def __enter__(self):
        self.logger.add_log(self.collater)
        return self

    def __exit__(self, *exc_info):
        self.logger.remove_log(self.collater)
        if self.log_messages:
            self.collater.log_summary(self.logger, self.summary_title)

class _EarlyCollator(CollatingLog):
    """Collate any errors that occur before any "real" log hits the log stack."""
    excludes_other_logs = False

    MAX_COLLATION_LEVEL = CollatingLog.LEVEL_BUG

    def log_summary(self, logger):
        if self.msgs[self.LEVEL_ERROR] or self.msgs[self.LEVEL_BUG]:
            title = "Startup Errors"
        else:
            title = "Startup Messages"
        CollatingLog.log_summary(self, logger, title)

#error_text_format = '<p style="color:crimson;font-weight:bold">%s</p>'
# although the below isn't HTML5, it avoids the line break in the above
error_text_format = '<font color="crimson"><b>%s</b></font>'

def html_to_plain(html):
    """'best effort' to convert HTML to plain text"""
    try:
        import html2text
    except ModuleNotFoundError:
        return html
    h = html2text.HTML2Text()
    h.unicode_snob = True
    # h.pad_tables = True  # 2018.1.9 is confused by multiline data in td
    # h.body_width = ?  # TODO: track terminal size changes
    return h.handle(html)

def image_info_to_html(msg, image_info):
    image, image_break = image_info
    import io
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    png_data = img_io.getvalue()
    import codecs
    bitmap = codecs.encode(png_data, 'base64')
    width, height = image.size
    img_src = '<img src="data:image/png;base64,%s" width=%d height=%d style="vertical-align:middle">'  % (
        bitmap.decode('utf-8'), width, height)
    if image_break:
        img_src += "<br>\n"
    return img_src

def log_version(logger):
    '''Show version information.'''
    from chimerax.core import buildinfo
    from chimerax import app_dirs as ad
    from . import toolshed
    t = toolshed.get_toolshed()
    if t:
        b = t.find_bundle('ChimeraX-Core', logger, True)
        version = b.version
    else:
        version = ad.version
    logger.info("%s %s version: %s (%s)" % (ad.appauthor, ad.appname, version, buildinfo.date.split()[0]))
    logger.info(buildinfo.copyright)
