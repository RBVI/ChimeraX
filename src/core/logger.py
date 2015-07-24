# vi: set expandtab ts=4 sw=4:
"""
logger: application log support
===============================

This module is very important.
"""


class Log:
    """Base class for the "real" log classes: :py:class:`HtmlLog` and :py:class:`PlainTextLog`.

    Attributes
    ----------
    LEVEL_ERROR : for error messages
    LEVEL_INFO : for informational messages
    LEVEL_WARNING : for warning messages
    """

    # log levels
    LEVEL_INFO = 0
    LEVEL_WARNING = 1
    LEVEL_ERROR = 2

    LEVEL_DESCRIPTS = ["info", "warning", "error"]

    # if excludes_other_logs is True, then if this log consumed the
    # message (log() returned True) downstream logs will not get
    # the message
    excludes_other_logs = False

    def status(self, msg, color, secondary):
        """Show a status message.

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
        code executed in the main thread (*e.g.*, wx.CallAfter).
        """
        return False

    def log(self, level, msg):
        """Log a message.

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
# multiply inheriting from C++ wrapped classes (like Wx) is _very_
# problematic with metaclasses
class HtmlLog(Log):
    """Base class for logs that support HTML output"""

    def log(self, level, msg, image_info, is_html):
        """Log a message.

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
        """Log a message.

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


class Logger:
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

    If the Logger :py:attr:`collapse_similar` attribute is True, then
    after a few occurances of consecutive similar log messages, the
    remainder will be collapsed into a single log message noting how
    many additional occurances there were.
    """

    _sim_test_size = 10
    _sim_collapse_after = 5

    def __init__(self, session):
        from chimera.core.orderedset import OrderedSet
        self.logs = OrderedSet()
        self.session = session
        self._prev_newline = True
        self._status_timer1 = self._status_timer2 = None
        self._follow_timer1 = self._follow_timer2 = None
        self._sim_info = None
        self._prev_info = None
        self._sim_timer = None
        self._collapse_similar = False
        self.method_map = {
            Log.LEVEL_ERROR: self.error,
            Log.LEVEL_WARNING: self.warning,
            Log.LEVEL_INFO: self.info
        }

    def add_log(self, log):
        """Add a logger"""
        if not isinstance(log, (HtmlLog, PlainTextLog)):
            raise ValueError("Cannot add log that is not instance of"
                             " HtmlLog or PlainTextLog")
        if log in self.logs:
            # move to top
            self.logs.discard(log)
        self.logs.add(log)

    @property
    def collapse_similar(self):
        return self._collapse_similar

    @collapse_similar.setter
    def collapse_similar(self, cs):
        if cs == self._collapse_similar:
            return
        self._collapse_similar = cs
        if not cs:
            if self._sim_timer != None:
                self._sim_timer.cancel()
                self._sim_timer_cb()
            else:
                self._sim_info = None

    def clear(self):
        """clear all loggers"""
        self.logs.clear()
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
        if self._sim_timer:
            self._sim_timer.cancel()
            self._sim_timer = None
        self._prev_info = self._sim_info = None

    def error(self, msg, add_newline=True, image=None, is_html=False):
        """Log an error message

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
                  last_resort=sys.stderr)

    def info(self, msg, add_newline=True, image=None, is_html=False):
        """Log an info message

        The parameters are the same as for the :py:meth:`error` method.
        """
        import sys
        self._log(Log.LEVEL_INFO, msg, add_newline, image, is_html,
                  last_resort=sys.stdout)

    def remove_log(self, log):
        """remove a logger"""
        if log.excludes_other_logs and self._sim_timer is not None:
            self._sim_timer.cancel()
            self._sim_timer_cb()
        self.logs.discard(log)

    def status(self, msg, color="black", log=False, secondary=False,
               blank_after=None, follow_with="", follow_time=20,
               follow_log=None):
        """Show status."""
        if log:
            self.info(msg)

        for log in self.logs:
            log.status(msg, color, secondary)
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
            follow_timer = Timer(
                follow_time,
                lambda fw=follow_with, clr=color, log=log, sec=secondary,
                fl=follow_log: self._follow_timeout(fw, clr, log, sec, fl))
            follow_timer.start()
        elif msg:
            if blank_after is None:
                blank_after = blank_default
            if blank_after:
                from threading import Timer
                status_timer = Timer(blank_after, lambda sec=secondary:
                                     self._status_timeout(sec))
                status_timer.start()

        if secondary:
            self._status_timer2 = status_timer
            self._follow_timer2 = follow_timer
        else:
            self._status_timer1 = status_timer
            self._follow_timer1 = follow_timer

    def warning(self, msg, add_newline=True, image=None, is_html=False):
        """Log a warning message

        The parameters are the same as for the :py:meth:`error` method.
        """
        import sys
        self._log(Log.LEVEL_WARNING, msg, add_newline, image, is_html,
                  last_resort=sys.stderr)

    def _follow_timeout(self, follow_with, color, log, secondary, follow_log):
        if secondary:
            self._follow_timer2 = None
        else:
            self._follow_timer1 = None
        if follow_log is None:
            follow_log = log
        self.status(follow_with, color=color, log=follow_log,
                    secondary=secondary)

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

        if self.collapse_similar:
            # Judge similarity to preceding messages and perhaps collapse...
            if self._sim_info:
                sim_level, sim_reps, sim_type, sim_data = self._sim_info
                st = self._sim_test_size
                if sim_level != level:
                    similar = False
                elif sim_type == "front":
                    similar = msg[:2*st] == sim_data
                elif sim_type == "back":
                    similar = msg[-2*st:] == sim_data
                else:
                    similar = msg[st:] == sim_data[0] \
                        and msg[-st:] == sim_data[1]
                if similar:
                    sim_reps += 1
                    self._sim_info = (sim_level, sim_reps, sim_type, sim_data)
                    if sim_reps >= self._sim_collapse_after+1:
                        if self._sim_timer is not None:
                            self._sim_timer.cancel()
                        from threading import Timer
                        self._sim_timer = Timer(0.5, self._sim_timer_cb)
                        self._sim_timer.start()
                        return
                    # let first few reps get logged immediately...
                else:
                    if self._sim_timer is not None:
                        self._sim_timer.cancel()
                        self._sim_timer_cb()
                    else:
                        self._sim_info = None
            elif self._prev_info is not None:
                st = self._sim_test_size
                prev_level, prev_msg = self._prev_info
                if level == prev_level:
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
                        self._sim_info = (level, 2, sim_type, sim_data)
        self._prev_info = (level, msg)

        msg_handled = False
        # "highest prority" log is last added, so:
        for log in reversed(list(self.logs)):
            if isinstance(log, HtmlLog):
                args = (level, msg, (image, add_newline), is_html)
            else:
                args = (level, self._html_to_plain(msg, image, is_html))
            if log.log(*args):
                # message displayed
                msg_handled = True
                if log.excludes_other_logs:
                    break

        if not msg_handled:
            if last_resort:
                msg = self._html_to_plain(msg, image, is_html)
                if prev_newline:
                    output = "{}: {}".format(level.upper(), msg)
                else:
                    output = msg
                print(output, end="", file=last_resort)

    def _sim_timer_cb(self):
        self._sim_timer = None
        level, reps = self._sim_info[:2]
        self._sim_info = None
        self._log(level, "{} messages similar to the above omitted".format(
            reps - self._sim_collapse_after), True, None, False)


    def _status_timeout(self, secondary):
        if secondary:
            self._status_timer2 = None
        else:
            self._status_timer1 = None
        self.status("", secondary=secondary)


class CollatingLog(PlainTextLog):
    """Collates log messages

    This class is designed to be used when some operation may produce
    many log messages that would be more convenient to present as one
    combined message.  You call the logger's :py:meth:`~Logger.add_log`
    method to start collating, and remove it with :py:meth:`~Logger.remove_log`
    to stop collating.  If the operation may produce many consecutive
    simiilar (or identical) log messagesm you may also want to set the logger's
    :py:attr:`~Logger.collapse_similar` attribute to True after adding
    the log, and set it back to its original value before removing the log.
    
    To get the collated messages, call :py:meth:`summarize` on the log.
    That will return a 2-tuple consisting of the maximum log level of the
    messages, and their combined text.  The text will list the errors,
    warnings, etc. in separate sections of the text.  To log the result,
    use the logger's :py:attr:`~Logger.method_map` dictionary to convert
    the maximum level to a method to call, and call that method with 
    the summary as an argument (possibly preceded with some introductory
    text) and with the `add_newline` keyword set to False.
    """

    excludes_other_logs = True

    def __init__(self):
        self.max_level = -1
        self.msgs = []
        for _ in range(len(self.LEVEL_DESCRIPTS)):
            self.msgs.append([])
 
    def log(self, level, msg):
        self.msgs[level].append(msg)
        self.max_level = max(self.max_level, level)
        return True

    def summarize(self):
        if self.max_level < 0:
            return -1, ""
        msg = ""
        for level in range(self.max_level, -1, -1):
            msgs = self.msgs[level]
            if not msgs:
                continue
            if msg:
                msg += "\n"
            msg += "{}{}:\n".format(
                self.LEVEL_DESCRIPTS[level].capitalize(),
                "s" if len(msgs) > 1 else "")
            msg += "".join(msgs)
        return self.max_level, msg


def html_to_plain(html):
    """'best effort' to convert HTML to plain text"""
    from bs4 import BeautifulSoup
    # return BeautifulSoup(html).get_text() -- loses line breaks
    bs = BeautifulSoup(html)
    x = []
    for result in bs:
        s = result.string
        x.append(s if s is not None else '\n')
    return ''.join(x)
