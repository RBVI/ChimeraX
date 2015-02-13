# vim: set expandtab ts=4 sw=4:

from abc import ABCMeta, abstractmethod


class Log:
    """Base class for the "real" log classes: HtmlLog and PlainTextLog.
    """

    # log levels
    LEVEL_ERROR = "error"
    LEVEL_INFO = "info"
    LEVEL_WARNING = "warning"

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
        """
        return False


class HtmlLog(Log, metaclass=ABCMeta):
    """Base class for logs that support HTML output"""

    @abstractmethod
    def log(self, level, msg, image_info, is_html):
        """Log a message.

        Parameters
        ----------
        level : LEVEL_XXX constant from :class:`.Log' base class
            How important the message is (e.g. error, warning, info)
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


class PlainTextLog(Log, metaclass=ABCMeta):
    """Base class for logs that support only plain text output"""

    @abstractmethod
    def log(self, level, msg):
        """Log a message.

        Parameters
        ----------
        level : LOG_XXX constant from Log base class
            How important the message is (e.g. error, warning, info)
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
    :meth:`error`/
    :meth:`warning`/
    :meth:`info`/
    :meth:`status` methods
    to send messages to a log.  The message will be sent to the log at the
    top of the log stack and then each other log in order until one of
    the logs' log() methods returns True (i.e. message consumed).

    Message consumers must inherit from :class:`HtmlLog' or
    :class:`PlainTextLog` and register themselves with the Logger's
    :meth:`add_log` method, which will put them at the top of the log
    stack.  When quitting or otherwise no longer interested in receiving
    log messages they should deregister themselves with the
    :meth:`remove_log` method.  Consumers need to implement their log()
    abstract method, but need not implement the status() method if they are
    not interested in showing status.
    """

    def __init__(self, session):
        from ordered_set import OrderedSet
        self.logs = OrderedSet()
        self.session = session
        self._prev_newline = True

    def add_log(self, log):
        if not isinstance(log, (HtmlLog, PlainTextLog)):
            raise ValueError("Cannot add log that is not instance of"
                             " HtmlLog or PlainTextLog")
        if log in self.logs:
            # move to top
            self.logs.discard(log)
        self.logs.add(log)

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

        The parameters are the same as for the :meth:error method.
        """
        import sys
        self._log(Log.LEVEL_INFO, msg, add_newline, image, is_html,
                  last_resort=sys.stdout)

    def remove_log(self, log):
        self.logs.discard(log)

    def status(self, msg, **kw):
        print('status:', msg)

    def warning(self, msg, add_newline=True, image=None, is_html=False):
        """Log a warning message

        The parameters are the same as for the :meth:error method.
        """
        import sys
        self._log(Log.LEVEL_WARNING, msg, add_newline, image, is_html,
                  last_resort=sys.stderr)

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
        if self.logs:
            logs = self.logs
        elif getattr(self.session, 'ui', None) \
                and isinstance(self.session.ui, (HtmlLog, PlainTextLog)):
            logs = [self.session.ui]
        else:
            logs = []

        if add_newline:
            if is_html:
                msg += "<br>"
            else:
                msg += "\n"
        for log in logs:
            if isinstance(log, HtmlLog):
                args = (level, msg, (image, add_newline), is_html)
            else:
                args = (level, msg)
            if log.log(*args):
                # message consumed
                break
        else:
            if last_resort:
                msg = self._html_to_plain(msg, image, is_html)
                if prev_newline:
                    output = "{}: {}".format(level.upper(), msg)
                else:
                    output = msg
                print(output, end="", file=last_resort)
            return


def html_to_plain(html):
    """'best effort' to convert HTML to plain text"""
    from bs4 import BeautifulSoup
    return BeautifulSoup(html).get_text()
