"""Status-only windows must not cause discarded HTML conversions."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def status_logger():
    from chimerax.core.logger import Logger
    from chimerax.ui.gui import MainWindow

    # The log/status methods do not need a native window. Preserve its actual
    # type contract for the dispatcher's isinstance checks.
    window = Mock(spec=MainWindow)
    window.log = lambda *args: MainWindow.log(window, *args)
    window.status = lambda *args: MainWindow.status(window, *args)
    messages = []
    window._status_bar = SimpleNamespace(status=lambda *args: messages.append(args))
    logger = Logger.__new__(Logger)
    logger.logs = [window]
    logger._prev_newline = True
    logger._early_collation = False
    logger.session = SimpleNamespace(silent=False)
    logger._status_timer1 = logger._status_timer2 = None
    logger._follow_timer1 = logger._follow_timer2 = None
    return logger, messages


def test_html_passes_to_real_log_without_plain_conversion(status_logger, monkeypatch):
    from chimerax.core import logger as logging
    logger, _ = status_logger
    received = []

    class Consumer(logging.HtmlLog):
        def log(self, *args):
            received.append(args)
            return True

    logger.logs.append(Consumer())
    def unexpected_conversion(*args):
        pytest.fail('Status-only window requested an unused plain-text conversion')
    monkeypatch.setattr(logging, 'html_to_plain', unexpected_conversion)
    logger.info('<b>hello</b>', is_html=True)
    assert received == [(logging.Log.LEVEL_INFO, '<b>hello</b><br/>', (None, True), True)]


def test_plain_consumer_still_gets_converted_html(status_logger):
    from chimerax.core.logger import StringPlainTextLog
    logger, _ = status_logger
    capture = StringPlainTextLog(logger)
    logger.logs.append(capture)
    logger.info('<b>hello</b>', is_html=True)
    assert capture.getvalue().strip() == '**hello**'


def test_fallback_output_is_not_swallowed(status_logger):
    from chimerax.core.logger import Log
    logger, _ = status_logger
    fallback = StringIO()
    logger._log(Log.LEVEL_INFO, '<b>hello</b>', True, None, True,
                last_resort=fallback)
    assert fallback.getvalue().strip() == 'NOTE: **hello**'


def test_status_messages_still_reach_status_bar(status_logger):
    logger, messages = status_logger
    logger.status('working', color='red', secondary=True, blank_after=0)
    assert messages == [('working', 'red', True)]
