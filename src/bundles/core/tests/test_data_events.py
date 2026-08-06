import pytest


def test_tracker_works_as_context_manager():
    # Tracker.__exit__ took only self, so leaving a `with` block raised
    # "TypeError: __exit__() takes 1 positional argument but 4 were given"
    # and, because __exit__ never ran, left the tracker blocked.
    from chimerax.core.data_events import Tracker

    t = Tracker()
    before = t._blocked
    with t:
        assert t._blocked == before + 1
    assert t._blocked == before


def test_tracker_releases_on_exception():
    from chimerax.core.data_events import Tracker

    t = Tracker()
    before = t._blocked
    with pytest.raises(ValueError):
        with t:
            raise ValueError("boom")
    assert t._blocked == before


def test_tracker_enter_returns_the_tracker():
    from chimerax.core.data_events import Tracker

    t = Tracker()
    with t as entered:
        assert entered is t
