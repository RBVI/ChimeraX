from types import SimpleNamespace

from chimerax.animations.recording import (
    _PathTraceCaptureGate,
    pathtrace_frame_complete,
)


def _session(*, pathtrace, complete=None):
    renderer = SimpleNamespace()
    if complete is not None:
        renderer._pathtrace_accumulation_complete = complete
    view = SimpleNamespace(
        lighting=SimpleNamespace(pathtrace=pathtrace),
        _render=renderer,
        redraw_needed=False,
    )
    return SimpleNamespace(main_view=view)


def test_rasterized_frame_is_immediately_complete():
    assert pathtrace_frame_complete(_session(pathtrace=False))


def test_pathtraced_frame_waits_for_accumulation_target():
    assert not pathtrace_frame_complete(_session(pathtrace=True, complete=False))
    assert pathtrace_frame_complete(_session(pathtrace=True, complete=True))


def test_animation_recording_installs_movie_capture_gate():
    session = _session(pathtrace=True, complete=False)
    session.main_view._render._pathtrace_recording_max_accum = 128
    session.movie = SimpleNamespace(
        wait_for_rendering=False,
        frame_capture_ready=None,
    )
    gate = _PathTraceCaptureGate(session, 64)
    session.movie.wait_for_rendering = True
    session.movie.frame_capture_ready = gate

    try:
        assert session.movie.wait_for_rendering
        assert session.main_view._render._pathtrace_recording_max_accum == 64
        assert not session.movie.frame_capture_ready()
        session.main_view._render._pathtrace_accumulation_complete = True
        assert session.movie.frame_capture_ready()
    finally:
        gate.close()

    assert session.main_view._render._pathtrace_recording_max_accum == 128
    assert session.main_view.redraw_needed
