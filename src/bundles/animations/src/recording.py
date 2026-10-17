"""Animation-specific movie recording synchronization."""


def pathtrace_frame_complete(session):
    """Whether the displayed animation frame is ready to be captured."""
    view = session.main_view
    if not getattr(view.lighting, "pathtrace", False):
        return True
    renderer = getattr(view, "_render", None)
    return bool(
        renderer is not None
        and getattr(renderer, "_pathtrace_accumulation_complete", False)
    )


def configure_animation_recording(session):
    """Make movie capture wait for progressive rendering of each frame."""
    from .settings import get_settings

    samples = get_settings(session).recording_pathtrace_samples
    movie = session.movie
    movie.wait_for_rendering = True
    movie.frame_capture_ready = _PathTraceCaptureGate(session, samples)


class _PathTraceCaptureGate:
    """Gate captures and temporarily cap path-trace work per movie frame."""

    _missing = object()

    def __init__(self, session, samples):
        self.session = session
        self.renderer = renderer = getattr(session.main_view, "_render", None)
        self.previous_limit = self._missing
        if renderer is not None:
            attr = "_pathtrace_recording_max_accum"
            self.previous_limit = getattr(renderer, attr, self._missing)
            setattr(renderer, attr, max(int(samples), 1))

    def __call__(self):
        return pathtrace_frame_complete(self.session)

    def close(self):
        renderer = self.renderer
        if renderer is None:
            return
        attr = "_pathtrace_recording_max_accum"
        if self.previous_limit is self._missing:
            if hasattr(renderer, attr):
                delattr(renderer, attr)
        else:
            setattr(renderer, attr, self.previous_limit)
        self.renderer = None
        # The recorded frame stopped accumulating at the movie-specific
        # target.  Resume interactive refinement toward the normal lighting
        # target after the recording gate is removed.
        self.session.main_view.redraw_needed = True
