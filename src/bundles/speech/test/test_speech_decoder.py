import pytest
import os
from chimerax.speech.speech import SpeechDecoder


def sample_path(filename):
    return os.path.join(os.path.dirname(__file__), "data", filename)


@pytest.mark.parametrize(
    ("audio_path", "audio_content"),
    [
        (
            sample_path("assign right click to select.m4a"),
            "assign right click to select",
        ),
        (
            sample_path("color by bfactor.m4a"),
            "color by b factor",
        ),
        (
            sample_path("color bypolymer.m4a"),
            "color by polymer",
        ),
        (
            sample_path("display #1 in ball and stick style.m4a"),
            "display number one in ball and stick style",
        ),
        (
            sample_path("hbonds reveal true.m4a"),
            "h bonds reveal true",
        ),
        (
            sample_path("make the background white.m4a"),
            "make the background white",
        ),
        (
            sample_path("open 5y5s.m4a"),
            "open 5y5s",
        ),
        (
            sample_path("set bgColor black.m4a"),
            "set bg color black",
        ),
        (
            sample_path("show hydrogen bonds.m4a"),
            "show hydrogen bonds",
        ),
        (
            sample_path("style ball.m4a"),
            "style ball",
        ),
        (
            sample_path("ui mousemode right zoom.m4a"),
            "ui mouse mode right zoom",
        ),
    ],
)
def test_decoding(audio_path, audio_content):
    decoder = SpeechDecoder()
    result = decoder.decode_file(audio_path)
    assert result.getText() == audio_content
