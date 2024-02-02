import pytest
import os
from chimerax.speech.speech import SpeechDecoder


@pytest.mark.parametrize(
    ("audio_path", "audio_content"),
    [
        (
            os.path.join(
                os.path.dirname(__file__), "data", "assign right click to select.m4a"
            ),
            "assign right click to select",
        ),
        (
            os.path.join(os.path.dirname(__file__), "data", "color by bfactor.m4a"),
            "color by b factor",
        ),
        (
            os.path.join(os.path.dirname(__file__), "data", "color bypolymer.m4a"),
            "color by polymer",
        ),
        (
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "display #1 in ball and stick style.m4a",
            ),
            "display number one in ball and stick style",
        ),
        (
            os.path.join(os.path.dirname(__file__), "data", "hbonds reveal true.m4a"),
            "h bonds reveal true",
        ),
        (
            os.path.join(
                os.path.dirname(__file__), "data", "make the background white.m4a"
            ),
            "make the background white",
        ),
        (
            os.path.join(os.path.dirname(__file__), "data", "open 5y5s.m4a"),
            "open 5y5s",
        ),
        (
            os.path.join(os.path.dirname(__file__), "data", "set bgColor black.m4a"),
            "set bg color black",
        ),
        (
            os.path.join(os.path.dirname(__file__), "data", "show hydrogen bonds.m4a"),
            "show hydrogen bonds",
        ),
        (
            os.path.join(os.path.dirname(__file__), "data", "style ball.m4a"),
            "style ball",
        ),
        (
            os.path.join(
                os.path.dirname(__file__), "data", "ui mousemode right zoom.m4a"
            ),
            "ui mouse mode right zoom",
        ),
    ],
)
def test_decoding(audio_path, audio_content):
    decoder = SpeechDecoder()
    result = decoder.decode_file(audio_path)
    assert result.getText() == audio_content
