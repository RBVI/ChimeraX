import pytest
from chimerax.voice.speech import SpeechDecoder


@pytest.mark.parametrize(("audio_path", "audio_content"), [
    ("src/bundles/voice/test/data/assign right click to select.m4a", "assign right click to select")
    , ("src/bundles/voice/test/data/color by bfactor.m4a", "color by b factor")
    , ("src/bundles/voice/test/data/color bypolymer.m4a", "color by polymer")
    , ("src/bundles/voice/test/data/display #1 in ball and stick style.m4a", "display number one in ball and stick style")
    , ("src/bundles/voice/test/data/hbonds reveal true.m4a", "h bonds reveal true")
    , ("src/bundles/voice/test/data/make the background white.m4a", "make the background white")
    , ("src/bundles/voice/test/data/open 5y5s.m4a", "open 5y5s")
    , ("src/bundles/voice/test/data/set bgColor black.m4a", "set bg color black")
    , ("src/bundles/voice/test/data/show hydrogen bonds.m4a", "show hydrogen bonds")
    , ("src/bundles/voice/test/data/style ball.m4a", "style ball")
    , ("src/bundles/voice/test/data/ui mousemode right zoom.m4a", "ui mouse mode right zoom")
])
def test_decoding(audio_path, audio_content):
    decoder = SpeechDecoder()
    decoder.setAudioFile(audio_path)
    result = decoder.decode()
    assert result.getText() == audio_content
