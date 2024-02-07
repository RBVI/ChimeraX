import pytest
import os
from chimerax.speech.parser import NaturalLanguageSetParser


@pytest.mark.parametrize(
    ("input_text, output_text"),
    [
        ("set the background color to blue", "set bgColor blue"),
        ("make the background black", "set bgColor black"),
        ("reset the background color", "~set bgColor"),
        ("make the background color white", "set bgColor white"),
        ("unset the background", "~set bgColor"),
        ("reset subdivision", "~set subdivision"),
        ("unset subdivision", "~set subdivision"),
        ("set the subdivision value to 1", "set subdivision 1"),
        ("make the subdivision 2", "set subdivision 2"),
        ("make the subdivision two", "set subdivision 2"),
    ],
)
def test_nl_set_parser(input_text, output_text):
    parser = NaturalLanguageSetParser()
    result = parser.parse(input_text)
    assert result == output_text
