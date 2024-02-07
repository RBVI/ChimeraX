import pytest
import os
from chimerax.speech.parser import NaturalLanguageSelectParser


@pytest.mark.parametrize(
    ("input_text", "output_text"),
    [
        ("select number one", "select #1"),
        ("select model one", "select #1"),
        ("select model number one", "select #1"),
        ("select number two submodel three", "select #2.3"),
        ("select number two, submodel three", "select #2.3"),
        ("add number one chains A through F to the selection", "select add #1/A-F"),
        (
            "select all the atoms of residue 12, and the calcium atom of residue 14",
            "select :12 :14@CA",
        ),
        ("select residue 12, and calcium atoms on residue 14", "select :12 :14@ca"),
        ("select residue 12 and residue 14's calcium atoms", "select :12 :14@ca"),
        ("select models 1.1 and 2.1", "select #1.1 #2.1"),
        ("add model 1.1 and model 2.1 to the selection", "select add #1.1 #2.1"),
        ("select the nitrogen atom on model 1 chain A residue 1", "select #1/A:1@N"),
        (
            "select the nitrogen atom on residue 1 of chain A of model 1",
            "select #1/A:1@N",
        ),
        (
            "select the nitrogen atom on residue 1 of chain A of model 1 and chain B on number 2",
            "select #1/A:1@N #2/B",
        ),
        ("clear the selection", "~select"),
        ("remove model one from the selection", "~select #1"),
    ],
)
def test_nl_atomspec_parser(input_text, output_text):
    parser = NaturalLanguageSelectParser()
    result = parser.parse(input_text)
    assert result == output_text
