import pytest
import os
from chimerax.speech.parser import NaturalLanguageAtomSpecParser


@pytest.mark.parametrize(
    ("input_text", "output_text"),
    [
        ("number one", "#1"),
        ("model one", "#1"),
        ("model number one", "#1"),
        ("number two submodel three", "#2.3"),
        ("number two, submodel three", "#2.3"),
        ("number one chains A through F", "#1/A-F"),
        (
            "all the atoms of residue 12, and the calcium atom of residue 14",
            ":12 :14@CA",
        ),
        ("residue 12, and calcium atoms on residue 14", ":12 :14@ca"),
        ("residue 12 and residue 14's calcium atoms", ":12 :14@ca"),
        ("models 1.1 and 2.1", "#1.1 #2.1"),
        ("model 1.1 and model 2.1", "#1.1 #2.1"),
        ("the nitrogen atom on model 1 chain A residue 1", "#1/A:1@N"),
        ("the nitrogen atom on residue 1 of chain A of model 1", "#1/A:1@N"),
        (
            "the nitrogen atom on residue 1 of chain A of model 1 and chain B on number 2",
            "#1/A:1@N #2/B",
        ),
    ],
)
def test_nl_atomspec_parser(input_text, output_text):
    parser = NaturalLanguageAtomSpecParser()
    result = parser.parse(input_text)
    assert result == output_text
