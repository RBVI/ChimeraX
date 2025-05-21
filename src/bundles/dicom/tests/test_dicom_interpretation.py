import os
import pytest

test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(test_data_dir):
    pytest.skip("Skipping DICOM opening tests because test data is unavailable.",
    allow_module_level=True,
)

test_cases = os.listdir(test_data_dir)

@pytest.mark.parametrize('data', test_cases)
def test_opening_dicom(test_production_session, data):
    if 'RT Plan' in data:
        pytest.xfail("RTPlan is not yet supported")
    session = test_production_session
    test_file = os.path.join(test_data_dir, data)
    
    from chimerax.core.commands import run
    run(session, "open \"%s\" format dicom" % test_file)

