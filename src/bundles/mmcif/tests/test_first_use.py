"""The native mmCIF reader must work before another format loads atomic types."""
import subprocess
import sys


def test_mmcif_can_be_the_first_molecular_format_imported():
    # An existing PDB model materializes atomic.Bonds and hides this failure.
    code = '''import sys
import chimerax.atomic as atomic
assert 'Bonds' not in vars(atomic)
assert 'chimerax.atomic._data_model' not in sys.modules
import chimerax.mmcif
from chimerax.mmcif import _mmcif
from chimerax.atomic.molarray import Bonds
assert vars(atomic)['Bonds'] is Bonds
assert _mmcif.quote_value('two words') == '"two words"'
'''
    subprocess.run([sys.executable, '-I', '-c', code], check=True,
                   capture_output=True, text=True)
