from os.path import expanduser, exists
data_path = expanduser('~/.boltz')
if not exists(data_path):
    from os import mkdir
    mkdir(data_path)

from boltz.main import download
from pathlib import Path
download(Path(data_path))
