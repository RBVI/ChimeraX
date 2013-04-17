import chimera
from chimera import preset
import Midas
import os
import webgl

models = chimera.openModels.list(modelTypes=[chimera.Molecule])
assert len(models) == 1, 'Only one molecule works'

preset.preset(allAtoms=True)
Midas.represent('bs', '#')
Midas.wait(1)

for m in models:
    filename = 'pdb%s_atoms.py' % os.path.splitext(m.name)[0]
    with open(filename, 'w') as f:
        f.write('# created from %s\n' % m.name)
        f.write('data = ')
        webgl.write_json(f)
