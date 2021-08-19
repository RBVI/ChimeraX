# Create session
from chimerax.core.session import Session
session = Session('cx standalone', offscreen_rendering = True)

# Setup residue template path.
from chimerax.atomic import initialize_atomic
initialize_atomic(session)

# Open PDB file
from chimerax.pdb import open_pdb
from os.path import expanduser
models, msg = open_pdb(session, expanduser('~/Downloads/ChimeraX/PDB/1a0s.pdb'))
session.models.add(models)

s = models[0]
print('%s has %d atoms, %d bonds, %d residues, centroid %s, average atom radius %.2f'
      % (s.name, s.num_atoms, s.num_bonds, s.num_residues,
         tuple(s.atoms.coords.mean(axis=0)), s.atoms.radii.mean()))

# Save an image
from chimerax.image_formats import save_image
save_image(session, 'test.png')
