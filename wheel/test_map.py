# Create session
from chimerax.core.session import Session
session = Session('cx standalone', offscreen_rendering = True)

# Open map
from chimerax.map import open_map
from os.path import expanduser
models, msg = open_map(session, expanduser('~/Downloads/ChimeraX/EMDB/emd_1080.map'))
session.models.add(models)

# Compute surfaces
for v in models:
    print ('opened map', v, 'size', v.data.size)
    v.update_drawings()		# Compute surface

# Write GLTF file of map surface
from chimerax.gltf import write_gltf
write_gltf(session, 'test.glb')

# Save an image
from chimerax.image_formats import save_image
save_image(session, 'test.png')
