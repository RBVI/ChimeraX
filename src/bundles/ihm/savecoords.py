from chimerax.core.atomic import Structure
for m in session.models.list(type = Structure):
    if m.num_coord_sets > 1:
        from chimerax.ihm import coordsets
        coordsets.write_coordinate_sets(m.filename + '.crd', m)
