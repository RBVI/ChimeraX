"""Chain and residue selection, undo, and reset for helix inside colors."""


def test_helix_inside_color(test_production_session):
    from chimerax.atomic import AtomicStructure
    from chimerax.core.commands import run

    session = test_production_session
    models = [AtomicStructure(session, name='first'), AtomicStructure(session, name='second')]
    for model in models:
        for chain in ('A', 'B'):
            for i in range(3):
                residue = model.new_residue('ALA', chain, i + 1)
                residue.add_atom(model.new_atom('CA', 'C'))
    session.models.add(models)
    first, second = models
    a1, a2, a3, b1, b2, b3 = first.residues
    red, blue = (255, 0, 0, 255), (0, 0, 255, 255)
    cross_sections = first.ribbon_xs_mgr.take_snapshot(session, 0)
    try:
        run(session, 'cartoon helixInsideColor red %s/A' % first.atomspec)
        assert first.residues.helix_inside_colors.tolist() == [red] * 3 + [None] * 3
        assert second.residues.helix_inside_colors.tolist() == [None] * 6
        run(session, 'cartoon helixInsideColor blue %s/B' % first.atomspec)
        assert first.residues.helix_inside_colors.tolist() == [red] * 3 + [blue] * 3
        # Residue selections work within an independently colored chain.
        run(session, 'cartoon helixInsideColor blue %s/A:2' % first.atomspec)
        assert [a1.helix_inside_color, a2.helix_inside_color, a3.helix_inside_color] == [red, blue, red]
        run(session, 'cartoon helixInsideColor default %s/A' % first.atomspec)
        assert first.residues.helix_inside_colors.tolist() == [None] * 3 + [blue] * 3
        session.undo.undo()
        assert first.residues.helix_inside_colors.tolist() == [red, blue, red] + [blue] * 3
        session.undo.undo()
        assert first.residues.helix_inside_colors.tolist() == [red] * 3 + [blue] * 3
        session.undo.redo()
        assert a2.helix_inside_color == blue
        # Undo skips deleted residues and still restores the rest of the chain.
        run(session, 'cartoon helixInsideColor default %s/A' % first.atomspec)
        a2.delete()
        session.undo.undo()
        assert a1.helix_inside_color == a3.helix_inside_color == red
        session.undo.redo()
        assert a1.helix_inside_color is None and a3.helix_inside_color is None
        assert [b1.helix_inside_color, b2.helix_inside_color, b3.helix_inside_color] == [blue] * 3
        assert first.ribbon_xs_mgr.take_snapshot(session, 0) == cross_sections
        run(session, 'cartoon helixInsideColor default')
        assert first.residues.helix_inside_colors.tolist() == [None] * 5
        assert second.residues.helix_inside_colors.tolist() == [None] * 6
    finally:
        session.models.close(models)
