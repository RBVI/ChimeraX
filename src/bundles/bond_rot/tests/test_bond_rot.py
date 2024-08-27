def test_bond_rot(test_production_session):
    from chimerax.core.commands import run
    session = test_production_session
    run(session, "open 1gcn")
    dist_atoms = []
    for r in session.models[0].residues:
        if r.number == 17:
            r17 = r
            nh1 = r.find_atom('NH1')
            dist_atoms.append(nh1)
        elif r.number == 24:
            oe1 = r.find_atom('OE1')
            from chimerax.geometry import distance
            d = distance(nh1.coord, oe1.coord)
            assert(not (d < 9.87 or d > 9.88)), "Distance between %s and %s not initially 9.871!" % (nh1, oe1)
            run(session, "torsion :17@nh1,cd,cg,cb 50")
            d = distance(nh1.coord, oe1.coord)
            assert(not (d < 9.21 or d > 9.23)), "Distance between %s and %s after torsion change not 9.220!" % (nh1, oe1)

            run(session, "setattr :17 r phi -120")
            assert(not (r17.phi < -120.1 or r17.phi > -119.9)), "Setting phi to -120 didn't work! (%g instead)" % r17.phi
