def applyPlan(plan, r, cs, f, atomMap, xf0, xf1):
        "Apply a residue interpolation plan."
        # First we compute the starting and ending coordinates
        c0map = {}
        c1map = {}
        for a0 in r.atoms:
                c0map[a0] = xf0 * a0.coord
                c1map[a0] = xf1 * atomMap[a0].coord

        # Now apply the steps of the plan
        for step in plan:
                func, args = step
                func(c0map, c1map, f, cs, *args)

def interpCartesian(c0map, c1map, f, cs, a0):
        "Interpolate Cartesian position between start and end position."
        try:
                c0 = c0map[a0]
                c1 = c1map[a0]
        except KeyError:
                print ('a0', a0)
                for a in c0map.iterkeys():
                        print (" ", 'a', a)
                print (c0map)
                print (c1map)
                raise
        c = c0 + (c1 - c0) * f
        cs[a0.coord_index,:] = c

def interpInternal(c0map, c1map, f, cs, a0, a1, a2, a3):
        """Computer coordinate of atom a0 by interpolating dihedral angle
        defined by atoms (a0, a1, a2, a3)"""
        from chimerax.core.geometry import distance, angle, dihedral, dihedral_point
        c00 = c0map[a0]
        c01 = c0map[a1]
        c02 = c0map[a2]
        c03 = c0map[a3]
        length0 = distance(c00, c01)
        angle0 = angle(c00, c01, c02)
        dihed0 = dihedral(c00, c01, c02, c03)
        c10 = c1map[a0]
        c11 = c1map[a1]
        c12 = c1map[a2]
        c13 = c1map[a3]
        length1 = distance(c10, c11)
        angle1 = angle(c10, c11, c12)
        dihed1 = dihedral(c10, c11, c12, c13)
        length = length0 + (length1 - length0) * f
        angle = angle0 + (angle1 - angle0) * f
        ddihed = dihed1 - dihed0
        if ddihed > 180:
                ddihed -= 360
        elif ddihed < -180:
                ddihed += 360
        dihed = dihed0 + ddihed * f
        c1 = cs[a1.coord_index,:]
        c2 = cs[a2.coord_index,:]
        c3 = cs[a3.coord_index,:]
        try:
                c0 = dihedral_point(c1, c2, c3, length, angle, dihed)
        except ValueError:
                print ('a1', a1)
                print ('a2', a2)
                print ('a3', a3)
                raise
        cs[a0.coord_index,:] = c0

def planCartesian(r):
        "Create a plan for Cartesian interpolation for all atoms."
        plan = []
        for a0 in r.atoms:
                plan.append((interpCartesian, (a0,)))
        return plan

def planInternal(r):
        """Create a plan for dihedral interpolation when possible,
        and Cartesian interpolation otherwise."""
        # First find the atoms that are connected to preceding
        # or succeeding residues.  If none, pick an arbitrary atom.
        # These atoms are always interpolated in Cartesian space.
        plan = []
        done = set()
        todo = []
        fixed = set()
        neighbors = set()
        c = r.chain
        if c:
                for rn in (c.residue_before(r), c.residue_after(r)):
                        if rn:
                                neighbors.add(rn)
        if neighbors:
                for a0 in r.atoms:
                        for na in a0.neighbors:
                                if na.residue in neighbors:
                                        fixed.add(a0)
                                        break
        if not fixed:
                fixed.add(r.atoms[0])
        for a0 in fixed:
                plan.append((interpCartesian, (a0,)))
                _finished(a0, done, todo)

        # Now we look for atoms that are connected to those in
        # "fixed".  If we can find three atoms that define a
        # dihedral, we use dihedral interpolation; otherwise
        # we use Cartesian interpolation.
        while todo:
                na, a = todo.pop(0)
                if na in done:
                        # May be part of a loop and have been
                        # visited via another path
                        continue
                anchors = _findAnchor(a, done)
                if len(anchors) >= 2:
                        # Found two anchor atoms connected to the
                        # fixed atom, we can use them for defining
                        # the dihedral
                        plan.append((interpInternal,
                                        (na, a, anchors[0], anchors[1])))
                        _finished(na, done, todo)
                        continue
                if len(anchors) == 1:
                        # Found one anchor atom connected to the
                        # fixed atom, so we need to get another
                        # anchor atom connected to the one we found
                        # (but is not our original fixed atom)
                        anchors2 = _findAnchor(anchors[0], done, a)
                        if len(anchors2) >= 1:
                                plan.append((interpInternal,
                                        (na, a, anchors[0], anchors2[0])))
                                _finished(na, done, todo)
                                continue
                # Cannot find three fixed atoms to define dihedral.
                # Use Cartesian interpolation for this atom.
                plan.append((interpCartesian, (na,)))
                _finished(na, done, todo)

        # Any left over atoms (usually disconnected) uses Cartesian interp
        for a0 in r.atoms:
                if a0 not in done:
                        plan.append((interpCartesian, (a0,)))
        return plan

def _finished(a, done, todo):
        done.add(a)
        for na in a.neighbors:
                if na not in done and na.residue is a.residue:
                        todo.append((na, a))

def _findAnchor(a, done, ignore=None):
        anchors = []
        for na in a.neighbors:
                if na in done and na is not ignore:
                        anchors.append(na)
        return anchors
