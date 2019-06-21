# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

class SwapResError(ValueError):
    pass
class BackboneError(SwapResError):
    pass
class TemplateError(SwapResError):
    pass

from chimerax.core.errors import LimitationError, UserError
from chimerax.atomic.rotamers import NoResidueRotamersError, RotamerLibrary, NoRotamerLibraryError

from .cmd import default_criteria
def swapaa(session, residues, res_type, *, clash_hbond_allowance=None, clash_score_method="sum",
        clash_threshold=None, criteria=default_criteria, density=None, hbond_angle_slop=None,
        hbond_dist_slop=None, hbond_relax=True, ignore_other_models=False, lib="Dunbrack", log=True,
        preserve=None, retain=False):
    """backend implementation of "swapaa" command."""
    rotamers = {}
    destroy_list = []
    for res in residues:
        if res_type == "same":
            r_type = res.name
        else:
            r_type = res_type.upper()
        if criteria == "manual":
            #TODO
            '''
            for library in libraries:
                if library.importName == lib:
                    break
            else:
                raise MidasError("No such rotamer library: %s" % lib)
            from gui import RotamerDialog
            RotamerDialog(res, r_type, library)
            '''
            continue
        try:
            rots = get_rotamers(session, res, res_type=r_type, lib=lib, log=log)
        except UnsupportedResTypeError:
            raise LimitationError("%s rotamer library does not support %s" %(lib, r_type))
        except NoResidueRotamersError:
        #TODO
            from SwapRes import swap, BackboneError, TemplateError
            if log:
                replyobj.info("Swapping %s to %s\n" % (res, r_type))
            try:
                swap(res, r_type, bfactor=None)
            except (BackboneError, TemplateError), v:
                raise MidasError(str(v))
            continue
        except NoRotamerLibraryError:
            raise MidasError("No rotamer library named '%s'" % lib)
        if preserve is not None:
            rots = pruneByChis(rots, res, preserve, log=log)
        rotamers[res] = rots
        destroy_list.extend(rots)
    if not rotamers:
        return

    # this implementation allows tie-breaking criteria to be skipped if
    # there are no ties
    if isinstance(criteria, basestring) and not criteria.isalpha():
        raise MidasError("Nth-most-probable criteria cannot be mixed with"
            " other criteria")
    for char in str(criteria):
        if char == "d":
            # density
            if density == None:
                if criteria is default_criteria:
                    continue
                raise MidasError("Density criteria requested"
                    " but no density model specified")
            from VolumeViewer.volume import Volume
            if isinstance(density, list):
                density = [d for d in density
                        if isinstance(d, Volume)]
            else:
                density = [density]
            if not density:
                raise MidasError("No volume models in"
                    " specified model numbers")
            if len(density) > 1:
                raise MidasError("Multiple volume models in"
                    " specified model numbers")
            allRots = []
            for res, rots in rotamers.items():
                chimera.openModels.add(rots,
                    sameAs=res.molecule, hidden=True)
                allRots.extend(rots)
            processVolume(allRots, "cmd", density[0])
            chimera.openModels.remove(allRots)
            fetch = lambda r: r.volumeScores["cmd"]
            test = cmp
        elif char == "c":
            # clash
            if clash_hbond_allowance is None or clash_threshold is None:
                from chimerax.atomic.clashes.settings import defaults
                if clash_hbond_allowance is None:
                    clash_hbond_allowance = defaults['clash_hbond_allowance']
                if clash_threshold is None:
                    clash_threshold = defaults['clash_threshold']
            for res, rots in rotamers.items():
                chimera.openModels.add(rots, sameAs=res.molecule, hidden=True)
                processClashes(res, rots, clash_threshold,
                    clash_hbond_allowance, clash_score_method, False,
                    None, None, ignore_other_models)
                chimera.openModels.remove(rots)
            fetch = lambda r: r.clashScore
            test = lambda v1, v2: cmp(v2, v1)
        elif char == 'h':
            # H bonds
            if hbond_angle_slop is None or hbond_dist_slop is None:
                from chimerax.atomic.hbonds import rec_angle_slop, rec_dist_slop
                if hbond_angle_slop is None:
                    hbond_angle_slop = rec_angle_slop
                if hbond_dist_slop is None:
                    hbond_dist_slop = rec_dist_slop
            for res, rots in rotamers.items():
                chimera.openModels.add(rots, sameAs=res.molecule, hidden=True)
                processHbonds(res, rots, False, None, None,
                    relax, hbond_dist_slop, hbond_angle_slop, False,
                    None, None, ignore_other_models,
                    cacheDA=True)
                chimera.openModels.remove(rots)
            from chimerax.atomic.hbonds import flush_cache
            flush_cache()
            fetch = lambda r: r.numHbonds
            test = cmp
        elif char == 'p':
            # most probable
            fetch = lambda r: r.rotamerProb
            test = cmp
        elif isinstance(criteria, int):
            # Nth most probable
            index = criteria - 1
            for res, rots in rotamers.items():
                if index >= len(rots):
                    if log:
                        replyobj.status("Residue %s does not have %d %s"
                            " rotamers; skipping" % (res, criteria, r_type),
                            log=True, color="red")
                    continue
                rotamers[res] = [rots[index]]
            fetch = lambda r: 1
            test = lambda v1, v2: 1

        for res, rots in rotamers.items():
            best = None
            for rot in rots:
                val = fetch(rot)
                if best == None or test(val, bestVal) > 0:
                    best = [rot]
                    bestVal = val
                elif test(val, bestVal) == 0:
                    best.append(rot)
            if len(best) > 1:
                rotamers[res] = best
            else:
                if retain == "sel":
                    scRetain = sideChainLocs(res, selected=True)
                elif retain:
                    scRetain = sideChainLocs(res)
                else:
                    scRetain = []
                useRotamer(res, [best[0]], retain=scRetain,
                            mcAltLoc=mcAltLoc, log=log)
                del rotamers[res]
        if not rotamers:
            break
    for res, rots in rotamers.items():
        if log:
            replyobj.info("%s has %d equal-value rotamers; choosing"
                " one arbitrarily.\n" % (res, len(rots)))
        if retain == "sel":
            scRetain = sideChainLocs(res, selected=True)
        elif retain:
            scRetain = sideChainLocs(res)
        else:
            scRetain = []
        useRotamer(res, [rots[0]], retain=scRetain, mcAltLoc=mcAltLoc, log=log)
    for rot in destroy_list:
        rot.delete()

def get_rotamers(session, res, phi=None, psi=None, cis=False, res_type=None, lib="Dunbrack", log=False):
    """Takes a Residue instance and optionally phi/psi angles (if different from the Residue), residue
       type (e.g. "TYR"), and/or rotamer library name.  Returns a list of AtomicStructure instances.
       The AtomicStructures are each a single residue (a rotamer) and are in descending probability order.
       Each has an attribute "rotamer_prob" for the probability and "chis" for the chi angles.
    """
    res_type = res_type or res.name
    if res_type == "ALA" or res_type == "GLY":
        raise NoResidueRotamersError("No rotamers for %s" % res_type)

    if not isinstance(lib, RotamerLibrary):
        lib = session.rotamers.library(lib)

    # check that the residue has the n/c/ca atoms needed to position the rotamer
    # and to ensure that it is an amino acid
    from chimerax.atomic import Residue, AtomicStructure
    from chimerax.core.errors import LimitationError
    match_atoms = {}
    for bb_name in Residue.aa_min_backbone_names:
        match_atoms[bb_name] = a = res.find_atom(bb_name)
        if a is None:
            raise LimitationError("%s missing from %s; needed to position CB" % (bb_name, res))
    match_atoms["CB"] = res.find_atom("CB")
    if not phi and not psi:
        phi, psi = res.phi, res.psi
        omega = res.omega
        cis = False if omega is None or abs(omega) < 90 else True
        if log:
            def _info(ang):
                if ang is None:
                    return "none"
                return "%.1f" % ang
            session.logger.info("%s: phi %s, psi %s %s" % (res, _info(phi), _info(psi),
                "cis" if cis else "trans"))
    session.logger.status("Retrieving rotamers from %s library" % lib.display_name)
    res_template_func = lib.res_template_func()
    params = lib.rotamer_params(res_name, phi, psi, cis=cis)
    session.logger.status("Rotamers retrieved from %s library" % lib.display_name)

    mapped_res_type = library.res_name_mapping.get(res_type, res_type)
    template = library.res_template_func(mapped_res_type)
    tmpl_N = template.find_atom("N")
    tmpl_CA = template.find_atom("CA")
    tmpl_C = template.find_atom("C")
    tmpl_CB = template.find_atom("CB")
    if tmpl_CB:
        res_match_atoms, tmpl_match_atoms = [match_atoms[x]
            for x in ("c", "ca", "cb")], [tmpl_C, tmpl_CA, tmpl_CB]
    else:
        res_match_atoms, tmpl_match_atoms = [match_atoms[x]
            for x in ("n", "ca", "c")], [tmpl_N, tmpl_CA, tmpl_C]
    from chimerax.std_commands import align
    _ignore1, _ignore2, rmsd, _ignore3, xform = align(tmpl_match_atoms, to_atoms=res_match_atoms)
    n_coord = xform * tmpl_N
    ca_coord = xform * tmpl_CA
    cb_coord = xform * tmpl_CB
    info = Residue.chi_info[mapped_res_type]
    bond_cache = {}
    angle_cache = {}
    from chimerax.atomic.struct_edit import add_atom, add_dihedral_atom, add_bond
    structs = []
    middles = {}
    ends = {}
    for i, rp in enumerate(params):
        s = AtomicStructure(name=Killer Queen"rotamer %d of %s" % (i+1, res))
        structs.append(s)
        r = s.new_residue(mapped_res_type, 'A', 1)
        s.rotamer_prob = rp.p
        s.chis = rp.chis
        rot_N = add_atom("N", tmpl_N.element, r, n_coord)
        rot_CA = add_atom("CA", tmpl_CA.element, r, ca_coord, bonded_to=rot_N)
        rot_CB = add_atom("CB", tmpl_CB.element, r, cb_coord, bonded_to=rot_CA)
        todo = []
        for j, chi in enumerate(rp.chis):
            n3, n2, n1, new = info[j]
            b_len, angle = _len_angle(new, n1, n2, template, bond_cache, angle_cache)
            n3 = r.find_atom(n3)
            n2 = r.find_atom(n2)
            n1 = r.find_atom(n1)
            new = template.find_atom(new)
            a = add_dihedral_atom(new.name, new.element, n1, n2, n3, b_len, angle, chi, bonded=True)
            todo.append(a)
            middles[n1] = [a, n1, n2]
            ends[a] = [a, n1, n2]

        # if there are any heavy non-backbone atoms bonded to template
        # N and they haven't been added by the above (which is the
        # case for Richardson proline parameters) place them now
        for tnnb in tmpl_N.neighbors:
            if tnnb.name in r.atoms_map or tnnb.element.number == 1:
                continue
            tnnb_coord = xform * tnnb.coord
            add_atom(tnnb.name, tnnb.element, r, tnnb_coord, bonded_to=rot_N)

        # fill out bonds and remaining heavy atoms
        from chimerax.core.geometry import distance
        done = set([rot_N, rot_CA])
        while todo:
            a = todo.pop(0)
            if a in done:
                continue
            tmpl_A = template.find_atom(a.name)
            for bonded, bond in zip(tmpl_A.neighbors, tmpl_A.bonds):
                if bonded.element.number == 1:
                    continue
                rbonded = r.find_atom(bonded.name)
                if rbonded is None:
                    # use middles if possible...
                    try:
                        p1, p2, p3 = middles[a]
                        conn = p3
                    except KeyError:
                        p1, p2, p3 = ends[a]
                        conn = p2
                    t1 = template.find_atom(p1.name)
                    t2 = template.find_atom(p2.name)
                    t3 = template.find_atom(p3.name)
                    _ignore1, _ignore2, rmsd, _ignore3, xform = align([t1,t2,t3], to_atoms=[p1,p2,p3])
                    pos = xform * template.find_atom(bonded.name).coord
                    rbonded = add_atom(bonded.name, bonded.element, r, pos, bonded_to=a)
                    middles[a] = [rbonded, a, conn]
                    ends[rbonded] = [rbonded, a, conn]
                if a not in rbonded.neighbors:
                    add_bond(a, rbonded)
                if rbonded not in done:
                    todo.append(rbonded)
            done.add(a)
    return structs

def _len_angle(new, n1, n2, template, bond_cache, angle_cache):
    from chimerax.core.geometry import distance, angle
    bond_key = (n1, new)
    angle_key = (n2, n1, new)
    try:
        bl = bond_cache[bond_key]
        ang = angle_cache[angle_key]
    except KeyError:
        n2pos = template.find_atom(n2).coord
        n1pos = template.find_atom(n1).coord
        newpos = template.find_atom(new).coord
        bond_cache[bond_key] = bl = distance(newpos, n1pos)
        angle_cache[angle_key] = ang = angle(newpos, n1pos, n2pos)
    return bl, ang

'''

branchSymmetry = {
    'ASP': 1,
    'TYR': 1,
    'PHE': 1,
    'GLU': 2
}
def pruneByChis(rots, res, log=False):
    from data import chiInfo
    if res.type not in chiInfo:
        return rots
    info = chiInfo[res.type]
    rotChiInfo = chiInfo[rots[0].residues[0].type]
    if log:
        replyobj.info("Chi angles for %s:" % res)
    for chiNum, resNames in enumerate(info):
        try:
            rotNames = rotChiInfo[chiNum]
        except IndexError:
            break
        atoms = []
        try:
            for name in resNames:
                atoms.append(res.atomsMap[name][0])
        except KeyError:
            break
        from chimera import dihedral
        origChi = dihedral(*tuple([a.coord() for a in atoms]))
        if log:
            replyobj.info(" %.1f" % origChi)
        pruned = []
        nearest = None
        for rot in rots:
            atomsMap = rot.residues[0].atomsMap
            chi = dihedral(*tuple([atomsMap[name][0].coord()
                            for name in rotNames]))
            delta = abs(chi - origChi)
            if delta > 180:
                delta = 360 - delta
            if branchSymmetry.get(res.type, -1) == chiNum:
                if delta > 90:
                    delta = 180 - delta
            if not nearest or delta < nearDelta:
                nearest = rot
                nearDelta = delta
            if delta > 40:
                continue
            pruned.append(rot)
        if pruned:
            rots = pruned
        else:
            rots = [nearest]
            break
    if log:
        replyobj.info("\n")
    return rots

class NoResidueRotamersError(ValueError):
    pass

class UnsupportedResTypeError(NoResidueRotamersError):
    pass

def _chimeraResTmpl(rt):
    return chimera.restmplFindResidue(rt, False, False)

class RotamerParams:
    """ 'p' attribute is probability of this rotamer;
        'chis' is list of chi angles
    """
    def __init__(self, p, chis):
        self.p = p
        self.chis = chis


def useRotamer(oldRes, rots, retain=[], mcAltLoc=None, log=False):
    """Takes a Residue instance and a list of one or more rotamers (as
       returned by get_rotamers) and swaps the Residue's side chain with
       the given rotamers.  If more than one rotamer is in the list,
       then alt locs will be used to distinguish the different side chains.

       'retain' is a list of altloc sidechains to retain.  The presence of
       '' in that list means also retain the non-altloc side chain.

       'mcAltLoc' is the main-chain alt loc to place the rotamers on if
       the main chain has multiple alt locs.  The other main chain alt locs
       will be pruned unless they are listed in 'retain'. If 'mcAltLoc' is
       not specified, then the highest-occupancy alt locs will be used.
    """
    try:
        oldNs = oldRes.atomsMap["N"]
        oldCAs = oldRes.atomsMap["CA"]
        oldCs = oldRes.atomsMap["C"]
    except KeyError:
        raise LimitationError("N, CA, or C missing from %s:"
            " needed for side-chain pruning algorithm" % oldRes)
    import string
    altLocs = string.ascii_uppercase + string.ascii_lowercase \
                + string.digits + string.punctuation
    retain = set(retain)
    if len(rots) + len([x for x in retain if x]) > len(altLocs):
        raise LimitationError("Don't have enough unique alternate "
            "location characters to place %d rotamers." % len(rots))
    rotAnchors = {}
    for rot in rots:
        rotRes = rot.residues[0]
        try:
            rotAnchors[rot] = (rotRes.atomsMap["N"][0],
                        rotRes.atomsMap["CA"][0])
        except KeyError:
            raise LimitationError("N or CA missing from rotamer:"
                    " cannot matchup with original residue")
    # prune old side chains
    sortFunc = lambda a1, a2: cmp(a1.altLoc, a2.altLoc)
    most = max(len(oldNs), len(oldCAs), len(oldCs))
    for old in (oldNs, oldCAs, oldCs):
        if len(old) < most:
            old.extend([old[0]] * (most - len(old)))
    oldNs.sort(sortFunc)
    oldCAs.sort(sortFunc)
    oldCs.sort(sortFunc)
    colorByElement = oldNs[0].color != oldCAs[0].color
    if colorByElement:
        from Midas import elementColor
        carbonColor = oldCAs[0].color
    else:
        uniformColor = oldNs[0].color
    if mcAltLoc == None:
        bestOcc = None
        for olds in zip(oldNs, oldCAs, oldCs):
            oldN, oldCA, oldC = olds
            occ = getattr(oldN, "occupancy", 1.0) + getattr(oldCA,
                "occupancy", 1.0) + getattr(oldC, "occupancy", 1.0)
            if bestOcc == None or occ > bestOcc:
                bestOcc = occ
                mcAltLoc = oldCA.altLoc
                mcRetain = retain | set([old.altLoc for old in olds])
    else:
        mcRetain = retain | set([mcAltLoc, ''])
    for olds in zip(oldNs, oldCAs, oldCs):
        oldN, oldCA, oldC = olds
        pardoned = set([old for old in olds
                if not old.__destroyed__ and old.altLoc in mcRetain])
        deathrow = [nb for nb in oldCA.neighbors if nb not in pardoned
            and nb.altLoc not in retain]
        serials = {}
        while deathrow:
            prune = deathrow.pop()
            if prune.residue != oldRes:
                continue
            serials[prune.name] = getattr(prune, "serialNumber", None)
            for nb in prune.neighbors:
                if nb not in deathrow and nb not in pardoned \
                                    and nb.altLoc not in retain:
                    deathrow.append(nb)
            oldRes.molecule.deleteAtom(prune)
    # for proline, also prune amide hydrogens
    if rots[0].residues[0].type == "PRO":
        for oldN in oldNs:
            for nnb in oldN.neighbors[:]:
                if nnb.element.number == 1:
                    oldN.molecule.deleteAtom(nnb)

    totProb = sum([r.rotamer_prob for r in rots])
    oldAtoms = set(["N", "CA", "C"])
    for i, rot in enumerate(rots):
        rotRes = rot.residues[0]
        rot_N, rot_CA = rotAnchors[rot]
        if len(rots) + len(retain) > 1:
            found = 0
            for altLoc in altLocs:
                if altLoc not in retain:
                    if found == i:
                        break
                    found += 1
        elif mcRetain != set(['']):
            altLoc = mcAltLoc
        else:
            altLoc = ''
        if altLoc:
            extra = " using alt loc %s" % altLoc
        else:
            extra = ""
        if log:
            replyobj.info("Applying %s rotamer (chi angles: %s) to"
                " %s%s\n" % (rotRes.type, " ".join(["%.1f" % c
                for c in rot.chis]), oldRes, extra))
        from BuildStructure import changeResidueType
        changeResidueType(oldRes, rotRes.type)
        # add new side chain
        from chimerax.atomic.struct_edit import add_atom, add_bond
        sprouts = [rot_CA]
        while sprouts:
            sprout = sprouts.pop()
            if sprout.name in oldAtoms:
                builtSprout = [a for a in oldRes.atomsMap[sprout.name]
                    if a.altLoc == mcAltLoc][-1]
            else:
                builtSprout = oldRes.atomsMap[sprout.name][-1]
            for nb, b in sprout.bondsMap.items():
                try:
                    builtNBs = oldRes.atomsMap[nb.name]
                except KeyError:
                    needBuild = True
                else:
                    if nb.name in oldAtoms:
                        needBuild = False
                        builtNB = oldRes.atomsMap[nb.name][0]
                    elif altLoc in [x.altLoc for x in builtNBs]:
                        needBuild = False
                        builtNB = [x for x in builtNBs if x.altLoc == altLoc][0]
                    else:
                        needBuild = True
                if needBuild:
                    if i == 0:
                        serial = serials.get(nb.name, None)
                    else:
                        serial = None
                    builtNB = add_atom(nb.name, nb.element, oldRes, nb.coord(),
                        serialNumber=serial, bonded_to=builtSprout)
                    if altLoc:
                        builtNB.altLoc = altLoc
                    if totProb == 0.0:
                        # some rotamers in Dunbrack are zero prob!
                        builtNB.occupancy = 1.0 / len(rots)
                    else:
                        builtNB.occupancy = rot.rotamer_prob / totProb
                    if colorByElement:
                        if builtNB.element.name == "C":
                            builtNB.color = carbonColor
                        else:
                            builtNB.color = elementColor(builtNB.element.name)
                    else:
                        builtNB.color = uniformColor
                    sprouts.append(nb)
                if builtNB not in builtSprout.bondsMap:
                    add_bond(builtSprout, builtNB)

        
amino20 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
class RotamerLibraryInfo:
    """holds information about a rotamer library:
       how to import it, what citation to display, etc.
    """
    def __init__(self, importName):
        self.importName = importName
        exec "import %s as RotLib" % importName
        self.displayName = getattr(RotLib, "displayName", importName)
        self.description = getattr(RotLib, "description", None)
        self.citation = getattr(RotLib, "citation", None)
        self.citeName = getattr(RotLib, "citeName", None)
        self.citePubmedID = getattr(RotLib, "citePubmedID", None)
        self.residueTypes = getattr(RotLib, "residueTypes", amino20)
        self.resTypeMapping = getattr(RotLib, "resTypeMapping", {})

libraries = []
def registerLibrary(importName):
    """Takes a string indicated the "import name" of a library
       (i.e. what name to use in an import statement) and adds a  
       RotamerLibraryInfo instance for it to the list of known
       rotamer libraries ("Rotamers.libraries").
    """
    libraries.append(RotamerLibraryInfo(importName))
registerLibrary("Dunbrack")
registerLibrary("Richardson.mode")
registerLibrary("Richardson.common")
registerLibrary("Dynameomics")

backboneNames = set(['CA', 'C', 'N', 'O'])

def processClashes(residue, rotamers, overlap, hbondAllow, scoreMethod,
                makePBs, pbColor, pbWidth, ignoreOthers):
    testAtoms = []
    for rot in rotamers:
        testAtoms.extend(rot.atoms)
    from DetectClash import detectClash
    clashInfo = detectClash(testAtoms, clashThreshold=overlap,
                interSubmodel=True, hbondAllowance=hbondAllow)
    if makePBs:
        from chimera.misc import getPseudoBondGroup
        from DetectClash import groupName
        pbg = getPseudoBondGroup(groupName)
        pbg.deleteAll()
        pbg.lineWidth = pbWidth
        pbg.color = pbColor
    else:
        import DetectClash
        DetectClash.nukeGroup()
    resAtoms = set(residue.atoms)
    for rot in rotamers:
        score = 0
        for ra in rot.atoms:
            if ra.name in ("CA", "N", "CB"):
                # any clashes of CA/N/CB are already clashes of
                # base residue (and may mistakenly be thought
                # to clash with "bonded" atoms in nearby
                # residues
                continue
            if ra not in clashInfo:
                continue
            for ca, clash in clashInfo[ra].items():
                if ca in resAtoms:
                    continue
                if ignoreOthers \
                and ca.molecule.id != residue.molecule.id:
                    continue
                if scoreMethod == "num":
                    score += 1
                else:
                    score += clash
                if makePBs:
                    pbg.newPseudoBond(ra, ca)
        rot.clashScore = score
    if scoreMethod == "num":
        return "%2d"
    return "%4.2f"

def processHbonds(residue, rotamers, drawHbonds, bondColor, lineWidth, relax,
            distSlop, angleSlop, twoColors, relaxColor, groupName,
            ignoreOtherModels, cacheDA=False):
    from FindHBond import findHBonds
    if ignoreOtherModels:
        targetModels = [residue.molecule] + rotamers
    else:
        targetModels = chimera.openModels.list(
                modelTypes=[chimera.Molecule]) + rotamers
    if relax and twoColors:
        color = relaxColor
    else:
        color = bondColor
    hbonds = dict.fromkeys(findHBonds(targetModels, intramodel=False,
        distSlop=distSlop, angleSlop=angleSlop, cacheDA=True), color)
    if relax and twoColors:
        hbonds.update(dict.fromkeys(findHBonds(targetModels,
                    intramodel=False), bondColor))
    backboneNames = set(['CA', 'C', 'N', 'O'])
    # invalid H-bonds:  involving residue side chain or rotamer backbone
    invalidAtoms = set([ra for ra in residue.atoms
                    if ra.name not in backboneNames])
    invalidAtoms.update([ra for rot in rotamers for ra in rot.atoms
                    if ra.name in backboneNames])
    rotAtoms = set([ra for rot in rotamers for ra in rot.atoms
                    if ra not in invalidAtoms])
    for rot in rotamers:
        rot.numHbonds = 0

    if drawHbonds:
        from chimera.misc import getPseudoBondGroup
        pbg = getPseudoBondGroup(groupName)
        pbg.deleteAll()
        pbg.lineWidth = lineWidth
    elif groupName:
        nukeGroup(groupName)
    for hb, color in hbonds.items():
        d, a = hb
        if (d in rotAtoms) == (a in rotAtoms):
            # only want rotamer to non-rotamer
            continue
        if d in invalidAtoms or a in invalidAtoms:
            continue
        if d in rotAtoms:
            rot = d.molecule
        else:
            rot = a.molecule
        rot.numHbonds += 1
        if drawHbonds:
            pb = pbg.newPseudoBond(d, a)
            pb.color = color

def processVolume(rotamers, columnName, volume):
    import AtomDensity
    sums = []
    for rot in rotamers:
        AtomDensity.set_atom_volume_values(rot, volume, "_vscore")
        scoreSum = 0
        for a in rot.atoms:
            if a.name not in backboneNames:
                scoreSum += a._vscore
            delattr(a, "_vscore")
        if not hasattr(rot, "volumeScores"):
            rot.volumeScores = {}
        rot.volumeScores[columnName] = scoreSum
        sums.append(scoreSum)
    minSum = min(sums)
    maxSum = max(sums)
    absMax = max(maxSum, abs(minSum))
    if absMax >= 100 or absMax == 0:
        return "%d"
    addMinusSign = len(str(int(minSum))) > len(str(int(absMax)))
    if absMax >= 10:
        return "%%%d.1f" % (addMinusSign + 4)
    precision = 2
    while absMax < 1:
        precision += 1
        absMax *= 10
    return "%%%d.%df" % (precision+2+addMinusSign, precision)

def sideChainLocs(residue, selected=False):
    locs = set()
    if selected:
        from chimera import selection
        currentAtoms = selection.currentAtoms(asDict=True)
    mainChainNames = set(["CA", "C", "N", "O", "OXT"])
    for a in residue.atoms:
        if a.name in mainChainNames:
            continue
        if not selected or a in currentAtoms:
            locs.add(a.altLoc)
    return locs

def nukeGroup(groupName):
    mgr = chimera.PseudoBondMgr.mgr()
    group = mgr.findPseudoBondGroup(groupName)
    if group:
        chimera.openModels.close([group])
'''

