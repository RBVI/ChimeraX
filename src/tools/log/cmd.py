# vi: set expandtab ts=4 sw=4:

from chimera.core import cli


def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from .gui import Log
    running = session.tools.find_by_class(Log)
    if len(running) > 1:
        raise RuntimeError("too many log instances running")
    if not running:
        if create:
            tool_info = session.toolshed.find_tool('log')
            return Log(session, tool_info)
        else:
            return None
    else:
        return running[0]


def hide(session):
    log = get_singleton(session)
    if log is not None:
        log.display(False)
hide_desc = cli.CmdDesc()


def show(session):
    log = get_singleton(session, create=True)
    if log is not None:
        log.display(True)
show_desc = cli.CmdDesc()


def test(session):
    session.logger.info("Something in <i>italics</i>!", is_html=True)
    session.logger.error("HTML <i>error</i> message", is_html=True)
    #session.logger.error("\n".join(["%d" % i for i in range(200)]))
    #session.logger.warning("Plain text warning")
    from PIL import Image
    session.logger.info(
        "axes", image=Image.open("/Users/pett/Documents/axes.png"))
    session.logger.info("Text after the image\nSecond line")
    session.logger.status("Status test", follow_with="follow text", follow_time=5)
    session.logger.status("Secondary text", blank_after=20, secondary=True)
    structures = [model for model in session.models.list()
                  if model.__class__.__name__ == "StructureModel"]
    if len(structures) == 2:
        f = open("/Users/pett/rm/diff.txt", "w")
        import io
        log_string = io.StringIO("")
        s1 = structures[0]
        s2 = structures[1]
        s1_ab = s1.atoms
        s2_ab = s2.atoms
        s1_id = structures[0].id
        s2_id = structures[1].id
        s1_rb = s1_ab.residues
        s2_rb = s2_ab.residues
        s1_bb = s1.bonds
        s2_bb = s2.bonds
        print("# chains in model {}: {}".format(s1_id, s1.num_chains), file=log_string)
        print("# chains in model {}: {}".format(s1_id, s1.num_chains), file=f)
        print("# chains in model {}: {}".format(s2_id, s2.num_chains), file=log_string)
        print("# chains in model {}: {}".format(s2_id, s2.num_chains), file=f)
        s1_set = set(zip(s1_rb.strs, s1_ab.names))
        print("{} residues in model {} and {} in model {}".format(
            len(s1.residues), s1_id,
            len(s2.residues), s2_id), file=f)
        print("{} residues in model {} and {} in model {}".format(
            len(s1.residues), s1_id,
            len(s2.residues), s2_id), file=log_string)
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=log_string)
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=f)
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=log_string)
        print("# atoms in model {}: {}".format(s2_id, len(s2_ab)), file=f)
        print("# atoms in model {}: {}".format(s2_id, len(s2_ab)), file=log_string)
        print("# unique IDs in model {}: {}".format(s1_id, len(s1_set)), file=f)
        print("# unique IDs in model {}: {}".format(s1_id, len(s1_set)), file=log_string)
        s2_set = set(zip(s2_rb.strs, s2_ab.names))
        print("# unique IDs in model {}: {}".format(s2_id, len(s2_set)), file=f)
        print("# unique IDs in model {}: {}".format(s2_id, len(s2_set)), file=log_string)
        print("In {} model but not {} ({}):".format(s1_id, s2_id, len(s1_set - s2_set)), file=f)
        print("In {} model but not {} ({}):".format(s1_id, s2_id, len(s1_set - s2_set)), file=log_string)
        for rstr, aname in s1_set - s2_set:
            print("\t" + rstr + " " + aname, file=f)
            print("\t" + rstr + " " + aname, file=log_string)
        print("In {} model but not {} ({}):".format(s2_id, s1_id, len(s2_set - s1_set)), file=f)
        print("In {} model but not {} ({}):".format(s2_id, s1_id, len(s2_set - s1_set)), file=log_string)
        for rstr, aname in s2_set - s1_set:
            print("\t" + rstr + " " + aname, file=f)
            print("\t" + rstr + " " + aname, file=log_string)

        print("# bonds in model {}: {}".format(s1_id, len(s1_bb)), file=f)
        print("# bonds in model {}: {}".format(s1_id, len(s1_bb)), file=log_string)
        print("# bonds in model {}: {}".format(s2_id, len(s2_bb)), file=f)
        print("# bonds in model {}: {}".format(s2_id, len(s2_bb)), file=log_string)
        s1_bond_set = set()
        bond_ab1, bond_ab2 = s1_bb.atoms
        for rstr1, aname1, rstr2, aname2 in zip(
                bond_ab1.residues.strs, bond_ab1.names, bond_ab2.residues.strs, bond_ab2.names):
            id1 = (rstr1, aname1)
            id2 = (rstr2, aname2)
            if id1 > id2:
                id1, id2 = id2, id1
            s1_bond_set.add("{} {}/{} {}".format(id1[0], id1[1], id2[0], id2[1]))
        s2_bond_set = set()
        bond_ab1, bond_ab2 = s2_bb.atoms
        for rstr1, aname1, rstr2, aname2 in zip(
                bond_ab1.residues.strs, bond_ab1.names, bond_ab2.residues.strs, bond_ab2.names):
            id1 = (rstr1, aname1)
            id2 = (rstr2, aname2)
            if id1 > id2:
                id1, id2 = id2, id1
            s2_bond_set.add("{} {}/{} {}".format(id1[0], id1[1], id2[0], id2[1]))
        print("Bonds in {} model but not {} ({}):".format(s1_id, s2_id, len(s1_bond_set - s2_bond_set)), file=f)
        print("Bonds in {} model but not {} ({}):".format(s1_id, s2_id, len(s1_bond_set - s2_bond_set)), file=log_string)
        for bond_info in s1_bond_set - s2_bond_set:
            print("\t" + bond_info, file=f)
            print("\t" + bond_info, file=log_string)
        print("Bonds in {} model but not {} ({}):".format(s2_id, s1_id, len(s2_bond_set - s1_bond_set)), file=f)
        print("Bonds in {} model but not {} ({}):".format(s2_id, s1_id, len(s2_bond_set - s1_bond_set)), file=log_string)
        for bond_info in s2_bond_set - s1_bond_set:
            print("\t" + bond_info, file=f)
            print("\t" + bond_info, file=log_string)
        pb_map1 = s1.pseudobond_groups
        pb_map2 = s2.pseudobond_groups
        for name, pblob in pb_map1.items():
            if name in pb_map2:
                if len(pblob) == len(pb_map2[name]):
                    print("{} has {} bonds in both".format(name, len(pblob)), file=f)
                    print("{} has {} bonds in both".format(name, len(pblob)), file=log_string)
                    print("In {}:".format(s1_id), file=f)
                    for i1, i2 in pblob.bond_indices:
                        print("\t{}@{} <-> {}@{}".format(s1_rb.strs[i1], s1_ab.names[i1], s1_rb.strs[i2], s1_ab.names[i2]), file=f)
                    print("In {}:".format(s2_id), file=f)
                    for i1, i2 in pb_map2[name].bond_indices:
                        print("\t{}@{} <-> {}@{}".format(s2_rb.strs[i1], s2_ab.names[i1], s2_rb.strs[i2], s2_ab.names[i2]), file=f)
                    continue
                print("{} has {} bonds in {} model but {} in {}".format(name, len(pblob), s1_id, len(pb_map2[name]), s2_id), file=f)
                print("{} has {} bonds in {} model but {} in {}".format(name, len(pblob), s1_id, len(pb_map2[name]), s2_id), file=log_string)
                pb1_indices = set()
                for i1, i2 in pblob.bond_indices:
                    if i1 < i2:
                        pb1_indices.add((i1, i2))
                    else:
                        pb1_indices.add((i2, i1))
                pb2_indices = set()
                for i1, i2 in pb_map2[name].bond_indices:
                    if i1 < i2:
                        probe = (i1, i2)
                    else:
                        probe = (i2, i1)
                    pb2_indices.add(probe)
                    if probe not in pb1_indices:
                        print("\t{}@{} <-> {}@{} in {} but not in {}".format(s2_rb.strs[i1], s2_ab.names[i1], s2_rb.strs[i2], s2_ab.names[i2], s2_id, s1_id), file=f)
                for i1, i2 in pblob.bond_indices:
                    if i1 < i2:
                        probe = (i1, i2)
                    else:
                        probe = (i2, i1)
                    if probe not in pb2_indices:
                        print("\t{}@{} <-> {}@{} in {} but not in {}".format(s1_rb.strs[i1], s1_ab.names[i1], s1_rb.strs[i2], s1_ab.names[i2], s1_id, s2_id), file=f)
            else:
                print("{} in {} model but not in {}".format(
                    name, s1_id, s2_id), file=f)
                print("{} in {} model but not in {}".format(
                    name, s1_id, s2_id), file=log_string)
        for name, pblob in pb_map2.items():
            if name not in pb_map1:
                print("{} in {} model but not in {}".format(
                    name, s2_id, s1_id), file=f)
                print("{} in {} model but not in {}".format(
                    name, s2_id, s1_id), file=log_string)
        f.close()
        session.logger.info(log_string.getvalue())
test_desc = cli.CmdDesc()
