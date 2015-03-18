# vi: set expandtab ts=4 sw=4:

from chimera.core import cli

def _get_gui(session, create=False):
    from .gui import Log
    running = session.tools.find_by_class(Log)
    if len(running) > 1:
        raise RuntimeError("too many log instances running")
    if not running:
        if create:
            return Log(session)
        else:
            return None
    else:
        return running[0]

def hide(session):
    log = _get_gui(session)
    if log is not None:
        log.display(False)
hide_desc = cli.CmdDesc()

def show(session):
    log = _get_gui(session, create=True)
    if log is not None:
        log.display(True)
show_desc = cli.CmdDesc()

def test(session):
    session.logger.info("Something in <i>italics</i>!", is_html=True)
    #session.logger.error("HTML <i>error</i> message", is_html=True)
    #session.logger.warning("Plain text warning")
    from PIL import Image
    session.logger.info("axes",
        image=Image.open("/Users/pett/Documents/axes.png"))
    session.logger.info("Text after the image\nSecond line")
    session.logger.status("Status test", follow_with="follow text", follow_time=5)
    session.logger.status("Secondary text", blank_after=20, secondary=True)
    res_types = set()
    structures = [model for model in session.models.list()
        if model.__class__.__name__ == "StructureModel"]
    if len(structures) == 2:
        f = open("/Users/pett/rm/3zvf_diff.txt", "w")
        s1_ab = structures[0].mol_blob.atoms
        s1_id = structures[0].id
        s2_id = structures[1].id
        import io
        log_string = io.StringIO("")
        print("{} residues in model {} and {} in model {}".format(
            len(structures[0].mol_blob.residues), s1_id,
            len(structures[1].mol_blob.residues), s2_id), file=f)
        print("{} residues in model {} and {} in model {}".format(
            len(structures[0].mol_blob.residues), s1_id,
            len(structures[1].mol_blob.residues), s2_id), file=log_string)
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=log_string)
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=f)
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=log_string)
        s1_rb = s1_ab.residues
        s2_ab = structures[1].mol_blob.atoms
        print("# atoms in model {}: {}".format(s2_id, len(s2_ab)), file=f)
        print("# atoms in model {}: {}".format(s2_id, len(s2_ab)), file=log_string)
        s2_rb = s2_ab.residues
        s1_set = set(zip(s1_rb.strs, s1_ab.names))
        print("# unique IDs in model {}: {}".format(s1_id, len(s1_set)), file=f)
        print("# unique IDs in model {}: {}".format(s1_id, len(s1_set)), file=log_string)
        s2_set = set(zip(s2_rb.strs, s2_ab.names))
        print("# unique IDs in model {}: {}".format(s2_id, len(s2_set)), file=f)
        print("# unique IDs in model {}: {}".format(s2_id, len(s2_set)), file=log_string)
        print("In {} model but not {} ({}):".format(s1_id, s2_id, len(s1_set - s2_set)), file=f)
        print("In {} model but not {} ({}):".format(s1_id, s2_id, len(s1_set - s2_set)), file=log_string)
        for rstr, aname in  s1_set - s2_set:
            print("\t" + rstr + " " + aname, file=f)
            print("\t" + rstr + " " + aname, file=log_string)
        print("In {} model but not {} ({}):".format(s2_id, s1_id, len(s2_set - s1_set)), file=f)
        print("In {} model but not {} ({}):".format(s2_id, s1_id, len(s2_set - s1_set)), file=log_string)
        for rstr, aname in  s2_set - s1_set:
            print("\t" + rstr + " " + aname, file=f)
            print("\t" + rstr + " " + aname, file=log_string)

        s1_bb = structures[0].mol_blob.bonds
        print("# bonds in model {}: {}".format(s1_id, len(s1_bb)), file=f)
        print("# bonds in model {}: {}".format(s1_id, len(s1_bb)), file=log_string)
        s2_bb = structures[1].mol_blob.bonds
        print("# bonds in model {}: {}".format(s2_id, len(s2_bb)), file=f)
        print("# bonds in model {}: {}".format(s2_id, len(s2_bb)), file=log_string)
        s1_atom_info = list(zip(s1_rb.strs, s1_ab.names))
        s1_bond_set = set()
        for i1, i2 in structures[0].mol_blob.bond_indices:
            if s1_atom_info[i1] < s1_atom_info[i2]:
                b1, b2 = i1, i2
            else:
                b1, b2 = i2, i1
            s1_bond_set.add("{} {}/{} {}".format(s1_atom_info[b1][0],
                s1_atom_info[b1][1], s1_atom_info[b2][0], s1_atom_info[b2][1]))
        s2_atom_info = list(zip(s2_rb.strs, s2_ab.names))
        s2_bond_set = set()
        for i1, i2 in structures[1].mol_blob.bond_indices:
            if s2_atom_info[i1] < s2_atom_info[i2]:
                b1, b2 = i1, i2
            else:
                b1, b2 = i2, i1
            s2_bond_set.add("{} {}/{} {}".format(s2_atom_info[b1][0],
                s2_atom_info[b1][1], s2_atom_info[b2][0], s2_atom_info[b2][1]))
        print("Bonds in {} model but not {} ({}):".format(s1_id, s2_id, len(s1_bond_set - s2_bond_set)), file=f)
        print("Bonds in {} model but not {} ({}):".format(s1_id, s2_id, len(s1_bond_set - s2_bond_set)), file=log_string)
        for bond_info in  s1_bond_set - s2_bond_set:
            print("\t" + bond_info, file=f)
            print("\t" + bond_info, file=log_string)
        print("Bonds in {} model but not {} ({}):".format(s2_id, s1_id, len(s2_bond_set - s1_bond_set)), file=f)
        print("Bonds in {} model but not {} ({}):".format(s2_id, s1_id, len(s2_bond_set - s1_bond_set)), file=log_string)
        for bond_info in  s2_bond_set - s1_bond_set:
            print("\t" + bond_info, file=f)
            print("\t" + bond_info, file=log_string)
        f.close()
        session.logger.info(log_string.getvalue())
test_desc = cli.CmdDesc()
