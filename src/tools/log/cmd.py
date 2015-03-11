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
        f = open("/Users/pett/rm/1jj2_diff.txt", "w")
        s1_ab = structures[0].mol_blob.atoms
        s1_id = structures[0].id
        import io
        log_string = io.StringIO("")
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=f)
        print("# atoms in model {}: {}".format(s1_id, len(s1_ab)), file=log_string)
        s1_rb = s1_ab.residues
        s2_ab = structures[1].mol_blob.atoms
        s2_id = structures[1].id
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
        f.close()
        session.logger.info(log_string.getvalue())
test_desc = cli.CmdDesc()
