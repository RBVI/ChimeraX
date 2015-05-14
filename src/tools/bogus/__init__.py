# vi: set expandtab shiftwidth=4 softtabstop=4:


def start_tool(session, ti):
    # If providing more than one tool in package,
    # look at the name in 'ti.name' to see which is being started.
    from . import gui
    try:
        ui = getattr(gui, ti.name + "UI")
    except AttributeError:
        raise RuntimeError("cannot find UI for tool \"%s\"" % ti.name)
    else:
        return ui(session, ti)


def register_command(command_name):
    from . import cmd
    from chimera.core import cli
    desc_suffix = "_desc"
    for attr_name in dir(cmd):
        if not attr_name.endswith(desc_suffix):
            continue
        subcommand_name = attr_name[:-len(desc_suffix)]
        try:
            func = getattr(cmd, subcommand_name)
        except AttributeError:
            print("no function for \"%s\"" % subcommand_name)
            continue
        desc = getattr(cmd, attr_name)
        cli.register(command_name + ' ' + subcommand_name, desc, func)

    from chimera.core import atomspec
    atomspec.register_selector(None, "odd", _odd_models)


def _odd_models(session, models, results):
    for m in models:
        if m.id[0] % 2:
            results.add_model(m)
            results.add_atoms(m.mol_blob.atoms)
