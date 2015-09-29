# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
core_triggers: access to core triggers
======================================

Some core triggers are registered elsewhere, such as in the View constructor.

"""

trigger_info = {
    "atomic changes": False,
    "graphics update": True,
    "rendered frame": True,
    "shape changed": False,
}

def register_core_triggers(core_triggerset):
    for tn, rbh in trigger_names:
        core_triggerset.add_trigger(tn, remove_bad_handlers=rbh)
