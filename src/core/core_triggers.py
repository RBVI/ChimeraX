# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
core_triggers: access to core triggers
======================================

Some core triggers are registered elsewhere, such as in the View constructor.

"""

trigger_info = {
    "atomic changes": False,
    "begin restore session": False,
    "begin save session": False,
    "end restore session": False,
    "end save session": False,
    "frame drawn": True,
    "graphics update": True,
    "new frame": True,
    "shape changed": False,
}

def register_core_triggers(core_triggerset):
    for tn, rbh in trigger_info.items():
        core_triggerset.add_trigger(tn, remove_bad_handlers=rbh)
