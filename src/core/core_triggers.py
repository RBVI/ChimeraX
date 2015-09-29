# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
core_triggers: access to core triggers
======================================

"""

trigger_names = [
    "atomic changes",
    "graphics update",
    "new frame",
    "rendered frame",
    "shape changed",
]

def register_core_triggers(core_triggerset):
    for tn in trigger_names:
        core_triggerset.add_trigger(tn)
