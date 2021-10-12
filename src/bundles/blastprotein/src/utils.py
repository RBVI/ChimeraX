# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2021 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from itertools import count
from typing import NamedTuple

SeqGapChars = "-. "

class InstanceGenerator:
    _instance_iterator = count(1)
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        return next(InstanceGenerator._instance_iterator)

def make_instance_name(prefix="bp"):
    return "".join([prefix, str(next(InstanceGenerator()))])

class BlastParams(NamedTuple):
    chain: str
    database: str
    cutoff: float
    maxSeqs: int
    matrix: str

class SeqId(NamedTuple):
    hit_name: str
    sequence: str
