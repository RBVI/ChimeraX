"""
chimera2: package of chimera version 2 files
============================================

"""

# All chimera2 triggers should be in triggers
#
#	Trigger data that correspond to data types, should be a dictionary
#	of { 'created': list, 'modified': list, 'deleted': list }.  Instances
#	should only appear in one list.  The list of deleted instances should
#	be of instances that are about to be deleted.

from .triggerset import TriggerSet
triggers = TriggerSet()
