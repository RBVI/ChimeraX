"""
Common code for molecular structure
"""

from . import models

CATEGORY = "Molecular structure"
#@models.register(CATEGORY)
class Molecule(models.Model):

	def __init__(self):
		models.Model.__init__(self)
