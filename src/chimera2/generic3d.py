from . import models

CATEGORY = "Generic 3D object"
#@models.register(CATEGORY)
class Generic3D(models.Model):

	def __init__(self):
		models.Model.__init__(self)
