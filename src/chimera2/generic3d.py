from . import open_models

CATEGORY = "Generic 3D object"
#@open_models.register(CATEGORY)
class Generic3D(open_models.Model):

	def __init__(self):
		open_models.Model.__init__(self)
