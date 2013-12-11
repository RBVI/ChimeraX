from . import open_models, scene

CATEGORY = "Generic 3D object"
#@open_models.register(CATEGORY)
class Generic3D(open_models.Model):

	def __init__(self):
		open_models.Model.__init__(self)

	def make_graphics(self):
		self.graphics = scene.Graphics()
