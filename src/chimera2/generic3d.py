from . import open_models, scene

# TODO: make graphics a subclass that memo-izes what's added, so it can
# be recreated if the WebGL context needs to be recreated.  Might also
# be useful for normal OpenGL if context can be lost when laptop sleeps
# or hibernates (not sure if that happens or not).

CATEGORY = "Generic 3D object"
#@open_models.register(CATEGORY)
class Generic3D(open_models.Model):

	def __init__(self):
		open_models.Model.__init__(self)

	def make_graphics(self):
		self.graphics = scene.Graphics()
		self.graphics.default_group_add = True
