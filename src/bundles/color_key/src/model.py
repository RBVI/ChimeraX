# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

_model = None
def get_model(session, *, create=True):
	if not _model and create:
		# constructor sets _model, which makes session restore easier
		ColorKeyModel(session)
	return _model

from chimerax.core.models import Model

class ColorKeyModel(Model):

	pickable = False
	casts_shadows = False

	def __init__(self, session):
		super().__init__(self, "Color key", session)
		global _model
		_model = self

		self._window_size = None
		self._texture_pixel_scale = 1
		self._aspect = 1
		self.needs_update = True

		from chimerax.core.triggerset import TriggerSet
		self.triggers = TriggerSet()
		self.triggers.add_trigger("changed")
		self.triggers.add_trigger("closed")

		self._key_position = None
		self._rgbas_and_labels = [((0,0,1,1), "min"), ((1,1,1,1), ""), ((1,0,0,1), "max")]
		self._num_label_spacing = "proportional to value"
		self._color_treatment = "blended"

	def delete(self):
		self.triggers.activate_trigger("closed", None)
		super().delete()
		global _model
		_model = None

	def draw(self, renderer, draw_pass):
		if self._key_position is None:
			return
		self._update_graphics(renderer)
		super().draw(renderer, draw_pass)

	def _update_graphics(self, renderer):
		"""
		Recompute the key's texture image or update its texture coordinates that position it in the window
		based on the rendered window size.  When saving an image file, the rendered size may differ from the
		on-screen window size.  In that case, make the label size match its relative size seen on the screen.
		"""
		window_size = renderer.render_size()

		# Remember the on-screen size if rendering off screen
		if getattr(renderer, 'image_save', False):
			# When saving an image, match the key's on-screen fractional size, even though the image size
			# in pixels may be different from on screen
			sw, sh = self.session.main_window.window_size
			w, h = window_size
			pixel_scale = (w / sw) if sw > 0 else 1
			aspect = (w*sh) / (h*sw) if h*sw > 0 else 1
		else:
			pixel_scale = renderer.pixel_scale()
			aspect = 1
		if pixel_scale != self._texture_pixel_scale or aspect != self._aspect:
			self._texture_pixel_scale = pixel_scale
			self._aspect = aspect
			self.needs_update = True

		# Need to reposition key if window size changes
		win_size_changed = (window_size != self._window_size)
		if win_size_changed:
			self._window_size = window_size

		if self.needs_update:
			self.needs_update = False
			self._update_key_image()
		elif win_size_changed:
			self._position_key_image()

	def _update_key_image(self):
		upper_left_xy, lower_right_xy = self._key_position
		ulx, uly = upper_left_xy
		lrx, lry = lower_right_xy
		win_w, win_h = self._window_size
		x_pixels = int(win_w * (lrx - ulx) + 0.5)
		y_pixels = int(win_h * (uly - lry) + 0.5)
		if x_pixels > y_pixels:
			layout = "horizontal"
			long_size = x_pixels
		else:
			layout = "vertical"
			long_size = y_pixels
		rect_positions = self._rect_positions(long_size):
		#TODO: keep cribbing from label2d._update_label_image (in turn, from: drawing.text_image_rgba)

	def _position_key_image(self):
		#TODO
		pass

	def _rect_positions(self, long_size):
		proportional = False
		texts = [color_text[1] for color_text in self._rgbas_and_labels]
		if self._num_label_spacing == "proportional to value":
			try:
				values = [float(t) for t in texts]
			except ValueError:
				pass
			else:
				if values == sorted(values):
					proportional = True
				else:
					values.reverse()
					if values == sorted(values):
						proportional = True
					values.reverse()
		if not proportional:
			values = range(len(texts))
		if self._color_treatment == "blended":
			val_size = abs(values[0] - values[-1])
			rect_positions = [long_size * abs(v - values[0]/val_size for v in values]
		else:
			v0 = values[0] - (values[1] - values[0])/2.0
			vN = values[-1] + (values[-1] - values[-2])/2.0
			val_size = abs(vN-v0)
			positions = [long_size * abs(v-v0) / val_size for v in values]
			rect_positions = [0.0] + [(positions[i] + positions[i+1])/2.0
				for i in range(len(values)-1] + [1.0]
		return rect_positions
