RECIPES = ubuntu-20.04.def ubuntu-22.04.def \
	  rhel-8.def rhel-9.def \
	  debian-10.def
IMAGES = $(RECIPES:.def=.img)
SUDO =

IMAGE_DIR = /usr/local/projects/chimerax/nobackup

.SUFFIXES: .def .img

.def.img:
ifneq (,$(SUDO))
	$(SUDO) singularity build $*.img $<
else
	singularity build --fakeroot $*.img $<
endif

images: $(IMAGES)

install: $(IMAGES)
	mv $(IMAGES) $(IMAGE_DIR)

clean:
	rm -f $(IMAGES)
