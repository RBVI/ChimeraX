RECIPES = ubuntu-22.04.def ubuntu-24.04.def \
	  rhel-8.def rhel-9.def rhel-10.def
IMAGES = $(RECIPES:.def=.sif)
SUDO =

IMAGE_DIR = /usr/local/projects/chimerax/nobackup

.SUFFIXES: .def .sif

.def.sif:
ifneq (,$(SUDO))
	$(SUDO) apptainer build $*.sif $<
else
	apptainer build --fakeroot $*.sif $<
endif

images: $(IMAGES)

install: $(IMAGES)
	mv $(IMAGES) $(IMAGE_DIR)

clean:
	rm -f $(IMAGES)
