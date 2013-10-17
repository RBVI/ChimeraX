TOP = ../../..
include $(TOP)/mk/config.make

DATA_DIR = $(PYSITEDIR)/hydra/example_sessions
DATA_FILES = 6bna.hy 6bna.jpg Tropomyosin.hy Tropomyosin.jpg \
	     XrayDensity.hy XrayDensity.jpg Reovirus.hy Reovirus.jpg \
	     sessions

all:

install: all
	-mkdir $(DATA_DIR)
	$(RSYNC) $(DATA_FILES) $(DATA_DIR)
