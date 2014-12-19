TOP	= ../..
include $(TOP)/mk/config.make

all:

install:
	cp rsync.convert $(bindir)
	$(RSYNC) include/* $(includedir)

app-install:;

clean:
