PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

VC_DIR = '$(shell cygpath -u '$(VCINSTALLDIR)')'

all:

install:
	cp rsync.convert $(bindir)
	$(RSYNC) include/* $(includedir)
	$(RSYNC) $(VC_DIR)/redist/x64/Microsoft.VC140.CRT/*.dll $(bindir)


app-install:;
	$(RSYNC) include/* $(app_includedir)
	$(RSYNC) $(VC_DIR)/redist/x64/Microsoft.VC140.CRT/*.dll $(app_bindir)

clean:
