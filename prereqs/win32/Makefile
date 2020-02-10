PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

DLL_DIR = '$(shell cygpath -u '$(VCToolsRedistDir)')'/x64/Microsoft.VC142.CRT

all:

install:
	cp rsync.convert $(bindir)
	$(RSYNC) include/* $(includedir)
	$(RSYNC) $(DLL_DIR)/*.dll $(bindir)


app-install:;
	$(RSYNC) include/* $(app_includedir)
	$(RSYNC) $(DLL_DIR)/*.dll $(app_bindir)

clean:
