PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

DLL_DIR = '$(shell cygpath -u '$(VCToolsRedistDir)')'/x64/Microsoft.VC142.CRT

all:

install:
	cp rsync.convert $(bindir)
	$(RSYNC) include/* $(includedir)
	if [ -d $(DLL_DIR) ]; then $(RSYNC) $(DLL_DIR)/*.dll $(bindir); fi


app-install:;
	$(RSYNC) include/* $(app_includedir)
	if [ -d $(DLL_DIR) ]; then $(RSYNC) $(DLL_DIR)/*.dll $(bindir); fi

clean:
