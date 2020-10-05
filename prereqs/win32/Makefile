PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

DLL_DIR = '$(shell cygpath -u '$(VCToolsRedistDir)')'/x64/Microsoft.VC142.CRT

CMAKE_INSTALLER = cmake-3.5.2-win32-x86.msi
INNOSETUP_INSTALLER = innosetup-5.5.9-unicode.exe

all:

install:
	cp rsync.convert $(bindir)
	$(RSYNC) include/* $(includedir)
	if [ -d $(DLL_DIR) ]; then $(RSYNC) $(DLL_DIR)/*.dll $(bindir); fi


app-install:;
	$(RSYNC) include/* $(app_includedir)
	if [ -d $(DLL_DIR) ]; then $(RSYNC) $(DLL_DIR)/*.dll $(bindir); fi

fetch-cmake: $(CMAKE_INSTALLER)

$(CMAKE_INSTALLER):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/win32/$(CMAKE_INSTALLER)

fetch-innosetup: $(INNOSETUP_INSTALLER)

$(INNOSETUP_INSTALLER):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/win32/$(INNOSETUP_INSTALLER)

clean:
