PREREQ_MAKE = 1
TOP	= ../..
include $(TOP)/mk/config.make

# Install Qt shim that handle PyQt5 and PySide2

install:
	$(RSYNC) Qt $(PYSITEDIR)

app-install:
	$(RSYNC) Qt $(APP_PYSITEDIR)

upload_new_version: clean wheel
	$(RSYNC) dist/qtshim*.whl $(PREREQS_UPLOAD)/qt/

wheel:
	$(APP_PYTHON_BIN) -m build .

clean:
	rm -rf *.egg-info build dist
