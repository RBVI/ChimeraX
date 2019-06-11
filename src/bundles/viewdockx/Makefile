include ../Makefile.bundle

LIBS = src/lib/flot
CLEAN_ALWAYS += src/lib/flot src/lib/rateyo src/lib/jquery.multiselect.*

wheel install app-install:	$(LIBS)

# From http://www.flotcharts.org/
src/lib/flot:	flot-0.8.3.zip
	# touch needed because unzip restores update time
	rm -rf $@ && unzip $? && rm -rf flot/examples && mv flot $@ && touch $@

# From http://rateyo.fundoocode.ninja/
src/lib/rateyo:	rateyo-2.3.2.zip
	rm -rf $@ && unzip $? && mv v2.3.2 $@ && rm -rf __MACOSX && touch $@

# From https://www.jqueryscript.net/form/jQuery-Plugin-For-Multi-Select-List-with-Checkboxes-MultiSelect.html
MS_DIR = jQuery-Plugin-For-Multi-Select-List-with-Checkboxes-MultiSelect
src/lib/jquery.multiselect.js:	multiselect.zip
	unzip $? && mv $(MS_DIR)/jquery.multiselect.* src/lib \
		&& rm -rf $(MS_DIR) && touch $@
