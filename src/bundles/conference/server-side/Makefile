# Assumptions:
# - Running on CentOS 7 with pcs

# CHIMERAX_DIR: install location for ChimeraX
CHIMERAX_DIR=/home/socr/a/chimera/chimerax_daily

INSTALL=/usr/bin/install
MEETING_DIR=$(CHIMERAX_DIR)/lib/python3.7/site-packages/chimerax/meeting

SRC_MUX=$(MEETING_DIR)/mux.py
DST_MUX=/usr/local/bin/cxconference

SRC_CERT=$(MEETING_DIR)/server.pem
DST_CERT=/usr/local/etc/cxconference.pem
# DST_CERT must match location in mux.py

SERVICE=cxconference
INIT_SCRIPT=/etc/init.d/$(SERVICE)


#
# update the executable and certificate from current ChimeraX
#
update:
	$(INSTALL) --owner=root --group=root --mode=755 $(SRC_MUX) $(DST_MUX)
	$(INSTALL) --owner=root --group=root --mode=640 $(SRC_CERT) $(DST_CERT)


#
# create pcs resource for cxconference
#
pcs:
	$(INSTALL) --owner=root --group=root --mode=755 \
		cxconference.sh $(INIT_SCRIPT)
	# sclcp $(INIT_SCRIPT) $(INIT_SCRIPT)
	@echo Remember to sync $(INIT_SCRIPT), $(DST_MUX), and $(DST_CERT) to other cluster nodes
	pcs resource create cxconference.rbvi.ucsf.edu ocf:heartbeat:IPaddr2 \
		ip=169.230.27.25 cidr_netmask=24 \
		--disabled
	pcs resource create cxconference-mux lsb:$(SERVICE) \
		--disabled
	pcs resource group add Cxconference \
		cxconference.rbvi.ucsf.edu cxconference-mux
	@echo Remember to enable "Cxconference" PCS resource
