TOP=../..
RTOP=$(realpath $(TOP))

install-gitmessage:
	git config --local commit.template $(RTOP)/utils/git/gitmessage

install-commitlint-hook:
	ln -s $(RTOP)/utils/git/commit-msg $(RTOP)/.git/hooks/
