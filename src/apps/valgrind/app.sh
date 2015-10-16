#!/bin/sh

bindir=`/usr/bin/dirname "$0"`
app=`/usr/bin/basename "$0" .valgrind`
datadir="$bindir/../share"

exec valgrind --tool=memcheck --error-limit=no --suppressions="$datadir"/valgrind-python.supp --suppressions="$datadir"/valgrind.suppress --gen-suppressions=all --track-origins=yes --num-callers=30 "$bindir/$app" $@
