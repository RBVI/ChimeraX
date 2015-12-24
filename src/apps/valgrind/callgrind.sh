#!/bin/sh

bindir=`/usr/bin/dirname "$0"`
app=`/usr/bin/basename "$0" .valgrind`
datadir="$bindir/../share"

exec valgrind --tool=callgrind --collect-atstart=no "$bindir/$app" $@
