#!/bin/bash

# Run ChimeraX conference hub
# The hub code ($CXCONFERENCE) is actually a copy of
# the ChimeraX meeting multiplexer code:
# CHIMERAX/lib/python3.7/site-packages/chimerax/meeting/mux.py
# The arguments are:
#   -h hostname (Host name to listen on, default "localhost")
#   -p port (Port number to listen on, default 8443
#   -l loglevel (Logging level, default "INFO")
#   -a admin_word (Magic word for querying server, default "chimeraxmux")
#   -P pid_file (Path to PID file, run in background if not None, default None)

LogFile=/var/log/cxconference.log
PidFile=/var/run/cxconference.pid

CXCONFERENCE=/usr/local/bin/cxconference
CXCONFERENCE_ARGS="-h cxconference.rbvi.ucsf.edu -p 443 -L $LogFile -P $PidFile"
# For testing
#CXCONFERENCE_ARGS="-h localhost -p 32443 -L LogFile -P $PidFile"

# source function library
. /etc/rc.d/init.d/functions

case "$1" in
    'start')
        daemon --pidfile=$PidFile $CXCONFERENCE $CXCONFERENCE_ARGS
        exit 0
    ;;

    # Note that stop returns success immediately if it cannot
    # find a sshd process in memory.
    'stop')
        if [ -f $PidFile ]
        then
                killproc -p $PidFile $CXCONFERENCE
        else
                base=${CXCONFERENCE##*/}
                echo $"${base} is stopped"
        fi
        exit 0
    ;;

    'status')
        status -p $PidFile $CXCONFERENCE
        exit $?
    ;;

    *)
        $ECHO "usage: $0 {start|stop|status}"
        exit 1
    ;;
esac
