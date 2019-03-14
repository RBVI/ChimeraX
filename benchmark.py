# vi:set shiftwidth=4 expandtab:
# run "ChimeraX --nogui --exit --silent benchmark.py"
#
# This file is meant to be run weekly, and compared with
# previous weeks, so changes in performance can daylighted.
#
import gc
import os
import socket
import subprocess
from chimerax.core.commands import run
from chimerax.core.logger import PlainTextLog
from chimerax.core import buildinfo


PDB_MMCIF_IDS = ["3fx2", "2hmg", "5xnl"]
HUGE_MMCIF_ID = "3j3q"
COUNT = 5


def get_memory_use():
    gc.collect()
    output = subprocess.check_output(['/usr/bin/pmap', str(os.getpid())])
    usage = output.split()[-1].decode()
    return usage


class NoOutputLog(PlainTextLog):

    def log(self, level, msg):
        pass

    def status(self, msg, color, secondary):
        pass


session = session  # noqa -- shut up flake8
session.logger.add_log(NoOutputLog())


def time_command(command):
    from time import time
    t0 = time()
    run(session, command)
    t1 = time()
    return t1 - t0


def run_command(command, times):
    t = time_command(command)
    #session.main_view.check_for_drawing_change()
    times.append(t)


def time_open_close(open_cmd):
    open_times = []
    close_times = []
    for _ in range(COUNT):
        run_command(open_cmd, open_times)
        run_command("close", close_times)
    return open_times, close_times


def print_results(command, times):
    from numpy import std
    if len(times) >= 3:
        # throw out high and low
        times = sorted(times)[1:-1]
    mean = sum(times) / len(times)
    var = std(times)
    print(f"{round(mean, 4)} \N{Plus-Minus Sign} {round(var,3)}: {command}")


def print_delta_memory(tag, first, second):
    delta = int(second[:-1]) - int(first[:-1])
    print(f"{tag}: {delta}{first[-1]}")


print(f"UCSF ChimeraX version: {buildinfo.version} ({buildinfo.date.split()[0]})")
print(f"Running benchmark on {socket.gethostname()}")

start_usage = get_memory_use()
print(f"Starting memory use:  {start_usage}")
usage0 = start_usage

print("Average time: command")

huge_open_cmd = f"open {HUGE_MMCIF_ID} format mmcif loginfo false"
open_times, close_times = time_open_close(huge_open_cmd)
print_results(huge_open_cmd, open_times)
print_results(f"({HUGE_MMCIF_ID} mmcif) close", close_times)
usage1 = get_memory_use()
print_delta_memory("Increased memory use", usage0, usage1)
usage0 = usage1

for pdb_id in PDB_MMCIF_IDS:
    open_cmd = f"open {pdb_id} format pdb loginfo false"
    open_times, close_times = time_open_close(open_cmd)
    print_results(open_cmd, open_times)
    print_results(f"({pdb_id} pdb) close", close_times)
    open_cmd = f"open {pdb_id} format mmcif loginfo false"
    open_times, close_times = time_open_close(open_cmd)
    print_results(open_cmd, open_times)
    print_results(f"({pdb_id} mmcif) close", close_times)
    usage1 = get_memory_use()
    print_delta_memory("Increased memory use", usage0, usage1)
    usage0 = usage1

run(session, huge_open_cmd)

ball_times = []
run_command("style ball", ball_times)
print_results(f"style ball ({HUGE_MMCIF_ID})", ball_times)
usage1 = get_memory_use()
print_delta_memory("Increased memory use", usage0, usage1)
usage0 = usage1

cartoon_times = []
run_command("cartoon", cartoon_times)
print_results(f"cartoon ({HUGE_MMCIF_ID})", cartoon_times)
usage1 = get_memory_use()
print_delta_memory("Increased memory use", usage0, usage1)
usage0 = usage1

end_usage = get_memory_use()
print(f"Ending memory use:    {end_usage}")
print_delta_memory("Total memory increase", start_usage, end_usage)
