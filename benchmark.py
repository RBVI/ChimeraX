# vi:set shiftwidth=4 expandtab:
# run "ChimeraX --nogui --exit --silent benchmark.py"
#
# This file is meant to be run weekly, and compared with
# previous weeks, so changes in performance can daylighted.
#
from chimerax.core.commands import run
from chimerax.core.logger import PlainTextLog
from chimerax.core import buildinfo


PDB_MMCIF_IDS = ["3fx2", "2hmg", "5xnl"]
HUGE_MMCIF_ID = "3j3q"
COUNT = 5


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
    times.append(t)


def time_open_close(open_cmd):
    open_times = []
    close_times = []
    for _ in range(COUNT):
        run_command(open_cmd, open_times)
        run_command("close", close_times)
    return open_times, close_times


def print_results(command, times):
    if len(times) < 3:
        run_time = sum(times) / len(times)
    else:
        # throw out high and low and average the rest
        run_time = sum(sorted(times)[1:-1]) / (len(times) - 2)
    print(f"{round(run_time, 4)}: {command}")


print(f"UCSF ChimeraX version: {buildinfo.version} ({buildinfo.date.split()[0]})")
print("Average time: command")

huge_open_cmd = f"open {HUGE_MMCIF_ID} format mmcif loginfo false"
open_times, close_times = time_open_close(huge_open_cmd)
print_results(huge_open_cmd, open_times)
print_results(f"({HUGE_MMCIF_ID}) close", close_times)

for pdb_id in PDB_MMCIF_IDS:
    open_cmd = f"open {pdb_id} format pdb loginfo false"
    open_times, close_times = time_open_close(open_cmd)
    print_results(open_cmd, open_times)
    print_results("(pdb) close", close_times)
    open_cmd = f"open {pdb_id} format mmcif loginfo false"
    open_times, close_times = time_open_close(open_cmd)
    print_results(open_cmd, open_times)
    print_results("(mmcif) close", close_times)

run(session, huge_open_cmd)

ball_times = []
run_command("style ball", ball_times)
print_results(f"style ball ({HUGE_MMCIF_ID})", ball_times)

cartoon_times = []
run_command("cartoon", cartoon_times)
print_results(f"cartoon ({HUGE_MMCIF_ID})", cartoon_times)
