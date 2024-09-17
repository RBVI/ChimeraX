# try every command in bundle
import pytest

open_2tpk = ["open 2tpk autostyle false"]
alias_test_commands = ["alias foo bar", "alias list", "alias delete foo", "alias foo bar", "~alias foo"]
alignment_test_commands = [*open_2tpk, "align #1 to #1"]
camera_test_commands = ["camera"]
cartoon_test_commands = [*open_2tpk, "cartoon", "~cartoon", "cartoon :4-34 smooth .4 suppressBackboneDisplay false", "cartoon hide"]
directory_test_commands = ["cd", "pwd"]
clip_test_commands = [*open_2tpk, "clip", "clip axis x", "clip list", "~clip"]
cofr_test_commands = ["cofr", "~cofr"]
color_test_commands = [
    *open_2tpk,
    "color",
    "colour",
    "color green",
    "debug expectfail color ylbu",
    "color #1 pale green atoms trans .5 half true",
    "color name tpurple 60,0,80,50",
    "color name tgreen rgba(0%, 80%, 40%, 0.5)",
    "color list custom",
    "color list builtin",
    "color show blue",
    "colordef xyzzy 50,50,50,50",
    "~colordef xyzzy",
    "colourdef xyzzy 50,50,50,50",
    "~colourdef xyzzy",
    "palette list",

]
coordset_test_commands = [*open_2tpk, "coordset #1 0"]
delete_test_commands = [*open_2tpk, "delete :3"]
dssp_test_commands = [*open_2tpk, "dssp"]
graphics_test_commands = [
    *open_2tpk,
    "graphics",
    "graphics restart",
    "graphics triangles",
    "lighting",
    "material",
    "material ref .8 spec .8 exp 50 amb .2 trans true",
    "set bgcolor white",
    "style sphere",
    "style ball",
    "style stick",
    "transparency 50 atoms",
]
hide_show_test_commands = [*open_2tpk, "hide", "~show", "~display", "show", "display"]
perframe_test_commands = [
    *open_2tpk,
    'perframe "turn y 10; save \'~/Desktop/$1.png\'" frames 9 format %3d',
    "~perframe",
    "perframe stop",
]
rainbow_test_commands = [
    *open_2tpk,
    "rainbow",
    "rainbow residues palette RdBu-10",
]
rename_test_commands = [
    *open_2tpk,
    "rename #1 snafu"
]
roll_move_test_commands = [
    *open_2tpk,
    "roll",
    "roll z rock 100",
    "move x 10",
]
measure_test_commands = [
    *open_2tpk,
     "debug expectfail measure buriedarea #1 with #1",
    "measure buriedarea #1:1-18 with #1:19-36",
    "surface",
    "measure convexity #1",
    "measure length #1",
]
select_test_commands = [
    *open_2tpk,
    "select :12",
    "select up",
    "select down",
    "select zone :13 10 extend t",
    "select intersect :12 residues true",
    "~select"
]
setattr_test_commands = [
    *open_2tpk,
    "setattr /a res ribbon_color orange",
]
size_test_commands = [
    *open_2tpk,
    "size",
    "size :12 stickradius .15",
]
stop_test_commands = [
    *open_2tpk,
    "stop",
]
sym_test_commands = [
    *open_2tpk,
    "sym #1",
    "sym #1 assembly 1",
    "sym clear",
]
split_test_commands = [
    *open_2tpk,
    "split #1 atoms :1-18 atoms :19-36",
]
time_test_commands = [
    *open_2tpk,
    "time echo hi",
]
misc_test_commands = [
    *open_2tpk,
    "set",
    "turn x 30",
    "color red",
    "undo",
    "redo",
    "usage usage",
    "usage",
    "version",
    "version verbose",
    "version bundles",
    "version packages",
    "view",
    "view initial",
    "# debug expectfail wait",
    "windowsize",
    "windowsize 512 256",
    "zoom",
    "close",
]

commands = [
    open_2tpk,
    alias_test_commands,
    alignment_test_commands,
    camera_test_commands,
    cartoon_test_commands,
    directory_test_commands,
    clip_test_commands,
    cofr_test_commands,
    color_test_commands,
    coordset_test_commands,
    delete_test_commands,
    dssp_test_commands,
    graphics_test_commands,
    hide_show_test_commands,
    perframe_test_commands,
    rainbow_test_commands,
    rename_test_commands,
    roll_move_test_commands,
    select_test_commands,
    setattr_test_commands,
    size_test_commands,
    stop_test_commands,
    sym_test_commands,
    split_test_commands,
    time_test_commands,
    misc_test_commands,
]

@pytest.mark.parametrize("commands", commands)
def test_std_commands(test_production_session, commands):
    from chimerax.core.commands import run
    for command in commands:
        run(test_production_session, command)

def test_exit_command(test_production_session):
    from chimerax.core.commands import run
    with pytest.raises(SystemExit):
        run(test_production_session, "exit")
