# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

commands = [
    'log settings warn false error false',
    'open 1mtx',
    'mmaker #1.1 to #1.2',
    'mmaker #1.1-5 to #1.6 alg sw matrix PAM-150 cutoff 5.0',
    'close',
    'open 3fx2',
    'hbonds',
    '~hbonds',
    'hbonds two true intrares false',
    'close',
    'open 1zik',
    'clash #1/a restrict #1/b reveal t',
    '~clash',
    'contact #1 dist 3.5 reveal t',
    'close',
    'open 1mtx',
    'morph #1.1-20 same t',
    'wait 20',
    'close',
    'open 1a0m',
    'addh hb f',
    '~display /b',
    'style stick',
    'ribbon /b',
    'surface enclose #1 resolution 5',
    'surface close',
    'surface #1',
    'measure sasa /a',
    'measure buriedarea /a with /b',
    'color #1 bychain',
    'color @CA purple',
    'light soft',
    'camera field 30',
    'camera',
    'close #1',
    'open pdb:4hhb',
    'addh',
    'color /a@CB tan',
    'colordef wow green',
    'color /c wow',
    '~colordef wow',
    'interfaces #1',
    'crossfade 10',
    'delete /c',
    '~display',
    'display /d',
    'display',
    'echo Here we are',
    'color sequential #1 chains palette rainbow',
    'color /a dodger blue target a',
    'open emdb:1080',
    'fitmap #1 in #2 resolution 10 metric correlation',
    'help fitmap',
    'ks ha',
    'ks rb',
    '2dlabel text "Look at this" size 36 xpos .2 ypos .8 color yellow',
    '2dlabel text "Great Scots"',
    '2dlabel delete all',
    'lighting',
    'lighting color red',
    'lighting full',
    'info',
    'log hide',
    'log show',
    'material specular 1.2 exponent 100',
    'material shiny',
    'molmap #1 5 grid 1',
    'move y 1 20',
    'movie record super 3',
    'wait',
    'movie encode ~/Desktop/test_movie.mp4 quality high',
    # 'vr on',
    # 'vr off',
    'perframe "turn y 15" frames 10',
    'pwd',
    'roll z',
    'stop',
    'ribbon /b/c',
    '~disp /b/c',
    '~ribbon',
    'measure sasa /d',
    # 'save ~/Desktop/test_image.jpg',  # JPEG support missing, bug #186
    'save ~/Desktop/test_image.png',
    'surface #1',
    'color #1 byatom',
    'color sample #1 map #2',
    'set bg gray silhouettes true',
    'device snav on fly true',
    'device snav off',
    'split #1',
    'style ball',
    'surface :46-80',
    'surf hide #1',
    'surf show #1',
    'close',
    'open 2bbv',
    'sym #1',
    'sym #1 as 4',
    'sym clear #1',
    'toolshed list',
    'toolshed reload',
    'ui tool hide \"Command Line Interface\"',
    'ui tool show \"Command Line Interface\"',
    'ui tool hide Help',
    'info models',
    'molmap #1 5 grid 1',
    'volume #3 level .05',
    'volume #3 color tan enclose 1e5',
    'vop gaussian #3 sdev 2',
    'vop subtract #3,4',
    'view',
    'debug exectest',
    'echo finished test',
]


def run_commands(session, commands=commands, stderr=False):
    log = session.logger
    from chimerax.core.commands import run
    for c in commands:
        if stderr:
            import sys
            print(c, file=sys.__stderr__)
        log.info('> ' + c)
        run(session, c)


def run_exectest(session, bi):
    exec_dir = bi.executable_dir()
    if not exec_dir:
        raise RuntimeError("no executable directory in module %r" % bi.name)
    import os.path
    import subprocess
    log = session.logger
    executable = os.path.join(exec_dir, "exectest.exe")
    command = [executable]
    log.info("executing %s" % command)
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    result.check_returncode()
    if result.stdout:
        log.info("Output on stdout:\n%s\n-----" % result.stdout)
    else:
        log.info("No output on stdout")
    if result.stderr:
        log.warning("Output on stderr:\n%s\n-----" % result.stderr)
    else:
        log.info("No output on stderr")


def run_expectfail(session, command):
    from chimerax.core.errors import NotABug, UserError
    from chimerax.core.commands import run, Command

    # first confirm that command is an actual command
    cmd = Command(session)
    cmd.current_text = command
    cmd._find_command_name(no_aliases=True)
    if not cmd._ci:
        raise UserError("Unknown commmand")
    try:
        cmd.run(command)
    except NotABug:
        session.logger.info("Failed as expected")
    else:
        raise UserError("Command failed to fail")
