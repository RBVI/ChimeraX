# -----------------------------------------------------------------------------
#
def register_commands(commands):
    '''
    Registers the standard commands.
    '''
    add = commands.add_command
    from ..files.opensave import open_command, save_command, close_command
    add('open', open_command)
    add('save', save_command)
    add('close', close_command)
    from ..map import fetch_emdb, fetch_eds
    from ..molecule import fetch_pdb
    s = commands.session
    fetch_pdb.register_pdb_fetch(s)
    fetch_emdb.register_emdb_fetch(s)
    fetch_eds.register_eds_fetch(s)
    from ..map import molmap
    add('molmap', molmap.molmap_command)
    from ..map import volumecommand
    add('volume', volumecommand.volume_command)
    from ..map.fit import fitcmd
    add('fitmap', fitcmd.fitmap_command)
    from ..map.filter import vopcommand
    add('vop', vopcommand.vop_command)
    from ..map import series
    add('vseries', series.vseries_command)
    from ..molecule import align, mcommand
    add('align', align.align_command)
    add('show', mcommand.show_command)
    add('hide', mcommand.hide_command)
    add('color', mcommand.color_command)
    add('style', mcommand.style_command)
    from ..surface import gridsurf, sasa
    add('surface', gridsurf.surface_command)
    add('area', sasa.area_command)
    from .. import scenes
    add('scene', scenes.scene_command)
    from . import cameracmd
    add('camera', cameracmd.camera_command)
    from ..devices import device
    add('device', device.device_command)
    from . import lightcmd
    add('lighting', lightcmd.lighting_command)
    from . import materialcmd
    add('material', materialcmd.material_command)
    from .. import ui
    add('windowsize', ui.window_size_command)
    from ..molecule import blastpdb
    add('blast', blastpdb.blast_command)
    from ..molecule import ambient
    add('ambient', ambient.ambient_occlusion_command)
    from .cyclecmd import cycle_command
    add('cycle', cycle_command)
    from . import silhouettecmd
    add('silhouette', silhouettecmd.silhouette_command)
    from ..movie import movie_command, wait_command
    add('movie', movie_command)
    add('wait', wait_command)
    from .turncmd import turn_command
    add('turn', turn_command)
    from .motion import freeze_command
    add('freeze', freeze_command)
    from ..measure import measure_command
    add('measure', measure_command)

# -----------------------------------------------------------------------------
#
class Commands:
    '''Keep the list of commands and run them.'''
    def __init__(self, session):
        self.session = session
        self.commands = {}
        self.cmdabbrev = None
        self.history = Command_History(session)

    def add_command(self, name, function):
        '''
        Register a command with a given name and function to call.
        '''
        self.commands[name] = function
        self.cmdabbrev = None

    def run_command(self, text):
        '''
        Invoke a command.  The command and arguments are a string that will be
        parsed by a registered command function.
        '''
        self.history.add_to_command_history(text)
        for c in text.split(';'):
            self.run_single_command(c)

    def run_single_command(self, text):
        ses = self.session
        ses.show_info('> %s' % text, color = '#008000')
        fields = text.split(maxsplit = 1)
        if len(fields) == 0:
            return
        cmd = fields[0]
        cab = self.cmdabbrev
        if cab is None:
            from . import parse
            self.cmdabbrev = cab = parse.abbreviation_table(self.commands.keys())
        if cmd in cab:
            cmd = cab[cmd]
            f = self.commands[cmd]
            args = fields[1] if len(fields) >= 2 else ''
            failed = False
            from .parse import CommandError
            try:
                f(cmd, args, ses)
            except CommandError as e:
                ses.show_status(str(e))
                failed = True
            if not failed:
                ses.log.insert_graphics_image()
        else:
            ses.show_status('Unknown command %s' % cmd)

class Command_History:
    def __init__(self, session):
        self.session = session
        self.commands = None
        self.file_lines = None

    def command_list(self):
        if self.commands is None:
            self.read_command_history()
            self.session.at_quit(self.save_command_history)
        return self.commands
        
    def add_to_command_history(self, text):
        self.command_list().append(text)

    def read_command_history(self, filename = 'commands'):
        from ..files import history
        path = history.user_settings_path(filename)
        import os.path
        if os.path.exists(path):
            f = open(path, 'r')
            h = [line.rstrip() for line in f.readlines()]
            f.close()
        else:
            h = []
        self.commands = remove_repeats(h)
        self.file_lines = len(self.commands)

    def save_command_history(self, filename = 'commands'):
        h = self.commands
        if h is None:
            return
        if len(h) == self.file_lines:
            return      # No new commands
        from ..files import history
        path = history.user_settings_path(filename)
        f = open(path, 'a')
        for cmd in h[self.file_lines:]:
            f.write(cmd.strip() + '\n')
        f.close()

    def show_command_history(self, filename = 'commands'):
        '''
        Show a text window showing the invoked commands and their arguments for
        this session in the order they were executed.  Clicking on a command causes
        it to be run again.
        '''
        mw = self.session.main_window
        if mw.showing_text() and mw.text_id == 'command history':
            mw.show_graphics()
            return

        cmds = self.unique_commands()
        html = self.history_html(cmds)

        mw.show_text(html, html=True, id = 'command history',
                     anchor_callback = lambda url, s=self, c=cmds: s.insert_clicked_command(url,c))

    def unique_commands(self):
        lines = self.command_list()

        # Get unique lines, order most recent first.
        cmds = []
        found = set()
        for line in lines[::-1]:
            cmd = line.rstrip()
            if not cmd in found:
                cmds.append(cmd)
                found.add(cmd)
        return cmds

    def history_html(self, cmds):
        hlines = ['<html>', '<head>', '<style>',
                  'a { text-decoration: none; }',   # No underlining links
                  '</style>', '</head>', '<body>']
        hlines.extend(['<a href="%d"><p>%s</p></a>' % (i,line) for i,line in enumerate(cmds)])
        hlines.extend(['</body>','</html>'])
        html = '\n'.join(hlines)
        return html

    def insert_clicked_command(self, url, shown_commands):
        surl = url.toString(url.PreferLocalFile)
        # Work around Qt bug where it prepends a directory path to the anchor
        # even when QtTextBrowser search path and source are cleared.
        cnum = surl.split('/')[-1]
        c = int(cnum)         # command number
        if c < len(shown_commands):
            cmd = shown_commands[c]
            self.show_command(cmd)
            self.session.commands.run_command(cmd)

    def show_command(self, cmd):
        self.session.main_window.set_command_line_text(cmd)

    def show_previous_command(self, step = -1):
        cl = self.command_list()
        n = len(cl)
        if n == 0:
            return
        p = getattr(self, 'prev_command', n)
        p += step
        if p >= 0 and p < n:
            self.prev_command = p
            self.show_command(cl[p])

    def show_next_command(self):
        self.show_previous_command(step = 1)

# -----------------------------------------------------------------------------
#
def remove_repeats(strings):
    us = []
    sprev = None
    for s in strings:
        if s != sprev:
            us.append(s)
            sprev = s
    return us

