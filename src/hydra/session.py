from .models import Models
class Session(Models):
    '''
    A session holds the list of open models, camera, graphics window, scenes, ...
    all the state.  The main purpose is to bring together all of objects defining
    the 3d world.
    '''
    
    def __init__(self):

        Models.__init__(self)		# Manages list of open models

        self.application = None
        'Qt application object, a QApplication.'

        self.main_window = None
        'Main user interface window, a :py:class:`~.ui.gui.MainWindow.`'

        self.view = None
        'Main window view, a :py:class:`~.ui.view.View`'

        from .ui import commands, shortcuts
        self.commands = commands.Commands(self)
        'Available commands, a :py:class:`~.ui.commands.Commands`'

        self.keyboard_shortcuts = shortcuts.Keyboard_Shortcuts(self) 
        'Available keyboard shortcuts, a :py:class:`~.ui.shortcuts.Keyboard_Shortcuts`'

        self.log = None
        'Command, error, info log, :py:class:`~.ui.gui.Log`'

        self.file_types = None
        'Table of file types that can be read, used by :py:func:`~.file_io.opensave.file_types`'

        self.databases = {}
        'For fetching pdb and map models from the web, used by :py:func:`~.file_io.fetch.register_fetch_database`'

        from .file_io import history
        self.file_history = history.File_History(self)
        'Recently opened files, a :py:class:`.file_io.history.File_History`'

        self.last_session_path = None
        'File path for last opened session.'

        from . import scenes
        self.scenes = scenes.Scenes(self)
        'Saved scenes, a :py:class:`~.scenes.Scenes`'

        self.bond_templates = None
        'Templates for creating bonds for standard residues, a :py:class:`~.molecule.connect.Bond_Templates`'

        from .map import defaultsettings
        self.volume_defaults = defaultsettings.Volume_Default_Settings()
        'Default volume model display settings, a :py:class:`~.map.defaultsettings.Volume_Default_Settings`'

        self.fit_list = None
        'Dialog listing map fits, a :py:class:`~.map.fit.fitlist.Fit_List`'

        self.space_navigator = None
        'Space Navigator device handler, a :py:class:`~.ui.spacenavigator.snav.Space_Navigator`'

        self.oculus = None
        'Oculus Rift head tracking device handler, a :py:class:`~.ui.oculus.track.Oculus_Head_Tracking`'

    def start(self):

        from .ui import gui, commands, shortcuts
        import sys
        self.application = app = gui.Hydra_App(sys.argv, self)
        self.main_window = mw = app.main_window
        self.view = mw.view
        self.log = log = gui.Log(mw)
        commands.register_commands(self.commands)
        shortcuts.register_shortcuts(self.keyboard_shortcuts)
        self.file_history.show_thumbnails()

        from . import ui
        ui.set_show_status(self.show_status)
        ui.set_show_info(self.show_info)

        log.stdout_to_log()
        log.exceptions_to_log()

        # Set default volume data cache size.
        from .map.data import data_cache
        data_cache.resize(self.volume_defaults['data_cache_size'] * (2**20))

        mw.show()

        gui.start_event_loop(app)

        self.file_history.write_history()
        sys.exit(status)

    def show_status(self, msg, append = False):
        '''Show a status message at the bottom of the main window.'''
        self.main_window.show_status(msg, append)

    def show_info(self, msg, color = None):
        '''Write information such as command output to the log window.'''
        self.log.log_message(msg, color)
