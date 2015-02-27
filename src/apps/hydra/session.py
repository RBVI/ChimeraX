from .models import Models
class Session(Models):
    '''
    A session holds the list of open models, camera, graphics window, scenes, ...
    all the state.  The main purpose is to bring together all of objects defining
    the 3d world.
    '''
    
    def __init__(self):

        Models.__init__(self)		# Manages list of open models

        from . import ui
        ui.choose_window_toolkit()

        self.quit_callbacks = []        # Callbacks to run before quitting

        self.application = None
        'Qt application object, a QApplication.'

        self.main_window = None
        'Main user interface window, a :py:class:`~.ui.qt.gui.Main_Window.`'

        self.view = None
        'Main window view, a :py:class:`~.graphics.view.View`'

        from .commands import commands, shortcuts
        self.commands = commands.Commands(self)
        'Available commands, a :py:class:`~.commands.commands.Commands`'

        self.keyboard_shortcuts = shortcuts.Keyboard_Shortcuts(self) 
        'Available keyboard shortcuts, a :py:class:`~.command.shortcuts.Keyboard_Shortcuts`'

        self.log = None
        'Command, error, info log, :py:class:`~.ui.qt.gui.Log`'

        self.file_readers = None
        'Table of file types that can be read, used by :py:func:`~.files.opensave.file_readers`'

        self.databases = {}
        'For fetching pdb and map models from the web, used by :py:func:`~.files.fetch.register_fetch_database`'

        from .files import history
        self.file_history = history.File_History(self)
        'Recently opened files, a :py:class:`.files.history.File_History`'

        self.last_session_path = None
        'File path for last opened session.'

        from .files import session_file
        self.object_ids = session_file.Session_Object_Ids()
        'Identifiers for objects to refer to other objects in session files.'

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
        'Space Navigator device handler, a :py:class:`~.devices.spacenavigator.snav.Space_Navigator`'

        self.oculus = None
        'Oculus Rift head tracking device handler, a :py:class:`~.devices.oculus.track.Oculus_Head_Tracking`'

        self.bin_dir = None
        'Location of third party executables used by Hydra.'

    def start(self):

        from . import ui
        import sys
        self.application = app = ui.Hydra_App(sys.argv, self)
        self.main_window = mw = app.main_window
        self.view = v = mw.view
        self.log = log = ui.Log(mw)
        from .commands import commands, shortcuts
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

        # Handle molecule level of detail updates
        from . import molecule
        mlod = molecule.MoleculeLevelOfDetail(self)
        v.add_shape_changed_callback(mlod.update_level_of_detail)

        mw.show()

        status = ui.start_event_loop(app)

        for cb in self.quit_callbacks:
            cb()

        sys.exit(status)

    def show_status(self, msg, append = False):
        '''Show a status message at the bottom of the main window.'''
        self.main_window.show_status(msg, append)
    status = show_status        # Compatibility with Chimera 2

    def show_info(self, msg, color = None):
        '''Write information such as command output to the log window.'''
        self.log.log_message(msg, color)
    info = show_info		# Compatibility with Chimera 2

    def show_warning(self, msg):
        '''Write warning such as command output to the log window.'''
        self.show_status(msg)
        self.show_info(msg, color = 'red')
    warning = show_warning	# Compatibility with Chimera 2

    def executable_directory(self):
         return self.bin_dir

    def at_quit(self, callback):
        '''Register a callback to run just before the program exits, for example to write a history file.'''
        self.quit_callbacks.append(callback)
