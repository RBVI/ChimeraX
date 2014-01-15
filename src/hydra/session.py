class Session:
    '''
    A session holds the list of open models, camera, graphics window, scenes, ...
    all the state.  The main purpose is to bring together all of objects defining
    the 3d world without use of global variables.
    '''
    
    def __init__(self):

        self.application = None		# Qt application object, QApplication object
        self.main_window = None		# Main user interface window, ui.gui.MainWindow object
        self.view = None                # Main window view, ui.view.View object
        from .ui import commands, shortcuts
        self.commands = commands.Commands(self)	# Available commands
        self.keyboard_shortcuts = shortcuts.Keyboard_Shortcuts(self)  # Available keyboard shortcuts
        self.log = None			# Command, error, info log, ui.gui.Log object
        from .file_io import history
        self.file_history = history.File_History(self)	# Recently opened files
        self.last_session_path = None   # File path for last opened session.
        from . import scenes
        self.scenes = scenes.Scenes(self)  # Saved scenes
        self.databases = {}             # For fetching pdb and map models from the web

    def start(self):

        from .ui import gui, commands, shortcuts
        app, mw = gui.create_main_window(self)
        self.application = app
        self.main_window = mw
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

        mw.show()

        gui.start_event_loop(app)

    def show_status(self, msg, append = False):
        '''Show a status message at the bottom of the main window.'''
        self.main_window.show_status(msg, append)

    def show_info(self, msg, color = None):
        '''Write information such as command output to the log window.'''
        self.log.log_message(msg, color)

    def close_models(self, models = None):
        '''
        Close a list of models, or all models if none are specified.
        '''
        v = self.view
        if models is None:
            v.close_all_models()
        else:
            v.close_models(models)
