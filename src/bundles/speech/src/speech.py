# vim: set expandtab shiftwidth=4 softtabstop=4:


import re
_re_nopunct = re.compile(r'[^a-zA-Z0-9 ]')


class Speech:

    def __init__(self, session):
        self.session = session
        if not session.ui.is_gui:
            from chimerax.core.errors import UserError
            raise UserError("Speech recognition only works in GUI mode")
        import speech_recognition as sr
        import threading
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
        except OSError as e:
            from chimerax.core.errors import UserError
            raise UserError("Accessing microphone: %s" % str(e))
        self.terminate = None
        self.word_info = None

    def is_active(self):
        return self.terminate is not None

    def state(self):
        return "off" if self.terminate is None else "on"

    def activate(self):
        if self.terminate is not None:
            from chimerax.core.errors import UserError
            raise UserError("speech recognition is already active")
        if self.word_info is None:
            self.word_info = self._scan_menubar()
        r = self.recognizer
        r.energy_threshold = 4000
        s = self.microphone
        with s as source:
            r.adjust_for_ambient_noise(source)
        self.terminate = r.listen_in_background(s, self._transcribe)

    def deactivate(self):
        if self.terminate is None:
            from chimerax.core.errors import UserError
            raise UserError("speech recognition is already inactive")
        self.terminate()
        self.terminate = None

    def show_alternatives(self, alternative):
        alts = self.word_info.show_alternatives(alternative)
        log = self.session.logger.info
        if len(alts) == 0:
            log("There are no alternatives for \"%s\"" % alternative)
        else:
            log("The alternatives for \"%s\" %s: %s" %
                (alternative, "is" if len(alts) == 1 else "are",
                 ", ".join(["\"%s\"" % s for s in alts])))

    def add_alternative(self, alternative, original):
        count = self.word_info.add_alternative(alternative, original)
        log = self.session.logger.info
        log("\"%s\" added in %d place%s" % (alternative, count,
                                            "" if count == 1 else "s"))

    def _scan_menubar(self):
        from Qt.QtWidgets import QMenu, QToolButton
        mw = self.session.ui.main_window
        mb = mw.menuBar()
        word_info = WordInfo(self.session)
        for child in mb.children():
            if isinstance(child, QMenu):
                self._scan_menu(child, word_info)
            elif isinstance(child, QToolButton):
                pass
            else:
                print("Unexpected menubar entry", child)
        word_info.add_singular()
        # word_info.dump()
        return word_info

    def _words(self, s):
        words = _re_nopunct.sub(' ', s.lower().replace('&', '')).split()
        # print("converted", repr(s), "to", words)
        return words

    def _scan_menu(self, menu, parent_info):
        from Qt.QtWidgets import QMenu, QAction
        words = self._words(menu.title())
        name = ' '.join(words)
        # print("menu", name)
        info = parent_info.add_followed_by(words)
        for child in menu.actions():
            if isinstance(child, QAction):
                self._scan_action(child, info)
            else:
                print("--unexpected menu entry", child)

    def _scan_action(self, action, info):
        if action.isSeparator():
            return
        words = self._words(action.text())
        name = ' '.join(words)
        menu = action.menu()
        if menu is not None:
            # print("--submenu", name)
            self._scan_menu(menu, info)
        elif not action.isEnabled():
            # print("--disabled", name)
            pass
        # elif action.isCheckable():
        #     # print("--checkbox", name)
        #     self._add_action_info(info, words, action)
        else:
            # print("--item", name)
            sub_info = info.add_followed_by(words)
            sub_info.set_action(action)

    def _transcribe(self, recognizer, audio):
        import speech_recognition as sr
        try:
            transcription = recognizer.recognize_google(audio)
            # transcription = recognizer.recognize_sphinx(audio)
        except sr.RequestError:
            self._log("speech error: API unavailable")
        except sr.UnknownValueError:
            self._log("speech error: cannot recognize speech")
        else:
            self._log("speech: %s" % transcription)
            self.word_info.execute(transcription.lower())

    def _log(self, msg):
        self.session.ui.thread_safe(self.session.logger.info, msg)


class WordInfo:

    def __init__(self, session):
        self.session = session      # XXX: Should be weakref
        self.followed_by = {}
        self.callback = None

    def add_followed_by(self, words):
        if not words:
            return self
        try:
            word = self.followed_by[words[0]]
        except KeyError:
            word = self.followed_by[words[0]] = WordInfo(self.session)
        return word.add_followed_by(words[1:])

    def add_singular(self):
        # Should use stemming, but this is cheaper
        add = {}
        for word, info in self.followed_by.items():
            if word[-1] == 's':
                singular = word[:-1]
                if singular not in self.followed_by:
                    add[singular] = word
            info.add_singular()
        for singular, plural in add.items():
            self.followed_by[singular] = self.followed_by[plural]

    def show_alternatives(self, alternative):
        originals = set()
        try:
            alt_info = self.followed_by[alternative]
        except KeyError:
            pass
        else:
            for word, info in self.followed_by.items():
                if info is alt_info and word != alternative:
                    originals.add(word)
        for info in self.followed_by.values():
            originals.update(info.show_alternatives(alternative))
        return originals

    def add_alternative(self, alternative, original):
        count = 0
        try:
            orig_info = self.followed_by[original]
        except KeyError:
            pass
        else:
            try:
                alt_info = self.followed_by[alternative]
            except KeyError:
                self.followed_by[alternative] = orig_info
                count += 1
            else:
                if alt_info is not orig_info:
                    # This function is probably called from the main
                    # thread so just raise the exception instead of
                    # printing an error message
                    from chimerax.core.errors import UserError
                    raise UserError("\"%s\" is already in use" % alternative)
        for info in self.followed_by.values():
            count += info.add_alternative(alternative, original)
        return count

    def set_callback(self, cb):
        self.callback = cb

    def set_action(self, action):
        def f(action=action):
            action.activate(action.Trigger)
        self.callback = f

    def dump(self, depth=0):
        if self.callback:
            print("-" * depth, "LEAF")
        for word, info in self.followed_by.items():
            print("-" * depth, word)
            info.dump(depth=depth+1)

    def execute(self, cmd):
        words = cmd.split()
        self._execute_words(words, cmd)

    def _execute_words(self, words, full_cmd):
        if not words:
            if self.callback:
                self.session.ui.thread_safe(self.callback)
            elif len(self.followed_by) == 1:
                for info in self.followed_by.values():
                    info._execute_words(words, full_cmd)
            else:
                self._warning("No action associated with %r" % full_cmd)
            return
        try:
            info = self.followed_by[words[0]]
        except KeyError:
            self._warning("No command associated with %r" % full_cmd)
        else:
            info._execute_words(words[1:], full_cmd)

    def _warning(self, msg):
        self.session.ui.thread_safe(self.session.logger.warning, msg)

    def _error(self, msg):
        self.session.ui.thread_safe(self.session.logger.error, msg)
