# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
ask: Ask user simple questions
==============================

UI-independent functions for asking users questions.
"""


def ask(session, question, buttons=None, default=None, info=None, title=None):
    # Check/set button options
    if not buttons:
        buttons = ["yes", "no"]
    if not default:
        default = buttons[0]
    # Invoke the appropriate UI
    if session.ui.is_gui:
        return _ask_gui(session, question, buttons, default, info, title)
    else:
        return _ask_nogui(session, question, buttons, default, info, title)


def _ask_nogui(session, question, buttons, default, info, title):
    # title is ignored for nogui mode
    if info:
        print(info)
    print(question, end=" ")
    prompt = '[' + '/'.join(buttons) + ']'
    while True:
        answer = input(prompt)
        if not answer:
            return default
        for b in buttons:
            if b.startswith(answer):
                return b


def _ask_gui(session, question, buttons, default, info, title):
    from PyQt5.QtWidgets import QMessageBox
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    if title:
        msg.setWindowTitle(title)
    if info:
        msg.setInformativeText(info)
    msg.setText(question)
    for b in reversed(buttons):
        gb = msg.addButton(b.capitalize(), QMessageBox.AcceptRole)
    msg.setDefaultButton(gb)
    answer_index = msg.exec_()
    return buttons[-1 - answer_index]


if __name__ == "__main__":
    class Dummy:
        pass
    class Session:
        def __init__(self, is_gui):
            self.ui = Dummy()
            self.ui.is_gui = is_gui
    while True:
        if ask(Session(False), "Really Quit") == "yes":
            break
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtWidgets import QWidget, QPushButton
    def do_test():
        if ask(Session(True), "Really Quit", ["oui", "non"], title="Ask") == "oui":
            QApplication.quit()
    app = QApplication(["ask"])
    w = QWidget()
    b = QPushButton(w)
    b.setText("Test")
    b.move(50, 50)
    b.clicked.connect(do_test)
    w.setWindowTitle("Test ask")
    w.show()
    app.exec_()
