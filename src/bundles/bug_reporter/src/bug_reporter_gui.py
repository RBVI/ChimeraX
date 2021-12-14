# vim: set expandtab shiftwidth=4:

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

BUG_HOST = "www.rbvi.ucsf.edu"
BUG_SELECTOR = "/chimerax/cgi-bin/chimerax_bug_report.py"
BUG_URL = "http://" + BUG_HOST + BUG_SELECTOR

# -----------------------------------------------------------------------------
# User interface for bug reporter.
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class BugReporter(ToolInstance):

    def __init__(self, session, tool_name):
        import locale

        self._ses = session
        
        ToolInstance.__init__(self, session, tool_name)

        from .settings import BugReporterSettings
        self.settings = BugReporterSettings(session, 'Bug Reporter')

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area
        parent.setMinimumWidth(600)

        from Qt.QtWidgets import QGridLayout, QLabel, QPushButton, QLineEdit, QTextEdit
        from Qt.QtWidgets import QWidget, QHBoxLayout, QCheckBox
        from Qt.QtCore import Qt
        
        layout = QGridLayout(parent)
        layout.setContentsMargins(3,3,3,3)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(3)
        parent.setLayout(layout)

        row = 1
        
        intro = '''
        <center><h1>Report a Bug</h1></center>
        <p>Thank you for using our feedback system.
      Feedback is greatly appreciated and plays a crucial role
      in the development of ChimeraX.</p>
      <p><b>Note</b>:
          We do not automatically collect any personal information or the data
          you were working with when the problem occurred.  Providing your e-mail address is optional,
          but will allow us to inform you of a fix or to ask questions, if needed.
          Attaching data may also be helpful.  However, any information or data
          you wish to keep confidential should be sent separately (not using this form).</p>
        '''
        il = QLabel(intro)
        il.setWordWrap(True)
        layout.addWidget(il, row, 1, 1, 2)
        row += 1
        
        cnl = QLabel('Contact Name:')
        align_right = Qt.AlignRight|Qt.AlignVCenter
        cnl.setAlignment(align_right)
        layout.addWidget(cnl, row, 1)
        self.contact_name = cn = QLineEdit(self.settings.contact_name)
        layout.addWidget(cn, row, 2)
        row += 1

        eml = QLabel('Email Address:')
        eml.setAlignment(align_right)
        layout.addWidget(eml, row, 1)
        self.email_address = em = QLineEdit(self.settings.email_address)
        layout.addWidget(em, row, 2)
        row += 1

        class TextEdit(QTextEdit):
            def __init__(self, text, initial_line_height):
                self._lines = initial_line_height
                QTextEdit.__init__(self, text)
            def sizeHint(self):
                from Qt.QtCore import QSize
                fm = self.fontMetrics()
                h = self._lines * fm.lineSpacing() + fm.ascent()
                size = QSize(-1, h)
                return size
            def minimumSizeHint(self):
                from Qt.QtCore import QSize
                return QSize(1,1)

        dl = QLabel('Description:')
        dl.setAlignment(align_right)
        layout.addWidget(dl, row, 1)
        self.description = d = TextEdit('', 3)
        d.setText('<font color=blue>(Describe the actions that caused this problem to occur here)</font>')
        layout.addWidget(d, row, 2)
        row += 1

        gil = QLabel('Gathered Information:')
        gil.setAlignment(align_right)
        layout.addWidget(gil, row, 1)
        self.gathered_info = gi = TextEdit('', 3)
        import sys
        info = self.opengl_info()
        if sys.platform == 'win32':
            info += _win32_info()
        elif sys.platform == 'linux':
            info += _linux_info()
        elif sys.platform == 'darwin':
            info += _darwin_info()
        info += f"Locale: {locale.getdefaultlocale()}\n"
        info += _qt_info(session)
        info += _package_info()
        gi.setText(info)
        layout.addWidget(gi, row, 2)
        row += 1

        fal = QLabel('File Attachment:')
        fal.setAlignment(align_right)
        layout.addWidget(fal, row, 1)
        fb = QWidget()
        layout.addWidget(fb, row, 2)
        fbl = QHBoxLayout(fb)
        fbl.setSpacing(3)
        fbl.setContentsMargins(0,0,0,0)
        self.attachment = fa = QLineEdit('')
        fbl.addWidget(fa)
        fab = QPushButton('Browse')
        fab.clicked.connect(lambda e: self.file_browse())
        fbl.addWidget(fab)
        
        row += 1
        
        pl = QLabel('Platform:')
        pl.setAlignment(align_right)
        layout.addWidget(pl, row, 1)
        from platform import platform
        self.platform = p = QLineEdit(platform())
        p.setReadOnly(True)
        layout.addWidget(p, row, 2)
        row += 1
        
        vl = QLabel('ChimeraX Version:')
        vl.setAlignment(align_right)
        layout.addWidget(vl, row, 1)
        self.version = v = QLineEdit(self.chimerax_version())
        v.setReadOnly(True)
        layout.addWidget(v, row, 2)
        row += 1
        
        il = QWidget()
        layout.addWidget(il, row, 2)
        ilayout = QHBoxLayout(il)
        ilayout.setContentsMargins(0,0,0,0)
        self.include_log = ilc = QCheckBox()
        ilc.setChecked(True)
        ilayout.addWidget(ilc)
        ill = QLabel('Include log contents in bug report')
        align_left = Qt.AlignLeft|Qt.AlignVCenter
        ill.setAlignment(align_left)
        ilayout.addWidget(ill)
        ilayout.addStretch(1)
        row += 1

        self.result = rl = QLabel('')
        rl.setWordWrap(True)
        layout.addWidget(rl, row, 1, 1, 2)
        row += 1

        # Button row
        brow = QWidget()
        blayout = QHBoxLayout()
        blayout.setContentsMargins(0,0,0,0)
        blayout.setSpacing(2)
        brow.setLayout(blayout)
        layout.addWidget(brow, row, 1, 1, 2)
        row += 1

        blayout.addStretch(1)    # Extra space at start of button row.
        
        self.submit_button = sb = QPushButton('Submit', brow)
        sb.clicked.connect(lambda e: self.submit())
        blayout.addWidget(sb)
        
        self.cancel_button = cb = QPushButton('Cancel', brow)
        cb.clicked.connect(lambda e: self.cancel())
        blayout.addWidget(cb)

        tw.manage(placement=None)  # Don't dock it to main window.

    def show(self):
        self.tool_window.shown = True

    def hide(self):
        self.tool_window.shown = False

    def set_description(self, text):
        self.description.setText(text)

    def submit(self):

        entry_values = self.entry_values()

        # Include log contents in description
        if self.include_log.isChecked():
            from chimerax.log.cmd import get_singleton
            log = get_singleton(self._ses)
            if log:
                log_text = "\n\nLog:\n%s\n" % log.plain_text()
                entry_values['description'] += log_text

        # Include info field in description
        info = entry_values['info']
        if info:
            entry_values['description'] += "\n\n" + info
        del entry_values['info']

        # Set form data
        from .form_fields import hidden_attrs
        my_attrs = hidden_attrs.copy()
        for hk, hv in my_attrs.items():
            for k, v in entry_values.items():
                if k.upper() + "_VAL" == hv:
                    my_attrs[hk] = v

        # Add attachment to form data.
        file_list = self.read_attachment(entry_values['filename'])
        fields = [(k, None, v) for k,v in my_attrs.items()]
        fields.extend(file_list)

        # Post form data.
        self.status("Contacting CGL....", color="blue")
        from chimerax.webservices.post_form import post_multipart_formdata
        import socket
        try:
            errcode, errmsg, headers, body = post_multipart_formdata(BUG_HOST, BUG_SELECTOR, fields, timeout=10)
        except socket.gaierror:
            # Not connected to internet or hostname unknown.
            msg = 'Possibly no internet connection.'
            self._ses.logger.warning('Failed to send bug report. %s' % msg)
            self.report_failure(msg)
            return
        except (TimeoutError, socket.timeout):
            # Host did not respond.
            msg = 'Bug report server %s is unavailable' % BUG_HOST
            self._ses.logger.warning('Failed to send bug report. %s' % msg)
            self.report_failure(msg)
            return
        except Exception:
            self._ses.logger.warning('Failed to send bug report. Error while sending follows:')
            import traceback
            traceback.print_exc()	# Log detailed exception info
            self.report_failure()
            return

        # Report success or error.
        if int(errcode) == 200:
            self.report_success()
            self.cancel_button.setText("Close")
            self.submit_button.deleteLater()	# Prevent second submission
            s = self.settings
            s.contact_name = self.contact_name.text()
            s.email_address = self.email_address.text()
        else:
            self.report_failure()

    def read_attachment(self, file_path):
        if file_path:
            import os, os.path
            if not os.path.isfile(file_path):
                self.status("Couldn't locate file '%s'."
                            "  Please choose a valid file."
                            % os.path.split(file_path)[1], color='red')
                return
            try:
                file_content = open(file_path, 'rb').read()
            except IOError as what:
                error_msg = "Couldn't read file '%s' (%s)" % (
                    os.path.split(file_path)[1],
                    os.strerror(what.errno))
                self.status(error_msg, color='red')
                return
            from .form_fields import FILE_FIELD
            file_list = [(FILE_FIELD, os.path.split(file_path)[1], file_content)]
        else:
            file_list = []
        return file_list

    def status(self, text, color = None):
        if color is not None:
            text = '<font color="%s">%s</font>' % (color, text)
        self.result.setText(text)
        
    def report_success(self):
        thanks = ("<font color=blue><h3>Thank you for your report.</h3></font>"
                "<p>Your report will be evaluated by a Chimera developer"
                " and if you provided an e-mail address,"
                " then you will be contacted with a report status.")
        self.result.setText(thanks)

    def report_failure(self, reason = None):
        detail = ('<p>%s</p>' % reason) if reason else ''
        
        oops = ("<font color=red><h3>Error while submitting feedback.</h3></font>"
                + detail +
                "<p>An error occured when trying to submit your feedback."
                "  No information was received by the Computer Graphics Lab."
                "  This could be due to network problems, but more likely,"
                " there is a problem with Computer Graphics Lab's server."
                "  Please report this problem by sending email to"
                " <font color=\"blue\">chimerax-bugs@cgl.ucsf.edu</font>"
                " and paste a copy of the ChimeraX log into the email.</p>"
                "<p>We apologize for any inconvenience, and do appreciate"
                " you taking the time to provide us with valuable feedback.")
        self.result.setText(oops)

    def cancel(self):
        self.delete()

    def file_browse(self):
        from Qt.QtWidgets import QFileDialog
        path,type = QFileDialog.getOpenFileName()
        if path:
            self.attachment.setText(path)

    def opengl_info(self):
        r = self._ses.main_view.render
        try:
            r.make_current()
            lines = ['OpenGL version: ' + r.opengl_version(),
                     'OpenGL renderer: ' + r.opengl_renderer(),
                     'OpenGL vendor: ' + r.opengl_vendor()]
            r.done_current()
        except Exception:
            lines = ['OpenGL version: unknown',
                     'Could not make opengl context current']
        return '\n'.join(lines)

    def chimerax_version(self):
        from chimerax.core.buildinfo import version, date
        return '%s (%s)' % (version, date)

    def entry_values(self):
        values = {
            'name': self.contact_name.text(),
            'email': self.email_address.text(),
            'description': self.description.toPlainText(),
            'info': self.gathered_info.toPlainText(),
            'filename': self.attachment.text(),
            'platform': self.platform.text(),
            'version': self.version.text()
        }
        return values

def show_bug_reporter(session):
    tool_name = 'Bug Reporter'
    tool = BugReporter(session, tool_name)
    return tool

def add_help_menu_entry(session):
    ui = session.ui
    if ui.is_gui:
        def main_window_created(tname, tdata):
            mw = ui.main_window
            mw.add_menu_entry(['Help'], 'Report a Bug', lambda: show_bug_reporter(session),
                insertion_point = "Contact Us")
            return "delete handler"
    
        ui.triggers.add_handler('ready', main_window_created)


# from https://docs.microsoft.com/en-us/openspecs/office_standards/ms-oe376/6c085406-a698-4e12-9d4d-c3b0ee3dbc4a
OS_LANGUAGES = {
    1025: "ar-SA",
    1026: "bg-BG",
    1027: "ca-ES",
    1028: "zh-TW",
    1029: "cs-CZ",
    1030: "da-DK",
    1031: "de-DE",
    1032: "el-GR",
    1033: "en-US",
    1034: "es-ES",
    1035: "fi-FI",
    1036: "fr-FR",
    1037: "he-IL",
    1038: "hu-HU",
    1039: "is-IS",
    1040: "it-IT",
    1041: "ja-JP",
    1042: "ko-KR",
    1043: "nl-NL",
    1044: "nb-NO",
    1045: "pl-PL",
    1046: "pt-BR",
    1047: "rm-CH",
    1048: "ro-RO",
    1049: "ru-RU",
    1050: "hr-HR",
    1051: "sk-SK",
    1052: "sq-AL",
    1053: "sv-SE",
    1054: "th-TH",
    1055: "tr-TR",
    1056: "ur-PK",
    1057: "id-ID",
    1058: "uk-UA",
    1059: "be-BY",
    1060: "sl-SI",
    1061: "et-EE",
    1062: "lv-LV",
    1063: "lt-LT",
    1064: "tg-Cyrl-TJ",
    1065: "fa-IR",
    1066: "vi-VN",
    1067: "hy-AM",
    1068: "az-Latn-AZ",
    1069: "eu-ES",
    1070: "wen-DE",
    1071: "mk-MK",
    1072: "st-ZA",
    1073: "ts-ZA",
    1074: "tn-ZA",
    1075: "ven-ZA",
    1076: "xh-ZA",
    1077: "zu-ZA",
    1078: "af-ZA",
    1079: "ka-GE",
    1080: "fo-FO",
    1081: "hi-IN",
    1082: "mt-MT",
    1083: "se-NO",
    1084: "gd-GB",
    1085: "yi",
    1086: "ms-MY",
    1087: "kk-KZ",
    1088: "ky-KG",
    1089: "sw-KE",
    1090: "tk-TM",
    1091: "uz-Latn-UZ",
    1092: "tt-RU",
    1093: "bn-IN",
    1094: "pa-IN",
    1095: "gu-IN",
    1096: "or-IN",
    1097: "ta-IN",
    1098: "te-IN",
    1099: "kn-IN",
    1100: "ml-IN",
    1101: "as-IN",
    1102: "mr-IN",
    1103: "sa-IN",
    1104: "mn-MN",
    1105: "bo-CN",
    1106: "cy-GB",
    1107: "km-KH",
    1108: "lo-LA",
    1109: "my-MM",
    1110: "gl-ES",
    1111: "kok-IN",
    1112: "mni",
    1113: "sd-IN",
    1114: "syr-SY",
    1115: "si-LK",
    1116: "chr-US",
    1117: "iu-Cans-CA",
    1118: "am-ET",
    1119: "tmz",
    1120: "ks-Arab-IN",
    1121: "ne-NP",
    1122: "fy-NL",
    1123: "ps-AF",
    1124: "fil-PH",
    1125: "dv-MV",
    1126: "bin-NG",
    1127: "fuv-NG",
    1128: "ha-Latn-NG",
    1129: "ibb-NG",
    1130: "yo-NG",
    1131: "quz-BO",
    1132: "nso-ZA",
    1136: "ig-NG",
    1137: "kr-NG",
    1138: "gaz-ET",
    1139: "ti-ER",
    1140: "gn-PY",
    1141: "haw-US",
    1142: "la",
    1143: "so-SO",
    1144: "ii-CN",
    1145: "pap-AN",
    1152: "ug-Arab-CN",
    1153: "mi-NZ",
    2049: "ar-IQ",
    2052: "zh-CN",
    2055: "de-CH",
    2057: "en-GB",
    2058: "es-MX",
    2060: "fr-BE",
    2064: "it-CH",
    2067: "nl-BE",
    2068: "nn-NO",
    2070: "pt-PT",
    2072: "ro-MD",
    2073: "ru-MD",
    2074: "sr-Latn-CS",
    2077: "sv-FI",
    2080: "ur-IN",
    2092: "az-Cyrl-AZ",
    2108: "ga-IE",
    2110: "ms-BN",
    2115: "uz-Cyrl-UZ",
    2117: "bn-BD",
    2118: "pa-PK",
    2128: "mn-Mong-CN",
    2129: "bo-BT",
    2137: "sd-PK",
    2143: "tzm-Latn-DZ",
    2144: "ks-Deva-IN",
    2145: "ne-IN",
    2155: "quz-EC",
    2163: "ti-ET",
    3073: "ar-EG",
    3076: "zh-HK",
    3079: "de-AT",
    3081: "en-AU",
    3082: "es-ES",
    3084: "fr-CA",
    3098: "sr-Cyrl-CS",
    3179: "quz-PE",
    4097: "ar-LY",
    4100: "zh-SG",
    4103: "de-LU",
    4105: "en-CA",
    4106: "es-GT",
    4108: "fr-CH",
    4122: "hr-BA",
    5121: "ar-DZ",
    5124: "zh-MO",
    5127: "de-LI",
    5129: "en-NZ",
    5130: "es-CR",
    5132: "fr-LU",
    5146: "bs-Latn-BA",
    6145: "ar-MO",
    6153: "en-IE",
    6154: "es-PA",
    6156: "fr-MC",
    7169: "ar-TN",
    7177: "en-ZA",
    7178: "es-DO",
    7180: "fr-029",
    8193: "ar-OM",
    8201: "en-JM",
    8202: "es-VE",
    8204: "fr-RE",
    9217: "ar-YE",
    9225: "en-029",
    9226: "es-CO",
    9228: "fr-CG",
    10241: "ar-SY",
    10249: "en-BZ",
    10250: "es-PE",
    10252: "fr-SN",
    11265: "ar-JO",
    11273: "en-TT",
    11274: "es-AR",
    11276: "fr-CM",
    12289: "ar-LB",
    12297: "en-ZW",
    12298: "es-EC",
    12300: "fr-CI",
    13313: "ar-KW",
    13321: "en-PH",
    13322: "es-CL",
    13324: "fr-ML",
    14337: "ar-AE",
    14345: "en-ID",
    14346: "es-UY",
    14348: "fr-MA",
    15361: "ar-BH",
    15369: "en-HK",
    15370: "es-PY",
    15372: "fr-HT",
    16385: "ar-QA",
    16393: "en-IN",
    16394: "es-BO",
    17417: "en-MY",
    17418: "es-SV",
    18441: "en-SG",
    18442: "es-HN",
    19466: "es-NI",
    20490: "es-PR",
    21514: "es-US",
    58378: "es-419",
    58380: "fr-015",
}


def _win32_info():
    try:
        import wmi
        w = wmi.WMI()
        pi = w.CIM_Processor()[0]
        osi = w.CIM_OperatingSystem()[0]
        os_name = osi.Name.split('|', 1)[0]
        csi = w.Win32_ComputerSystem()[0]
        lang = OS_LANGUAGES.get(osi.OSLanguage, osi.OSLanguage)
        info = f"""
Manufacturer: {csi.Manufacturer}
Model: {csi.Model}
OS: {os_name} (Build {osi.BuildNumber})
Memory: {int(csi.TotalPhysicalMemory):,}
MaxProcessMemory: {int(osi.MaxProcessMemorySize):,}
CPU: {pi.NumberOfLogicalProcessors} {pi.Name}
OSLanguage: {lang}
"""
        return info
    except Exception:
        return ""


def _linux_info():
    import distro
    import platform
    import subprocess
    count = 0
    model_name = ""
    cache_size = ""
    virtual_machine = ""
    try:
        p = subprocess.run(
            ["systemd-detect-virt"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="UTF-8",
            env={
                "LANG": "en_US.UTF-8",
                "PATH": "/sbin:/usr/sbin:/bin:/usr/bin",
            })
        # ignore return code
        virtual_machine = p.stdout.split('\n', 1)[0]
    except Exception:
        # TODO: find other non-root methods to try 
        virtual_machine = "detection failed"
    try:
        with open("/proc/cpuinfo", encoding='utf-8') as f:
            for line in f.readlines():
                if not model_name and line.startswith("model name"):
                    info = line.split(':', 1)
                    if len(info) > 1:
                        model_name = info[1].strip()
                elif line.startswith("processor"):
                    count += 1
                elif not cache_size and line.startswith("cache size"):
                    info = line.split(':', 1)
                    if len(info) > 1:
                        cache_size = info[1].strip()
    except Exception:
        pass
    if not model_name:
        model_name = "unknown"
    if not cache_size:
        cache_size = "unknown"

    memory_info = ""
    try:
        output = subprocess.check_output(
            ["free", "-h"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            encoding="UTF-8",
            env={
                "LANG": "en_US.UTF-8",
                "PATH": "/sbin:/usr/sbin:/bin:/usr/bin",
            })
        lines = output.rstrip().split('\n')
        for line in lines:
            memory_info += f"\t{line}\n"
    except Exception:
        try:
            with open("/proc/meminfo", encoding='utf-8') as f:
                for line in f.readlines():
                    if line.startswith("Mem"):
                        memory_info += f"\t{line}"
        except Exception:
            memory_info = "\tunknown"

    try:
        output = subprocess.check_output(
            ["lspci", "-nnk"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            encoding="UTF-8",
            env={
                "LANG": "en_US.UTF-8",
                "PATH": "/sbin:/usr/sbin:/bin:/usr/bin",
            })
        lines = iter(output.split('\n'))
        for line in lines:
            if "VGA compatible" in line:
                break
        graphics_info = '\t\n'.join([line, next(lines), next(lines)])
    except Exception as e:
        graphics_info = "unknown"

    dmi_prefix = "/sys/devices/virtual/dmi/id/"
    try:
        vendor = open(dmi_prefix + "sys_vendor", encoding='utf-8').read().strip()
    except Exception:
        vendor = "unknown"

    try:
        product = open(dmi_prefix + "product_name", encoding='utf-8').read().strip()
    except Exception:
        product = "unknown"

    info = f"""
Manufacturer: {vendor}
Model: {product}
OS: {' '.join(distro.linux_distribution())}
Architecture: {' '.join(platform.architecture())}
Virutal Machine: {virtual_machine}
CPU: {count} {model_name}
Cache Size: {cache_size}
Memory:
{memory_info}
Graphics:
\t{graphics_info}
"""
    return info


def _darwin_info():
    import subprocess
    try:
        output = subprocess.check_output([
                "system_profiler",
                "-detailLevel", "mini",
                "SPHardwareDataType",
                "SPSoftwareDataType",
                "SPDisplaysDataType",
            ],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            encoding="UTF-8",
            env={
                "LANG": "en_US.UTF-8",
                "PATH": "/sbin:/usr/sbin:/bin:/usr/bin",
            })
        return output
    except Exception:
        return ""


def _qt_info(session):
    if not session.ui.is_gui:
        return ""
    import Qt
    return Qt.version + '\n'


def _package_info():
    import pkg_resources
    dists = list(pkg_resources.WorkingSet())
    dists.sort(key=lambda d: d.project_name.casefold())

    info = "Installed Packages:"
    for d in dists:
        name = d.project_name
        if d.has_version():
            version = d.version
        else:
            version = "unknown"
        info += f"\n    {name}: {version}"
    return info
