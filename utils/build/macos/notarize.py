#!/usr/bin/env python3
# vim: set expandtab ts=4 sw=4:
import getopt
import json
import os
import plistlib
import subprocess
import sys
import time

import lief


def usage():

    prog = os.path.basename(sys.argv[0])
    print(
        f"""Usage:
  Sign app and create dmg:
      {prog} make [options]
  Submit dmg for verification (prints RequestUUID):
      {prog} submit [options]
  Check notarization progress (look for matching RequestUUID):
      {prog} check [options] [RequestUUID]
  Get notarization result:
      {prog} info [options] RequestUUID
  Finish notarization:
      {prog} finish [options]
  Validate notarization:
      {prog} validate [options]
  Entire process including making, signing and notarization:
      {prog} everything [options]
  Options:
      -a path_to_app
      -d path_to_dmg
      -i APPLE_ID
      -p APP_PASSWORD
      -T TEAM_ID
      -P PROVIDER_NAME
      -k KEYCHAIN
      -K KEYCHAIN_PASSWORD
      -s SIGNER_NAME
      -t MACOSX_DEPLOYMENT_TARGET
      -v
""",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main():

    if len(sys.argv) < 2:
        usage()

    what = sys.argv[1]
    if what == "everything":
        action = sign_and_notarize
    elif what == "make":
        action = sign_binaries_and_make_dmg
    elif what == "sign":
        action = sign_binaries
    elif what == "submit":
        action = request_notarization
    elif what == "info":
        action = report_notarization_result
    elif what == "finish":
        action = staple_notarization_to_dmg
    elif what == "notarize":
        action = notarize_dmg
    elif what == "validate":
        action = validate_notarization
    else:
        print(f'unknown action: "{sys.argv[1]}"')
        usage()

    try:
        opts, args = getopt.getopt(sys.argv[2:], "a:d:f:i:p:P:k:K:s:t:T:v")
    except getopt.GetoptError as e:
        print(str(e), file=sys.stderr)
        usage()

    defaults = Defaults()
    defaults.set_options(opts)
    set_deployment_target(defaults.deployment_target)

    if what == "info":
        if len(args) != 1:
            print(f'"info" needs submission id argument, got {args}', file=sys.stderr)
            usage()
        submission_id = args[0]
        action(defaults, submission_id)
    else:
        action(defaults)


class Defaults:

    def __init__(self):

        self.app_path = "ChimeraX.app"
        self.dmg_path = "ChimeraX.dmg"
        self.dmg_format = "UDZO"
        self.apple_id = None
        self.app_password = None
        self.keychain_path = None
        self.keychain_password = None
        self.team_id = None
        self.signer = None
        self.deployment_target = "11.0"
        self.bundle_id = "edu.ucsf.cgl.ChimeraX"
        # Added entitlements so that the Python ctypes module works,
        # and so the ChimeraX webcam command can access the camera
        self.plist_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "entitlements.plist",
        )
        self.verbose = 3

    def set_options(self, opts):
        for opt, val in opts:
            if opt == "-a":
                self.app_path = val
            elif opt == "-d":
                self.dmg_path = val
            elif opt == "-f":
                self.dmg_format = val
            elif opt == "-i":
                self.apple_id = val
            elif opt == "-p":
                self.app_password = val
            elif opt == "-P":
                self.provider = val
            elif opt == "-T":
                self.team_id = val
            elif opt == "-k":
                self.keychain_path = val
            elif opt == "-K":
                self.keychain_password = val
            elif opt == "-s":
                self.signer = "(" + val + ")"
            elif opt == "-t":
                self.deployment_target = val
            elif opt == "-v":
                self.verbose += 1


def set_deployment_target(target):

    os.environ["MACOSX_DEPLOYMENT_TARGET"] = target


#
# "make" functions
#


def sign_binaries(defaults):

    keychain_unlock(defaults)
    try:
        # sign all executables, dynamic libraries and shared objects
        sign_internals(defaults)
        sign(defaults, [defaults.app_path])
    finally:
        keychain_lock(defaults)


def keychain_unlock(defaults):
    kp = defaults.keychain_path
    cmd = ["/usr/bin/security", "unlock", "-p", defaults.keychain_password, kp]
    cp = _execute(defaults, cmd, f"unlock keychain {kp}")


def keychain_lock(defaults):
    kp = defaults.keychain_path
    cmd = ["/usr/bin/security", "lock", kp]
    cp = _execute(defaults, cmd, f"lock keychain {kp}")


def sign_internals(defaults):
    paths = executable_paths(defaults.app_path)
    sign(defaults, paths)
    if defaults.verbose > 0:
        print("signed", len(paths), "internal binaries")


nonbinary_suffixes = set(
    [
        ".py",
        ".pyc",
        ".pyd",
        ".pyx",
        ".pyi",
        ".png",
        ".jpg",
        ".pgm",
        ".xbm",
        ".gif",
        ".pbm",
        ".ppm",
        ".icns",
        ".webp",
        ".html",
        ".css",
        ".js",
        ".dcm",
        ".csv",
        ".sty",
        ".tex",
        ".tex_t",
        ".pxi",
        ".sip",
        ".svg",
        ".afm",
        ".diag",
        ".txt",
        ".json",
        ".xml",
        ".pdb",
        ".pem",
        ".cfg",
        ".ini",
        ".qm",
        ".xsl",
        ".pxd",
        ".sav",
        ".wav",
        ".aifc",
        ".aiff",
        ".au",
        ".file",
        ".h5",
        ".nc",
        ".dat",
        ".mat",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".f",
        ".f90",
        ".npy",
        ".npz",
        ".gz",
        ".bz2",
        ".po",
        ".mo",
        ".qml",
        ".toml",
        ".whl",
        ".decTest",
        ".ttf",
        ".arff",
        ".pak",
        ".PAR",
        ".md",
        ".bat",
        ".wxs",
        ".DEF",
        ".DAT",
        ".TAB",
        ".TYPE",
        ".plist",
        ".pc",
        ".idatmres",
        ".matrix",
        ".sh",
        ".zip",
        ".tcl",
        ".enc",
        ".msg",
        ".mp4",
        ".ogv",
        ".gitignore",
        ".bild",
        ".cxc",
    ]
)

nonbinary_prefixes = [
    "RECORD",
    "METADATA",
    "WHEEL",
    "INSTALLER",
    "REQUESTED",
    "LICENSE",
    "README",
]


def is_nonbinary(file_name):

    if os.path.splitext(file_name)[1] in nonbinary_suffixes:
        return True
    for prefix in nonbinary_prefixes:
        if file_name.startswith(prefix):
            return True
    return False


def executable_paths(app_path):

    need_signature = set(
        [
            lief.MachO.Header.FILE_TYPE.BUNDLE,
            lief.MachO.Header.FILE_TYPE.DYLIB,
            lief.MachO.Header.FILE_TYPE.EXECUTE,
            lief.MachO.Header.FILE_TYPE.OBJECT,
        ]
    )
    contents_dir = os.path.join(app_path, "Contents")
    macos_dir = os.path.join(app_path, "Contents", "MacOS")
    paths = []
    for dirpath, dirnames, filenames in os.walk(app_path, topdown=False):
        if dirpath == contents_dir:
            dirnames.remove("MacOS")
        if dirpath == macos_dir:
            continue
        for file_name in filenames:
            if is_nonbinary(file_name):
                continue
            file_path = os.path.join(dirpath, file_name)
            if os.path.islink(file_path) or not os.path.isfile(file_path):
                continue
            if not lief.is_macho(file_path):
                continue
            m = lief.MachO.parse(file_path)
            if m is None:
                continue
            if isinstance(m, lief.lief_errors):
                continue
            if m.at(0) is None:
                if file_path.endswith(".a") or file_path.endswith(".o"):
                    # On Mac ARM64 lief fails to parse several .a archives.
                    # With python 3.11 ChimeraX a file config-3.11-darwin/python.o
                    # also is not recognized by lief.  ChimeraX ticket 9148
                    file_type = lief.MachO.Header.FILE_TYPE.OBJECT
                else:
                    continue
            else:
                file_type = m.at(0).header.file_type
            if file_type in need_signature:
                paths.append(file_path)

    return paths


def sign(defaults, paths, tries=5, retry_sleep=30):
    if len(paths) == 0:
        return
    cmd = [
        "/usr/bin/codesign",
        "--keychain",
        defaults.keychain_path,
        "--sign",
        defaults.signer,
        "--options=runtime",  # hardened runtime
        "--timestamp",  # secure timestamp
        f"--entitlements={defaults.plist_path}",  # allow ctypes to work
        "--force",  # in case already signed
        "--strict",
        "--verbose=4",
    ]
    cmd.extend(paths)

    tries -= 1
    try:
        cp = _execute(defaults, cmd, f"signing {len(paths)} files")
    except SystemExit:
        # Try signing multiple times because Apple time server
        # sometimes fails to respond and we do not get a timestamp
        if tries == 0:
            raise
        else:

            print("retrying codesign")
            time.sleep(retry_sleep)
            sign(defaults, paths, tries=tries, retry_sleep=retry_sleep + 30)


def create_dmg(defaults):
    dmg_settings = """{
  "title": "ChimeraX Installer",
  "background": "%s",
  "format": "%s",
  "compression-level": 9,
  "hide-extensions": ["%s"],
  "contents": [
    {
      "x": 190,
      "y": 190,
      "type": "file",
      "path": "%s"
    },
    {
      "x": 594,
      "y": 190,
      "type": "link",
      "path": "/Applications"
    }
  ],
  "window": {
    "position": {
      "x": 200,
      "y": 120
    },
    "size": {
      "width": 800,
      "height": 400
    }
  }
}"""

    with open("dmg-settings.json", "w") as f:
        f.write(
            dmg_settings
            % (
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "macos-dmg-background.png",
                ),
                defaults.dmg_format,
                defaults.app_path.split(os.path.sep)[
                    -1
                ],  # e.g. chimerax/ChimeraX_Daily.app --> ChimeraX_Daily.app
                defaults.app_path,
            )
        )
    cmd = [
        sys.executable,
        "-m",
        "dmgbuild",
        "-s",
        "dmg-settings.json",
        '"ChimeraX Installer"',
        defaults.dmg_path,
    ]
    cp = _execute(defaults, cmd, f"create dmg {defaults.dmg_path}")  # noqa


#
# "submit" functions
#


def request_notarization(defaults, show=True):
    cmd = [
        "/usr/bin/xcrun",
        "notarytool",
        "submit",
        "--apple-id",
        defaults.apple_id,
        "--team-id",
        defaults.team_id,
        "--password",
        defaults.app_password,
        "--output-format",
        "plist",
        "--wait",
        "--timeout",
        "3600s",
        defaults.dmg_path,
    ]
    cp = _execute(defaults, cmd, f"submit notarization request for {defaults.dmg_path}")

    plist = plistlib.loads(cp.stdout)
    return plist


def report_notarization_result(defaults, submission_id):
    cmd = [
        "/usr/bin/xcrun",
        "notarytool",
        "log",
        "--apple-id",
        defaults.apple_id,
        "--team-id",
        defaults.team_id,
        "--password",
        defaults.app_password,
        submission_id,
    ]
    cp = _execute(defaults, cmd, "report notarization status")

    log = json.loads(cp.stdout)

    print("Job:", log["jobId"])
    print("Status:", log["status"])
    print("Summary:", log["statusSummary"])
    print("Archive:", log["archiveFilename"])
    print("Uploaded:", log["uploadDate"])
    if log["statusCode"] != 0:
        report_errors(log)

    return log["statusCode"]


def report_errors(log):
    unsigned = set()
    insecure_timestamp = set()
    unhardened = set()
    others = []
    for issue in log["issues"]:
        message = issue["message"]
        if message == "The binary is not signed.":
            unsigned.add(issue["path"])
        elif message == "The signature does not include a secure timestamp.":
            insecure_timestamp.add(issue["path"])
        elif message == "The executable does not have the hardended runtime enabled.":
            unhardened.add(issue["path"])
        else:
            others.append(issue)
    bad_timestamp = insecure_timestamp - unsigned
    bad_runtime = unhardened - unsigned
    if unsigned:
        print("Unsigned binary:")
        for path in unsigned:
            print("   ", path)
    if bad_timestamp:
        print("Insecure timestamp:")
        for path in bad_timestamp:
            print("   ", path)
    if bad_runtime:
        print("Hardened runtime not enabled:")
        for path in bad_runtime:
            print("   ", path)
    if others:
        msgs = {}
        for issue in others:
            try:
                issues = msgs[issue["message"]]
            except KeyError:
                issues = msgs[issue["message"]] = []
            issues.append(issue)
        for msg, issues in msgs.items():
            print(msg)
            for issue in issues:
                print("   ", issue["severity"], issue["path"])


def staple_notarization_to_dmg(defaults, max_tries=10, initial_delay=10):
    # Apple's notarization ticket may not be immediately available after
    # notarization completes. Retry with exponential backoff.
    cmd = ["/usr/bin/xcrun", "stapler", "staple", "-v", defaults.dmg_path]

    delay = initial_delay
    for attempt in range(1, max_tries + 1):
        if attempt > 1:
            print(f"Waiting {delay}s for notarization ticket to propagate...")
            time.sleep(delay)

        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode == 0:
            if defaults.verbose > 0:
                print(f"staple ticket to {defaults.dmg_path} succeeded")
            if defaults.verbose > 1:
                _show_output(cp)
            break

        print(f"Staple attempt {attempt}/{max_tries} failed")
        if defaults.verbose > 1:
            _show_output(cp)
        delay = min(delay * 2, 120)  # Cap at 2 minutes
    else:
        print(f"staple ticket to {defaults.dmg_path} failed after {max_tries} attempts:")
        _show_output(cp)
        raise SystemExit(1)

    # Validate the stapled ticket
    cmd = ["/usr/bin/xcrun", "stapler", "validate", "-v", defaults.dmg_path]
    cp = _execute(defaults, cmd, f"validate ticket for {defaults.dmg_path}")


def sign_and_notarize(defaults):
    sign_binaries(defaults)
    make_dmg(defaults)
    notarize_dmg(defaults)


def make_dmg(defaults):

    if os.path.exists(defaults.dmg_path):

        os.remove(defaults.dmg_path)
    create_dmg(defaults)


def sign_binaries_and_make_dmg(defaults):
    sign_binaries(defaults)
    make_dmg(defaults)


def notarize_dmg(defaults):
    tries = 0
    status = None
    while tries < 5 and status != 0:
        plist = request_notarization(defaults)
        status = report_notarization_result(defaults, plist["id"])
        if status == 0:
            staple_notarization_to_dmg(defaults)
        else:
            tries += 1
    if status != 0:
        raise RuntimeError(f"Mac notarization failed with status code {status}")


def validate_notarization(defaults):
    cmd = ["/usr/bin/xcrun", "stapler", "validate", "-v", defaults.dmg_path]
    cp = _execute(defaults, cmd, f"validate ticket for {defaults.dmg_path}")
    _show_output(cp)


#
# Utility functions
#


def _execute(defaults, cmd, desc):

    if defaults.verbose > 2:
        print("command:", cmd, flush=True)
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        print(desc, "failed:")
        if defaults.verbose > 0:
            print("command:", cmd)
        _show_output(cp)
        raise SystemExit(1)
    if defaults.verbose > 0:
        print(desc, "succeeded")
    if defaults.verbose > 1:
        _show_output(cp)
    return cp


def _show_output(cp):
    if cp.stdout:
        print("stdout:")
        print(cp.stdout.decode("utf-8"))
    if cp.stderr:
        print("stderr:")
        print(cp.stderr.decode("utf-8"))


if __name__ == "__main__":
    main()
