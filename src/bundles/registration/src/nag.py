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


RegistrationFile = "registration"
UsageFile = "preregistration"
TimeFormat = "%a %b %d %H:%M:%S %Y"
GracePeriod = 14
NagMessage = """You have used ChimeraX for %d times over %d days.
Please register your copy by using the <a href="cxcmd:toolshed show Registration">Registration</a> tool or the "register" command.

Registration is free.  By providing the information
requested, you will be helping us document the impact
this software is having in the scientific community.
The information you supply will only be used for
reporting summary statistics; no individual data
will be released.
"""


def nag(session):
    if not check_registration(logger=session.logger):
        _check_usage(session.logger)

def install(session, registration):
    reg_file = _registration_file()
    try:
        with open(reg_file, "w") as f:
            f.write(registration)
        return True
    except IOError as e:
        session.logger.error("Cannot write registration file %r: %s" %
                             (reg_file, str(e)))
        return False

def check_registration(logger=None):
    """Returns datetime instance for expiration, or None."""
    from datetime import datetime, timedelta
    reg_file = _registration_file()
    try:
        param = {}
        with open(reg_file) as f:
            for line in f:
                key, value = [s.strip() for s in line.split(':', 1)]
                param[key] = value
    except IOError:
        return None
    if "User" not in param or "Email" not in param:
        if logger:
            logger.error("Registration file %r is invalid." % reg_file)
        return None
    try:
        expires = datetime.strptime(param["Expires"], TimeFormat)
    except KeyError:
        t = param.get("Signed", "Wed Sep 28 00:00:00 2010")
        expires = datetime.strptime(t, TimeFormat) + timedelta(year=1)
    if datetime.now() > expires:
        if logger:
            logger.warning("Registration file %r has expired" % reg_file)
        return None
    return expires


def _registration_file():
    from chimerax import app_dirs
    import os.path
    return os.path.join(app_dirs.user_data_dir, RegistrationFile)

def _usage_file():
    from chimerax import app_dirs
    import os.path
    return os.path.join(app_dirs.user_data_dir, UsageFile)

def _check_usage(logger):
    from datetime import datetime, date
    usage_file = _usage_file()
    usage = {"count":0, "dates":[]}
    try:
        # Read the usage file of count (total number of invocations)
        # and dates (first usage datetime on any day)
        with open(usage_file) as f:
            for line in f:
                key, value = [s.strip() for s in line.split(':', 1)]
                if key == "date":
                    usage["dates"].append(datetime.strptime(value, TimeFormat))
                elif key == "count":
                    usage["count"] = int(value)
    except IOError:
        # If there is no usage file, then it must be the first time
        # ChimeraX was invoked
        usage["count"] = 1
        usage["dates"].append(datetime.now())
        _write_usage(logger, usage)
    else:
        # Increment count and add date if this is the first invocation
        # of the day.  Then check if it's time to nag.
        usage["count"] += 1
        now = datetime.now()
        today = now.date()
        for dt in usage["dates"]:
            if dt.date() == today:
                break
        else:
            usage["dates"].append(now)
        _write_usage(logger, usage)
        days = len(usage["dates"])
        if days > GracePeriod:
            logger.info(NagMessage % (usage["count"], days), is_html=True)

def _write_usage(logger, usage):
    from datetime import datetime
    usage_file = _usage_file()
    try:
        with open(usage_file, "w") as f:
            print("count: %d" % usage["count"], file=f)
            for dt in usage["dates"]:
                print("date: %s" % dt.strftime(TimeFormat), file=f)
    except IOError as e:
        logger.error("Cannot write %r: %s" % (usage_file, str(e))) 
