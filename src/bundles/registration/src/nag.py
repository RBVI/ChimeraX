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
NagMessage = """You have used ChimeraX %d times over %d days.  Please register your copy by using the Registration tool or the "register" command.

Registration is optional and free.  Registration helps us document the impact of ChimeraX on the scientific community. The information you supply will only be used for reporting summary statistics; no individual data will be released.
"""


def nag(session):
    if not session.ui.is_gui:
        return
    if not check_registration(logger=session.logger):
        _check_usage(session)


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
    param = _get_registration(logger)
    if param is None:
        return None
    return _check_expiration(param, logger)


def extend_registration(logger=None, extend_by=None):
    """Extend registration by specified period.

    If user is not registered, no action is taken.

    `extend_by` may be an instance of `datetime.datetime`.
    If no `extend_by` value is given, registration is
    extended to one year from current time.
    """
    param = _get_registration(logger)
    if param is None:
        return
    if not _check_expiration(param, logger):
        return
    from datetime import datetime, timedelta
    if extend_by is None:
        when = datetime.now() + timedelta(days=365)
    elif isinstance(extend_by, datetime):
        when = extend_by
    else:
        raise ValueError("invalid extension period")
    param["Expires"] = datetime.strftime(when, TimeFormat)
    _write_registration(logger, param)


def report_status(logger, verbose):
    param = _get_registration(logger)
    if param:
        expires = _check_expiration(param, logger)
    else:
        expires = None
    if expires is None:
        # Report usage
        usage = _get_usage()
        logger.info("ChimeraX used %d times over %d days" %
                    (usage["count"], len(usage["dates"])))
        if verbose:
            for dt in usage["dates"]:
                logger.info("  %s" % dt.strftime(TimeFormat))
    else:
        # Check expiration
        exp = expires.strftime(TimeFormat)
        from datetime import datetime
        now = datetime.now()
        if expires < now:
            logger.warning("Registration expired (%s)" % exp)
        else:
            logger.info("Registration valid (expires %s)" % exp)
        if verbose:
            for key, value in param.items():
                logger.info("%s: %s" % (key.title(), value))


def _registration_file():
    from chimerax import app_dirs_unversioned
    import os.path
    return os.path.join(app_dirs_unversioned.user_data_dir, RegistrationFile)


def _get_registration(logger):
    reg_file = _registration_file()
    try:
        param = {}
        with open(reg_file) as f:
            for line in f:
                key, value = [s.strip() for s in line.split(':', 1)]
                param[key] = value
    except IOError:
        return None
    if "Name" not in param or "Email" not in param:
        if logger:
            logger.error("Registration file %r is invalid." % reg_file)
        return None
    return param


def _check_expiration(param, logger):
    from datetime import datetime, timedelta
    try:
        expires = datetime.strptime(param["Expires"], TimeFormat)
    except KeyError:
        try:
            signed = param.get("Signed", None)
        except KeyError:
            return None
        expires = datetime.strptime(signed, TimeFormat) + timedelta(year=1)
    if datetime.now() > expires:
        if logger:
            logger.warning("Registration file %r has expired" % reg_file)
        return None
    return expires


def _write_registration(logger, param):
    if ("Name" not in param or "Email" not in param or
        ("Expires" not in param and "Signed" not in param)):
        raise ValueError("invalid registration data")
    reg_file = _registration_file()
    try:
        with open(reg_file, "w") as f:
            for key, value in param.items():
                print("%s: %s" % (key, value), file=f)
    except IOError as e:
        if logger:
            logger.error("%r: %s" % (reg_file, str(e)))


def _usage_file():
    from chimerax import app_dirs_unversioned
    import os.path
    return os.path.join(app_dirs_unversioned.user_data_dir, UsageFile)


def _check_usage(session):
    from datetime import datetime
    usage = _get_usage()
    # Increment count and add date if this is the first invocation
    # of the day.  Then check if it's time to nag.
    usage["count"] += 1
    now = datetime.now()
    today = now.date()
    nagged = True
    for dt in usage["dates"]:
        if dt.date() == today:
            break
    else:
        usage["dates"].append(now)
        nagged = False
    _write_usage(session.logger, usage)
    days = len(usage["dates"])
    if not nagged and days > GracePeriod and session is not None:
        from chimerax.ui.ask import ask
        answer = ask(session, NagMessage % (usage["count"], days),
                     buttons=["Dismiss", "Register"])
        if answer == "Register":
            try:
                session.ui.settings.autostart.append("Registration")
            except AttributeError:
                session.ui.settings.autostart = ["Registration"]


def _get_usage():
    from datetime import datetime
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
        pass
    return usage


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
