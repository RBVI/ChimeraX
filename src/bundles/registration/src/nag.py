# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===


# Timestamps in registration file come from RBVI web server
# which runs with US English locale.  So we use our own
# strftime and strptime that are hardcoded to handle dates
# with English names.

import threading

RegistrationFile = "registration"
UsageFile = "preregistration"
TimeFormat = "%a %b %d %H:%M:%S %Y"
GracePeriod = 14
NagMessage = """You have used ChimeraX %d times over %d days.  Please register your copy by using the Registration tool or the "register" command."""

NagInfo = """Registration is optional and free.  Registration helps us document the impact of ChimeraX on the scientific community. The information you supply will only be used for reporting summary statistics; no individual data will be released.
"""

_registration_lock = threading.Lock()


def nag(session):
    if not session.ui.is_gui:
        return
    if not check_registration(logger=session.logger):
        _check_usage(session)


def install(session, registration):
    with _registration_lock:
        reg_file = _registration_file()
        try:
            with open(reg_file, "w", encoding='utf-8') as f:
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
    param["Expires"] = _strftime(when)
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
                if key in ('Expires', 'Signed'):
                    # show dates according to user's locale
                    t = _strptime(value)
                    value = t.strftime(TimeFormat)
                logger.info("%s: %s" % (key.title(), value))


def _registration_file():
    from chimerax import app_dirs_unversioned
    import os.path
    return os.path.join(app_dirs_unversioned.user_data_dir, RegistrationFile)


def _get_registration(logger):
    with _registration_lock:
        reg_file = _registration_file()
        try:
            param = {}
            with open(reg_file, encoding='utf-8') as f:
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
    from datetime import datetime
    expires = _expiration_time(param)
    if expires is None:
        return None
    if datetime.now() > expires:
        if logger:
            reg_file = _registration_file()
            logger.warning("Registration file %r has expired" % reg_file)
        return None
    return expires


def _expiration_time(param):
    from datetime import timedelta
    try:
        return _strptime(param["Expires"])
    except (KeyError, ValueError):
        pass
    try:
        return _strptime(param["Signed"]) + timedelta(days=365)
    except (KeyError, ValueError):
        pass
    return None


def _write_registration(logger, param):
    if ("Name" not in param or "Email" not in param or
            ("Expires" not in param and "Signed" not in param)):
        raise ValueError("invalid registration data")
    with _registration_lock:
        reg_file = _registration_file()
        try:
            with open(reg_file, "w", encoding='utf-8') as f:
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
        _ask_to_register(session, usage["count"], days)


def _ask_to_register(session, times_used, days_used, wait_for_main_window=True):
    if wait_for_main_window:
        def _delayed_ask(*args, session=session, times_used=times_used, days_used=days_used):
            _ask_to_register(session, times_used, days_used, wait_for_main_window=False)
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        session.triggers.add_handler('new frame', _delayed_ask)
        return
    from chimerax.ui.ask import ask
    answer = ask(session, NagMessage % (times_used, days_used),
                 buttons=["Dismiss", "Register"], info=NagInfo)
    if answer == "Register":
        from chimerax.core.commands import run
        run(session, 'ui tool show Registration')


def _get_usage():
    usage_file = _usage_file()
    usage = {"count": 0, "dates": []}
    try:
        # Read the usage file of count (total number of invocations)
        # and dates (first usage datetime on any day)
        with open(usage_file, encoding='utf-8') as f:
            for line in f:
                if ':' not in line:
                    # protect against corrupted files
                    continue
                key, value = [s.strip() for s in line.split(':', 1)]
                if key == "date":
                    try:
                        date = _strptime(value)
                    except ValueError:
                        # protect against corrupted files
                        from datetime import datetime
                        date = datetime(1, 1, 1)
                    usage["dates"].append(date)
                elif key == "count":
                    usage["count"] = int(value)
    except IOError:
        pass
    return usage


def _write_usage(logger, usage):
    usage_file = _usage_file()
    try:
        with open(usage_file, "w", encoding='utf-8') as f:
            print("count: %d" % usage["count"], file=f)
            for dt in usage["dates"]:
                print("date: %s" % _strftime(dt), file=f)
    except IOError as e:
        logger.error("Cannot write %r: %s" % (usage_file, str(e)))


# Need English version of days and months to match locale of server
_days = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
# months are 1-based in datetime
_months = ('XXX', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


def _strptime(value):
    # convert to datetime
    from datetime import datetime
    try:
        # _, month_name, day, time, year = value.split()
        # workaround earlier bug in _strftime
        values = value.split()
        if len(values) == 5:
            _, month_name, day, time, year = values
        else:
            month_name, day, time, year = values
            month_name = month_name[3:]

        month = _months.index(month_name)
        day = int(day)
        year = int(year)
        hour, minute, second = time.split(':')
        hour = int(hour)
        minute = int(minute)
        second = int(second)
        return datetime(year, month, day, hour, minute, second)
    except Exception:
        try:
            # try current locale
            return datetime.strptime(TimeFormat, value)
        except Exception:
            raise ValueError("time data does not match format")


def _strftime(dt):
    # convert to string
    return f'{_days[dt.weekday()]} {_months[dt.month]} {dt.day} {dt.hour}:{dt.minute}:{dt.second} {dt.year}'
