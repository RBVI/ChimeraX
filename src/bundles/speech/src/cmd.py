# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.commands import CmdDesc, BoolArg, StringArg


_sr = None


def speech(session, enable=None):
    """
    Enable speech recognition

    Parameters
    ----------
    enable : bool
      Turn speech recognition on or off.
    """
    global _sr
    if enable is None:
        state = "off" if _sr is None else _sr.state()
        session.logger.info("Speech recognition is %s" % state)
        return
    if enable:
        if _sr is None:
            from .speech import Speech
            _sr = Speech(session)
        _sr.activate()
    else:
        if _sr is not None:
            _sr.deactivate()
speech_desc = CmdDesc(optional=[("enable", BoolArg)])


def speech_alias(session, alternative, original=None):
    """
    Add an alternative to an original command word

    Parameters
    ----------
    alternative : str
       Alternative word.
    original : str
       Original command word.
    """
    if _sr is None:
        from chimerax.core.errors import UserError
        raise UserError("Speech recognition must be initialized first")
    if original is None:
        _sr.show_alternatives(alternative)
    else:
        _sr.add_alternative(alternative, original)
speech_alias_desc = CmdDesc(required=[("alternative", StringArg)],
                            optional=[("original", StringArg)])


command_map = {
    "speech": (speech, speech_desc),
    "speech alias": (speech_alias, speech_alias_desc),
}


def register_command(ci, logger):
    try:
        func, desc = command_map[ci.name]
    except KeyError:
        raise ValueError("trying to register unknown command: %s" % ci.name)
    if desc.synopsis is None:
        desc.synopsis = ci.synopsis
    from chimerax.core.commands import register
    register(ci.name, desc, func, logger=logger)
