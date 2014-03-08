#
# Device command for enabling input/output devices such as Oculus Rift, Space Navigator, or Leap Motion.
#
def device_command(cmd_name, args, session):

    from .commands import bool_arg, string_arg, perform_operation
    from .oculus import oculus_command
    from .spacenavigator import snav_command
    from .c2leap import leap_command
    ops = {
        'OculusRift': (oculus_command,
                       (),
                       (('enable', bool_arg),),
                       ()),
        'SpaceNavigator': (snav_command,
                           (),
                           (('enable', bool_arg),),
                           (('fly', bool_arg),)),
        'LeapMotion': (leap_command,
                       (),
                       (('enable', bool_arg),),
                       (('mode', string_arg),)),
    }
    perform_operation(cmd_name, args, ops, session)
