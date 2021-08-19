..  vim: set expandtab shiftwidth=4 softtabstop=4:

.. 
    === UCSF ChimeraX Copyright ===
    Copyright 2016 Regents of the University of California.
    All rights reserved.  This software provided pursuant to a
    license agreement containing restrictions on its disclosure,
    duplication and use.  For details see:
    http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
    This notice must be embedded in or attached to all copies,
    including partial copies, of the software or any revisions
    or derivations thereof.
    === UCSF ChimeraX Copyright ===

Command Style Guide
===================

The overall syntax of a command is:

	*command-name required-positional-arguments optional-positional-arguments keyword-value-arguments*

We will discuss each part of the command syntax in detail in turn.


Command Name
------------

When choosing the name for your command, there are a couple of factors to consider.
One is whether your command, broadly speaking, could be considered a subfunction
of an existing ChimeraX command (such as "color" or "measure").
If so, you could make your a command a subcommand of the existing command, *e.g.* "color *my-command*"
simply by registering your command as such
(*i.e.* making your command name literally "color " followed by your subcommand name).

The other factor to consider for your command name, if it's not a subcommand,
is to choose a name that, while mnemonic, is not overly generic in order to prevent
possible collisions with commands registered by other bundles,
or implemented in the future by ChimeraX itself.
So you might want to prefix or suffix your command name with a short text string
related to your bundle name.
Or if you are implementing multiple commands,
you might want to use that short text string as your "command name"
and have all your commands actually be subcommands of that.

When considering whether to make your command a top-level command or a subcommand,
there are a few additional factors to weigh.
A top-level command is typically less typing.
Subcommands produce less bloat in the documentation command index
and they may be easier to find (*e.g.* "usage color" will list all subcommands)
or remember (*e.g.* all your package's subcommands start with the same top-level command string).
Lastly, it may be easier in some situations to split a command into subcommands
(or split part of its functionality off into a subcommand)
if the command's syntax would otherwise be confusing or awkward.


General Argument Considerations
-------------------------------

We suggest that the arguments to the *Python* function implementing the command be "snake case"
(lower case words connected by underscores)
as per `PEP 8 <https://www.python.org/dev/peps/pep-0008/#function-and-variable-names>`_.
Registering the command arguments uses the same case as the implementing function.
The command-processing machinery will convert snake case to "camel case"
(words directly adjacent with the first letter capitalized for the second and later words)
for presentation in usage messages and when parsing typed commands.
Any documentation you write for your command should use camel case for arguments.

Try to avoid having the user repeatedly type long commands for common operations
by the judicious use of defaults and/or having omitted options use the current or last-used value.
Default values should be what is expected to be most common.

If an argument takes a specific discrete set of possible text-string arguments,
it is better to specify that set using an EnumOf annotation
(*e.g.* EnumOf(["min", "mid", "max"]) rather than a generic StringArg.
Doing so will allow the command parser to check the validity of the value
and expand shortened versions of the value,
and will allow "usage" to show the possible legal values.
Also, your implementing function will have to do less processing of the argument value.


Required Positional Arguments
-----------------------------

These correspond to the mandatory arguments of your implementing function.
If your command has a required atom/model spec, that should be the first positional argument.
You can allow the user to skip specifying a "required" argument by use of the EmptyArg annotation,
typically in conjunction with the Or annotation.
For example, to allow the user to provide a "required" integer argument or skip it,
you would supply Or(IntArg, EmptyArg) as the argument annotation.
If the user skips specifying the argument,
the implementing function will receive None as that arguments value.
Skipping a required positional argument is most frequently employed
if the first argument is an atom/model spec, where skipping it implies "all".


Optional Positional Arguments
-----------------------------

These correspond to the keyword (but not keyword only)
arguments that follow the implementing function's mandatory arguments (in order of declaration).
You can treat some, none, or all such arguments this way,
and the remainder as keyword-value arguments (next section).
The user is allowed to omit these arguments (without using any EmptyArg tricks)
in last-to-first order (*e.g.* you can't omit just a middle argument).
Omitted arguments will receive their declared default value.


Keyword-Value Arguments
-----------------------

These correspond to non-keyword-only keyword arguments not declared as optional positional arguments
as well as all keyword-only arguments of the implementing function.
Any or all of them can be omitted in the typed command, regardless of declaration order.
Omitted arguments will receive their declared default value.

If there is an existing similar command or analogous option, keywords should be analogous,
or if appropriate, the same (*e.g.* "log true" or "save filename|browse").
This will make it easier for the user to remember the syntax of your command.


Less Common Issues
-----------------------

*Multiple required atom/model specs*
    Some commands (*e.g.* "match") require two or more atom/model specs.
    Unfortunately, there is no reliable way to parse adjacent atom/model specs into two (or more) parts,
    and therefore any spec after the first has to be declared as a keyword argument
    (for the "match" command that keyword is "to").
    To ensure that the user supplied the additional atom/model specs despite them being declared as keyword
    arguments, the 'required_arguments' keyword of the CmdDesc constructor can be supplied
    with a list of the required keyword arguments as its value.

*Python-only arguments*
    In some cases you might want to be able to call the command-implementing function with arguments
    not appropriate for exposure to the user-level command.
    In such cases simply omit all mention of such arguments from the CmdDesc constructor.
    Such arguments would have to be keyword arguments —
    the CmdDesc needs to know about all mandatory arguments.

*Alternate boolean keywords*
    In some situations it may be more natural for a boolean value to be indicated
    by the presence or absence of a keyword (and no associated value)
    rather than a keyword plus a true/false value.
    In such cases, use the NoArg annotation for the keyword.
    If the user supplies the keyword,
    your implementing function will be called with that keyword set to True,
    otherwise it gets its declared default value.
