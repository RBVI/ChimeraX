"""
utils -- generically useful stuff that doesn't fit elsewhere
"""


# from Mike C. Fletcher's BasicTypes library
# via http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
# Except called flattened, like sorted, since it is nondestructive
def flattened(input, return_types=(list, tuple, set)):
    """Return new flattened version of input"""
    return_type = type(input)
    output = list(input)
    i = 0
    while i < len(output):
        while isinstance(output[i], return_types):
            if not output[i]:
                output.pop(i)
                i -= 1
                break
            else:
                output[i:i + 1] = output[i]
        i += 1
    if return_type == list:
        return output
    return return_type(output)


def html_user_agent(app_dirs):
    """"Return HTML User-Agent header according to RFC 2068"""
    # The name, author, and version must be "tokens"
    # 
    #   token          = 1*<any CHAR except CTLs or tspecials>
    #   CTLs           = <any US-ASCII control character
    #                     (octets 0 - 31) and DEL (127)>
    #   tspecials      = "(" | ")" | "<" | ">" | "@"
    #                    | "," | ";" | ":" | "\" | <">
    #                    | "/" | "[" | "]" | "?" | "="
    #                    | "{" | "}" | SP | HT
    #   comment        = "(" *( ctext | comment ) ")"
    #   ctext          = <any TEXT excluding "(" and ")">
    #   TEXT           = <any OCTET except CTLs,
    #                     but including LWS>
    #   LWS            = [CRLF] 1*( SP | HT )

    ctls = ''.join(char(x) for x in range(32)) + chr(127)
    tspecials = '()<>@,;:\"/[]?={} \t'
    bad = ctls + tspecials

    def token(text):
        return ''.join([c for c in text if c not in bad])
    def comment(text):
        # TODO: check for matched parenthesis
        # TODO: strip appropriate CTLs
        return text

    app_author = app_dirs.appauthor
    app_name = app_dirs.appname
    app_version = app_dirs.version

    user_agent = ''
    if app_author is not None:
        user_agent = "%s-" % token(app_author)
    user_agent += token(app_name)
    if app_version is not None:
        user_agent += "/%s" % token(app_version)
    import platform
    system = platform.system()
    if system:
        user_agent += " (%s)" % comment(system)
    return user_agent


