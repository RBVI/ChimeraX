# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
utils: Generically useful stuff that doesn't fit elsewhere
==========================================================
"""


# from Mike C. Fletcher's BasicTypes library
# via http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
# Except called flattened, like sorted, since it is nondestructive
def flattened(input, return_types=(list, tuple, set)):
    """Return new flattened version of input

    Parameters
    ----------
    input : a sequence instance (list, tuple, or set)

    Returns
    -------
    A sequence of the same type as the input.
    """
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
