# -----------------------------------------------------------------------------
# Routines to encode a tree of objects as basic Python types.
# The basic types are numbers, strings, None, tuples, lists, dictionaries.
# The purpose of this is to make a representation that is easily saved
# to a file.  The file should be human readable.
#
# The Python pickle module also serializes objects but does not provide
# human readable output.
#

# -----------------------------------------------------------------------------
# A tree of instances and basic Python types is converted to a tree of only
# basic types (numbers, strings, None, tuples, lists, dictionaries).
#
# Instances are replaced by dictionaries which include a key indicating the
# instance class name.  Instance should have an attribute called
# state_attributes set to a sequence of attribute names.  These are the
# only attributes that will be included in the basic tree.
#
# An instance cannot appear more than once in the input tree.
# A dictionary key cannot be an instance.
#
# These two restrictions could be lifted in a more complex implementation
# that makes unique copies of the instances and returns an object that
# gives the tree of basic types with references to unique instances,
# and a list of the unique instances converted to basic types.  This has
# not been done since it is not currently needed.
#
import numpy
basicTypes = (str, None.__class__, int, float, bool,
              numpy.integer, numpy.floating)
def instance_tree_to_basic_tree(itree,
                                already_converted = None,
                                allow_conversion = 1):

  if already_converted == None:
    already_converted = {}
    
  def it2bt(it, c = already_converted, a = allow_conversion):
    return instance_tree_to_basic_tree(it, c, a)
    
  if isinstance(itree, basicTypes):
    return itree
  elif isinstance(itree, (tuple, numpy.ndarray)):
    return tuple(it2bt(it) for it in itree)
  elif isinstance(itree, list):
    return tuple(it2bt(it) for it in itree)
  elif isinstance(itree, dict):
    d = {}
    for key, value in itree.items():
      k = instance_tree_to_basic_tree(key, already_converted,
                                      allow_conversion = 0)
      k = escape_dictionary_key(k)
      v = instance_tree_to_basic_tree(value, already_converted)
      d[k] = v
    return d
  elif hasattr(itree, 'state_attributes') and hasattr(itree, '__class__'):
    if not allow_conversion:
      raise ValueError('Not allowed to convert instance ' + str(itree))
    if itree in already_converted:
      raise ValueError('Instance appears more than once in tree ' + str(itree))
    already_converted[itree] = 1
    c = itree.__class__
    n = c.__name__
    d = {'class': n}
    if hasattr(itree, 'state_attributes'):
      for attr in itree.state_attributes:
        if hasattr(itree, attr):
          k = escape_dictionary_key(attr)
          d[k] = it2bt(getattr(itree, attr))
    return d

  raise ValueError("Can't convert type " + str(type(itree)))
  
# -----------------------------------------------------------------------------
#
def basic_tree_to_instance_tree(btree, name_to_class):

  def bt2it(t, n2c = name_to_class):
    return basic_tree_to_instance_tree(t, n2c)
  
  if isinstance(btree, basicTypes):
    return btree
  elif isinstance(btree, tuple):
    return tuple(bt2it(bt) for bt in btree)
  elif isinstance(btree, list):
    return [bt2it(bt) for bt in btree]
  elif isinstance(btree, dict):
    if 'class' in btree:
      classname = btree['class']
      if not classname in name_to_class:
        raise ValueError('Unknown class ' + classname)
      c = name_to_class[classname]
      i = c()
      for key, value in btree.items():
        if key != 'class':
          k = bt2it(key)
          k = unescape_dictionary_key(k)
          v = bt2it(value)
          setattr(i, k, v)
      return i
    else:
      d = {}
      for key, value in btree.items():
        k = bt2it(key)
        k = unescape_dictionary_key(k)
        v = bt2it(value)
        d[k] = v
      return d

  raise ValueError("Can't convert type " + str(type(btree)))


# -----------------------------------------------------------------------------
# Make sure the word 'class' does not occur as a dictionary key.
# Prefix a key that is 'class' with a backslash.  Also prefix all keys
# that start with a backslash with another backslash.
#
def escape_dictionary_key(key):
  
  if key == 'class' or (isinstance(key, str) and key[:1] == '\\'):
    k = '\\' + key
  else:
    k = key
  return k

# -----------------------------------------------------------------------------
# Remove leading backslash from string keys.
#
def unescape_dictionary_key(key):
  
  if isinstance(key, str) and key[:1] == '\\':
    k = key[1:]
  else:
    k = key
  return k

# -----------------------------------------------------------------------------
# Write a tree involving only basic Python types (numbers, strings, None,
# tuples, lists, dictionaries) to a file formatted for human readability.
# 
def write_basic_tree(btree, file, indent = '', start_of_line = 1):

  if start_of_line:
    sindent = indent
  else:
    sindent = ''

  if isinstance(btree, basicTypes):
    file.write(sindent + repr(btree))
  elif isinstance(btree, (tuple, list, numpy.ndarray)):
    if is_simple_sequence(btree):
      write_simple_sequence(btree, file, sindent)
    else:
      write_sequence(btree, file, indent, sindent)
  elif isinstance(btree, dict):
    if len(btree) == 0:
      file.write(sindent + '{}')
    else:
      file.write(sindent + '{\n')
      keys = list(btree.keys())
      keys.sort()
      for k in keys:
        v = btree[k]
        write_basic_tree(k, file, indent + ' ')
        file.write(': ')
        write_basic_tree(v, file, indent + '  ', start_of_line = 0)
        file.write(',\n')
      file.write(indent + '}')
  else:
    raise ValueError("Can't write type " + str(type(btree)))

# -----------------------------------------------------------------------------
# A simple sequence is of length <= 4 and contains only basic types.
#
def is_simple_sequence(seq):

  if len(seq) > 4:
    return False

  for e in seq:
    if not isinstance(e, basicTypes):
      return False

  return True

# -----------------------------------------------------------------------------
# Write sequence elements on one line.
#
def write_simple_sequence(btree, file, indent):

  if isinstance(btree, tuple):
    open_bracket, close_bracket = '(', ')'
  else:
    open_bracket, close_bracket = '[', ']'

  file.write(indent + open_bracket + ' ')
  for e in btree:
    write_basic_tree(e, file)
    file.write(', ')
  file.write(close_bracket)

# -----------------------------------------------------------------------------
# Write sequence elements on separate lines.
#
def write_sequence(btree, file, indent, sindent):

  if isinstance(btree, (tuple, numpy.ndarray)):
    open_bracket, close_bracket = '(', ')'
  else:
    open_bracket, close_bracket = '[', ']'

  if len(btree) > 100:
    eindent = ''
  else:
    eindent = indent + ' '

  file.write(sindent + open_bracket + '\n')
  for e in btree:
    write_basic_tree(e, file, eindent)
    file.write(',\n')
  file.write(indent + close_bracket)
