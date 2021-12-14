# -----------------------------------------------------------------------------
#
def read_cmm(session, path):
    """Read Chimera marker model file."""

    f = open(path, 'r')
    msets = load_markerset_xml(session, f)
    f.close()
    mc = sum([m.num_atoms for m in msets], 0)
    return msets, ('Opened %d marker sets containing %d markers' % (len(msets), mc))

# -----------------------------------------------------------------------------
#
def write_cmm(session, path, models = None):
    from .markers import MarkerSet
    if models is None:
        mlist = session.models.list(type = MarkerSet)
    else:
        mlist = [m for m in models if isinstance(m, MarkerSet)]
    f = open(path, 'w')
    f.write(markersets_as_xml(mlist))
    f.close()
    mc = sum([m.num_atoms for m in mlist], 0)
    session.logger.info('Wrote %d marker sets containing %d markers' % (len(mlist), mc))
    
# ---------------------------------------------------------------------------
#
def markerset_as_xml(mset):

  ea = getattr(mset, 'markerset_extra_attributes', {})
  lines = ['<marker_set name="%s"%s>' % (mset.name, attribute_strings(ea))]

  markers = list(mset.atoms)
  markers.sort(key = lambda a: a.residue.number)
  from chimerax.core.colors import rgba8_to_rgba
  for m in markers:
    id_text = 'id="%d"' % m.residue.number
    xyz_text = 'x="%.5g" y="%.5g" z="%.5g"' % tuple(m.scene_coord)

    rgb = rgba8_to_rgba(m.color)[:3]
    if rgb == (1,1,1):
      rgb_text = ''
    else:
      rgb_text = 'r="%.5g" g="%.5g" b="%.5g"' % rgb

    radius_text = 'radius="%.5g"' % m.radius

    if hasattr(m, 'marker_note'):
      note_text = ' note="%s"' % xml_escape(m.marker_note)
      note_rgb = tuple(m.marker_note_rgba)[:3] if hasattr(m, 'marker_note_rgba') else None
      if note_rgb is None:
        note_rgb_text = ''
      else:
        note_rgb_text = ' nr="%.5g" ng="%.5g" nb="%.5g"' % note_rgb
    else:
      note_text = ''
      note_rgb_text = ''

    
    frame_text = ' frame="%d"' % m.frame if hasattr(m, 'frame') else ''
        
    ea = getattr(m, 'marker_extra_attributes', {})

    lines.append('<marker %s %s %s %s%s%s%s%s/>' %
                 (id_text, xyz_text, rgb_text, radius_text,
                  note_text, note_rgb_text, frame_text, attribute_strings(ea)))

  links = mset.bonds
  for e in links:
    m1, m2 = e.atoms
    id_text = 'id1="%d" id2="%d"' % (m1.residue.number, m2.residue.number)
    rgb_text = 'r="%.5g" g="%.5g" b="%.5g"' % rgba8_to_rgba(e.color)[:3]
    radius_text = 'radius="%.5g"' % e.radius
    ea = getattr(e, 'link_extra_attributes', {})
    lines.append('<link %s %s %s%s/>' % (id_text, rgb_text, radius_text,
      				           attribute_strings(ea)))
    
  lines.append('</marker_set>')
  xml = '\n'.join(lines)
  return xml

# -----------------------------------------------------------------------------
#
def markersets_as_xml(mslist):

  lines = []
  if len(mslist) > 1:
    lines.append('<marker_sets>')
    
  for ms in mslist:
      lines.append(markerset_as_xml(ms))

  if len(mslist) > 1:
    lines.append('</marker_sets>')

  xml = '\n'.join(lines)
  return xml
  
# -----------------------------------------------------------------------------
# Make string name1="value1" name2="value2" ... string for XML output.
#
def attribute_strings(dict):

  s = ''
  for name, value in dict.items():
    s = s + (' %s="%s"' % (name, xml_escape(str(value))))
  return s
  
# -----------------------------------------------------------------------------
# Replace & by &amp; " by &quot; and < by &lt; and > by &gt; in a string.
#
def xml_escape(s):

  s1 = s.replace('&', '&amp;')
  s2 = s1.replace('"', '&quot;')
  s3 = s2.replace('<', '&lt;')
  s4 = s3.replace('>', '&gt;')
  s5 = s4.replace("'", '&apos;')
  return s5
  
# -----------------------------------------------------------------------------
#
def load_markerset_xml(session, input):

  # ---------------------------------------------------------------------------
  # Handler for use with Simple API for XML (SAX2).
  #
  from xml.sax import ContentHandler
  class MarkerSetSAXHandler(ContentHandler):

    def __init__(self):

      self.marker_set_tuples = []
      self.set_attributes = None
      self.marker_attributes = None
      self.link_attributes = None

    # -------------------------------------------------------------------------
    #
    def startElement(self, name, attrs):

      if name == 'marker_set':
        self.set_attributes = self.attribute_dictionary(attrs)
        self.marker_attributes = []
        self.link_attributes = []
      elif name == 'marker':
        self.marker_attributes.append(self.attribute_dictionary(attrs))
      elif name == 'link':
        self.link_attributes.append(self.attribute_dictionary(attrs))

    # -------------------------------------------------------------------------
    # Convert Attributes object to a dictionary.
    #
    def attribute_dictionary(self, attrs):

      d = {key:value for key, value in attrs.items()}
      return d

    # -------------------------------------------------------------------------
    #
    def endElement(self, name):

      if name == 'marker_set':
        mst = (self.set_attributes,
	       self.marker_attributes,
	       self.link_attributes)
        self.marker_set_tuples.append(mst)
        self.set_attributes = None
        self.marker_attributes = None
        self.link_attributes = None


  from xml.sax import make_parser
  xml_parser = make_parser()

  from xml.sax.handler import feature_namespaces
  xml_parser.setFeature(feature_namespaces, 0)

  h = MarkerSetSAXHandler()
  xml_parser.setContentHandler(h)
  xml_parser.parse(input)

  msets = create_marker_sets(session, h.marker_set_tuples)
  return msets

# -----------------------------------------------------------------------------
#
def create_marker_sets(session, marker_set_tuples):

  marker_sets = []
  for set_attributes, marker_attributes, link_attributes in marker_set_tuples:
    name = str(set_attributes.get('name', ''))
    from .markers import MarkerSet, create_link
    ms = MarkerSet(session, name)
    ms.markerset_extra_attributes = leftover_keys(set_attributes, ('name',))

    id_to_marker = {}
    have_frame_attr = have_extra_attr = False
    for mdict in marker_attributes:
      id = int(mdict.get('id', '0'))
      x = float(mdict.get('x', '0'))
      y = float(mdict.get('y', '0'))
      z = float(mdict.get('z', '0'))
      r = float(mdict.get('r', '1'))
      g = float(mdict.get('g', '1'))
      b = float(mdict.get('b', '1'))
      radius = float(mdict.get('radius', '1'))
      note = str(mdict.get('note', ''))
      nr = float(mdict.get('nr', '1'))
      ng = float(mdict.get('ng', '1'))
      nb = float(mdict.get('nb', '1'))
      rgba = tuple(int(min(255,max(0,c*255))) for c in (r,g,b,1))
      m = ms.create_marker((x,y,z), rgba, radius, id)
      if 'note' in mdict:
          m.marker_note = str(mdict['note'])
          if 'nr' in mdict and 'ng' in mdict and 'nb' in mdict:
              m.marker_note_rgba = (float(mdict['nr']),float(mdict['ng']),float(mdict['nb']),1)
      if 'frame' in mdict:
          m.frame = int(mdict['frame'])
          have_frame_attr = True
      e = leftover_keys(mdict, ('id','x','y','z','r','g','b', 'radius','note',
				'nr','ng','nb','frame'))
      m.marker_extra_attributes = e
      if e:
          have_extra_attr = True
      id_to_marker[id] = m

    if have_frame_attr:
        ms.save_marker_attribute_in_sessions('frame', int)
    if have_extra_attr:
        ms.save_marker_attribute_in_sessions('marker_extra_attributes')
        
    for ldict in link_attributes:
      if 'id1' not in ldict or 'id2' not in ldict:
          continue
      id1 = int(ldict['id1'])
      id2 = int(ldict['id2'])
      r = float(ldict.get('r', '1'))
      g = float(ldict.get('g', '1'))
      b = float(ldict.get('b', '1'))
      radius = float(ldict.get('radius', '1'))
      rgba = tuple(int(c*255) for c in (r,g,b,1))
      l = create_link(id_to_marker[id1], id_to_marker[id2], rgba, radius)
      e = leftover_keys(ldict, ('id1','id2','r','g','b', 'radius'))
      l.link_extra_attributes = e

    marker_sets.append(ms)

  return marker_sets

# -----------------------------------------------------------------------------
#
def leftover_keys(dict, keys):

  leftover = {}
  leftover.update(dict)
  for k in keys:
    if k in leftover:
      del leftover[k]
  return leftover
