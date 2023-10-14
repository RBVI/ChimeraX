# vim: set expandtab ts=4 sw=4:

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

# ------------------------------------------------------------------------------
#
from chimerax.interfaces.graph import Graph
class CrosslinksPlot(Graph):

    help = 'help:user/commands/crosslinks.html#network'
    
    def __init__(self, session, chains, crosslinks):

        nx = sum([len(xl.pseudobonds) for xl in crosslinks], 0)
        ni = sum([c.num_intralinks for c in chains], 0)

        # Create matplotlib panel
        title = '%d Chains %d Crosslinks %d Intra-chain' % (len(chains), nx, ni)
        Graph.__init__(self, session, chains, crosslinks,
                       tool_name = "Crosslinks", title = title)
        self.font_size = 10

        self.chains = chains
        self.crosslinks = crosslinks

        self.draw_graph()

    def mouse_click(self, item, event):
        if item:
            item.select()

    def fill_context_menu(self, menu, item):
        add = lambda *args: self.add_menu_entry(menu, *args)
        if item:
            add('Select ' + item.description(), lambda i=item: i.select())
        add('Lay Out to Match Structure', self.draw_graph)
        add('Save Plot As...', self.save_plot_as)

# ------------------------------------------------------------------------------
#
from chimerax.interfaces.graph import Node
class ChainNode(Node):
    size = 400
    def __init__(self, structure, asym_id):
        self.structure = structure
        self.asym_id = asym_id
        satoms = structure.atoms
        cids = satoms.chain_ids
        self.atoms = satoms.filter(cids == asym_id)
        self.name = asym_id
        self.color = self.atoms.colors.mean(axis=0)/255
        self.num_intralinks = 0
    @property
    def position(self):
        return self.atoms.scene_coords.mean(axis=0)
    def select(self):
        self.structure.session.selection.clear()
        self.atoms.selected = True
    def description(self):
        return '%s %s' % (self.structure.name, self.asym_id)

# ------------------------------------------------------------------------------
#
from chimerax.interfaces.graph import Edge
class ChainCrosslinks(Edge):
    def __init__(self, pseudobonds, chains):
        self.pseudobonds = pseudobonds
        self.nodes = chains
        n = len(pseudobonds)
        self.weight = n
        self.label = '%d' % n
    def select(self):
        pb = self.pseudobonds
        if len(pb) > 0:
            a1,a2 = pb.atoms
            a1[0].structure.session.selection.clear()
            a1.selected = True
            a2.selected = True
            pb.selected = True
    def description(self):
        return '%d Pseudobonds' % len(self.pseudobonds)

# ------------------------------------------------------------------------------
#
def chains_and_edges(pseudobonds):

    # Group pseudobonds by the pair of chains they join
    chain_xlinks = {}
    for pb in pseudobonds:
        chains = [(a.structure, a.residue.chain_id) for a in pb.atoms]
        chains.sort(key = lambda si: (id(si[0]), si[1]))
        chain_xlinks.setdefault(tuple(chains), []).append(pb)

    # Create a chain node for each chain
    nodes = []
    cnodes = {}
    for c2 in chain_xlinks.keys():
        for c in c2:
            if c not in cnodes:
                cnodes[c] = cn = ChainNode(*c)
                nodes.append(cn)

    # Create chain crosslinks
    xlinks = []
    from chimerax.atomic import Pseudobonds
    for (c1,c2), pbonds in chain_xlinks.items():
        cn1, cn2 = cnodes[c1], cnodes[c2]
        if cn1 is cn2:
            n = len(pbonds)
            cn1.name = '%d' % n
            cn1.num_intralinks = n
        else:
            cc = ChainCrosslinks(Pseudobonds(pbonds), (cn1, cn2))
            xlinks.append(cc)
              
    return nodes, xlinks
        
        
