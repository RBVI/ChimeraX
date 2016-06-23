# vim: set expandtab shiftwidth=4 softtabstop=4:
# -----------------------------------------------------------------------------
# Return a list of atom chains.  An atom chain is a sequence
# of atoms connected by bonds where all non-end-point atoms have exactly 2
# bonds.  A chain is represented by a 2-tuple, the first element being the
# ordered list of atoms, and the second being the ordered list of bonds.
# In a chain which is a cycle all atoms have 2 bonds and the first and
# last atom in the chain are the same.  Non-cycles have end point atoms
# with more or less than 2 bonds.
#
def atom_chains(atoms):

  atom_bonds = {}       # Bonds connecting specified atoms.
  aset = set(atoms)
  for a in atoms:
      atom_bonds[a] = [b for b in a.bonds if b.other_atom(a) in aset]

  used_bonds = {}
  chains = []
  for a in atoms:
    if len(atom_bonds[a]) != 2:
      for b in atom_bonds[a]:
        if not b in used_bonds:
          used_bonds[b] = 1
          c = trace_chain(a, b, atom_bonds)
          chains.append(c)
          end_bond = c[1][-1]
          used_bonds[end_bond] = 1

  #
  # Pick up cycles
  #
  reached_atoms = {}
  for catoms, bonds in chains:
    for a in catoms:
      reached_atoms[a] = 1

  for a in atoms:
    if not a in reached_atoms:
      bonds = atom_bonds[a]
      if len(bonds) == 2:
        b = bonds[0]
        c = trace_chain(a, b, atom_bonds)
        chains.append(c)
        for a in c[0]:
          reached_atoms[a] = 1
      
  return chains
          
# -----------------------------------------------------------------------------
#
def trace_chain(atom, bond, atom_bonds):

  atoms = [atom]
  bonds = [bond]

  a = atom
  b = bond
  while 1:
    a = b.other_atom(a)
    atoms.append(a)
    if a == atom:
      break                     # loop
    blist = list(atom_bonds[a])
    blist.remove(b)
    if len(blist) != 1:
      break
    b = blist[0]
    bonds.append(b)
    
  return (atoms, bonds)
