heavy_dict = {}
hyd_dict = {}
for file_prefix in ["AUP5", "CGP5", "DATP5", "DCGP5"]:
	charges = {}
	gaffs = {}
	for file_type, attr_dict in [("charge", charges), ("gaff_type", gaffs)]:
		with open(file_prefix + "." + file_type + ".defattr") as f:
			for line in f:
				if line.startswith("chain:"):
					chain_to_res = {}
					fields = line.strip().split()
					chain_to_res[fields[1][:-1]] = fields[2][:-1] + "5PP"
					chain_to_res[fields[3][:-1]] = fields[4] + "5PP"
				elif line.startswith('\t'):
					spec, value = line.strip().split()
					if not value[0].isalpha():
						value = float(value)
					chain_spec, rem = spec.split(':')
					chain = chain_spec[1:]
					res = chain_to_res[chain]
					res_dict = attr_dict.setdefault(res, {})
					ignore, atom = rem.split('@')
					if atom != "HO3'":
						res_dict[atom.lower()] = value
	target_sum = -2.3079
	# need to scale sum of charges to -2.3079
	for res, atom_values in charges.items():
		raw_sum = sum([c for a,c in atom_values.items() if a != "hO3'"])
		#print(f"Non-HO3' charges for {res} sum to:", raw_sum)
		num_adj_atoms = len(atom_values)-1
		adjust = (target_sum - raw_sum) / num_adj_atoms
		#print(f"Adjust {num_adj_atoms} atom charges by {adjust}")
		new_sum = 0.0
		for a, charge in list(atom_values.items()):
			new_charge = float("%g" % (charge+adjust))
			new_sum += new_charge
			atom_values[a] = new_charge
		#print(f"Total for {res} after adjustment is {new_sum} (target: {target_sum})")
	h_name_mapping = {
		"h1": "n1",
		"h1'": "c1'",
		"h2": "c2",
		"h2'": "c2'",
		"h2''": "c2'",
		"h21": "n2",
		"h22": "n2",
		"ho2'": "o2'",
		"h3": "n3",
		"h3'": "c3'",
		"ho3'": "o3'",
		"h4'": "c4'",
		"h41": "n4",
		"h42": "n4",
		"h5": "c5",
		"h5'": "c5'",
		"h5''": "c5'",
		"h6": "c6",
		"h61": "n6",
		"h62": "n6",
		"h71": "c7",
		"h72": "c7",
		"h73": "c7",
		"h8": "c8"
	}
	for res in charges.keys():
		res_charges = charges[res]
		res_gaffs = gaffs[res]
		for a in res_charges.keys():
			charge = res_charges[a]
			gaff = res_gaffs[a]
			if a[0] == 'h':
				heavy_name = h_name_mapping[a]
				hyd_dict[(res, heavy_name)] = (charge, gaff)
			else:
				heavy_dict[(res, a)] = (charge, gaff)
import pprint
print("heavy_data =", pprint.pformat(heavy_dict))
print("hyd_data =", pprint.pformat(hyd_dict))
