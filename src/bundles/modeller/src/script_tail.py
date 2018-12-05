# Assesses the quality of models using
# the DOPE (Discrete Optimized Protein Energy) method (Shen & Sali 2006)
# and the GA341 method (Melo et al 2002, John & Sali 2003)
a.assess_methods = (assess.GA341, assess.normalized_dope)

# ------------------------- build all models --------------------------
a.make()

# ---------- Accesing output data after modeling is complete ----------

# Get a list of all successfully built models from a.outputs
if loopRefinement:
	ok_models = filter(lambda x: x['failure'] is None, a.loop.outputs)
else:
	ok_models = filter(lambda x: x['failure'] is None, a.outputs)

# Rank the models by index number
#key = 'num'
#ok_models.sort(lambda a,b: cmp(a[key], b[key]))
def numSort(a, b, key="num"):
	return cmp(a[key], b[key])
ok_models.sort(numSort)


# Output the list of ok_models to file ok_models.dat 
fMoutput = open('ok_models.dat', 'w')
fMoutput.write('File name of aligned model\t GA341\t zDOPE \n')

for m in ok_models:
	results  = '%s\t' % m['name']
	results += '%.5f\t' % m['GA341 score'][0]
	results += '%.5f\n' % m['Normalized DOPE score']
	fMoutput.write( results )

fMoutput.close()


