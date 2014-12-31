from __future__ import print_function

from distlib.locators import Locator
class ChimeraLocator(Locator):

	def __init__(self, url, **kw):
		Locator.__init__(self, **kw)
		self.__url = url
		self.__dist_cache = None

	@property
	def url(self):
		return self.__url

	def get_project(self, name):
		cache = self._get_cache()
		return cache.get(name, {})

	def get_distribution_names(self):
		cache = self._get_cache()
		return cache.keys()

	def get_distributions(self):
		distributions = []
		cache = self._get_cache()
		for project in cache.values():
			distributions.extend(project)
		return distributions

	def _get_cache(self):
		if self.__dist_cache:
			return self.__dist_cache
		distributions = get_distributions(url=self.__url)
		self.__dist_cache = {}
		for d in distributions:
			name = d.name
			try:
				project = self.__dist_cache[name]
			except KeyError:
				project = {}
				self.__dist_cache[name] = project
			project[d.version] = d
		return self.__dist_cache

def get_distributions(url):
	full_url = url + "/packages/METADATA.json"
	import json
	try:
		from urllib.request import urlopen
	except ImportError:
		from urllib2 import urlopen
	f = urlopen(full_url)
	if f.getcode() != 200:
		raise IOError("request to %s failed" % full_url)
	data = f.read()
	f.close()
	data = json.loads(data.decode(encoding="UTF-8"))
	return _make_distributions(data, url)

def _make_distributions(metadata_list, locator_url):
	from distlib.database import Distribution
	from distlib.metadata import Metadata
	distributions = []
	url = locator_url + "/packages/"
	for md in metadata_list:
		try:
			wheel = md["wheel"]
		except KeyError:
			pass
		else:
			md["source_url"] = url + wheel
			del md["wheel"]
		distributions.append(Distribution(Metadata(mapping=md)))
	return distributions

if __name__ == "__main__":
	if False:
		# Test using local file
		with open("../packages/METADATA.json") as f:
			import json
			distributions = _make_distributions(json.load(f))
		for d in distributions:
			print(d, d.run_requires)
	if False:
		# Test using URL but not Locator
		distributions = get_distributions()
		for d in distributions:
			print(d, d.run_requires)
	if True:
		# Test using ChimeraLocator
		locator = ChimeraLocator("http://localhost:8080")
		#p = locator.get_project("ChimeraCore")
		p = locator.locate("ChimeraCore")
		from pprint import pprint
		pprint(p)
