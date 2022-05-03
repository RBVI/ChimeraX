import glob

from ..pyproject_to_bundle_info import toml_to_dict, main

def pytest_generate_tests(metafunc):
    toml_file_list = glob.glob('src/bundles/*/*.toml')
    bundles = [item.split('/')[-2] for item in toml_file_list]
    metafunc.parametrize("bundle,bundle_toml", zip(bundles, toml_file_list))

def test_conversion(bundle, bundle_toml):
    bundle_info_contents = None
    with open('src/bundles/%s/bundle_info.xml' % bundle) as f:
        bundle_info_contents = f.read()
    converted_bundle_info = main(toml_to_dict(bundle_toml), write=False)
    assert converted_bundle_info == bundle_info_contents
