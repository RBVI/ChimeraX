# import os
#
# import pytest
#
# data_dir = os.path.join(os.path.dirname(__file__), "data")
#
# xml_bundle = "xml_test_bundle"
# toml_bundle = "toml_test_bundle"
#
#
# def get_path_to_bundle(bundle):
#    return os.path.join(data_dir, bundle)
#
#
# @pytest.mark.parametrize("bundle", [xml_bundle, toml_bundle])
# def test_metadata_generation(bundle):
#    bpath = get_path_to_bundle(bundle)
#    assert os.path.exists(bpath)
