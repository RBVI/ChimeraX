def download_weights():
    '''Call OpenFold code to fetch the model weights.'''
    from openfold3.entry_points.validator import _maybe_download_parameters, DEFAULT_CACHE_PATH, CHECKPOINT_NAME
    _maybe_download_parameters(DEFAULT_CACHE_PATH / CHECKPOINT_NAME)

download_weights()
