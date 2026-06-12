def download_weights():
    '''Call OpenFold code to fetch the model weights.'''
    from openfold3.setup_openfold import main
    main(prompt = False, test = False)

download_weights()
