from model import RDF

if __name__ == '__main__':

    # False
    if False:
        rdf = RDF(input_shape=(100, 100, 3), latent_dim=40)
        rdf.train(train_dir='celeba_data/train', val_dir='celeba_data/val', epochs=2)
    else:
        rdf = RDF(input_shape=(100, 100, 3), latent_dim=40)
        rdf.restore_weights()
        rdf.compute_distance('test')