import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

path_separator = os.path.sep
# Environment settings
IS_COLAB = (os.name == 'posix')
LOAD_DATA = not (os.name == 'posix')
IS_EXPERIMENT = False
train_name = 'train'
test_name = 'test'
WIDTH = HEIGHT = 105
CEELS = 1
loss_type = "binary_crossentropy"
validation_size = 0.2
early_stopping = True

if IS_COLAB:
    # the google drive folder we used
    data_path = os.path.sep + os.path.join('content', 'drive', 'My\ Drive', 'datasets', 'lfw2').replace('\\', '')
else:
    # locally
    from data_loader import DataLoader
    from siamese_network import SiameseNetwork

    data_path = os.path.join('lfwa', 'lfw2')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_combination(l, bs, ep, pat, md, seed, train_path, test_path):
    """
    This function gets the parameters and run the experiment.
    :return: loss - loss on the testing set, accuracy - accuracy on the testing set
    """
    # file types
    model_save_type = 'h5'
    # files paths
    initialize_seed(seed)
    parameters_name = f'seed_{seed}_lr_{l}_bs_{bs}_ep_{ep}_val_{validation_size}_' \
                      f'es_{early_stopping}_pa_{pat}_md_{md}'
    print(f'Running combination with {parameters_name}')
    # A path for the weights
    load_weights_path = os.path.join(data_path, 'weights', f'weights_{parameters_name}.{model_save_type}')

    siamese = SiameseNetwork(seed=seed, width=WIDTH, height=HEIGHT, cells=CEELS, loss=loss_type, metrics=['accuracy'],
                             optimizer=Adam(lr=l), dropout_rate=0.4)
    siamese.fit(weights_file=load_weights_path, train_path=train_path, validation_size=validation_size,
                batch_size=bs, epochs=ep, early_stopping=early_stopping, patience=pat,
                min_delta=md)
    loss, accuracy = siamese.evaluate(test_file=test_path, batch_size=bs, analyze=True)
    print(f'Loss on Testing set: {loss}')
    print(f'Accuracy on Testing set: {accuracy}')
    # predict_pairs(model)
    return loss, accuracy


def run():
    """
    The main function that runs the training and experiments. Uses the global variables above.
    """
    # file types
    data_set_save_type = 'pickle'
    train_path = os.path.join(data_path, f'{train_name}.{data_set_save_type}')  # A path for the train file
    test_path = os.path.join(data_path, f'{test_name}.{data_set_save_type}')  # A path for the test file
    if LOAD_DATA:  # If the training data already exists
        loader = DataLoader(width=WIDTH, height=HEIGHT, cells=CEELS, data_path=data_path, output_path=train_path)
        loader.load(set_name=train_name)
        loader = DataLoader(width=WIDTH, height=HEIGHT, cells=CEELS, data_path=data_path, output_path=test_path)
        loader.load(set_name=test_name)

    result_path = os.path.join(data_path, f'results.csv')  # A path for the train file
    results = {'lr': [], 'batch_size': [], 'epochs': [], 'patience': [], 'min_delta': [], 'seed': [], 'loss': [],
               'accuracy': []}
    for l in lr:
        for bs in batch_size:
            for ep in epochs:
                for pat in patience:
                    for md in min_delta:
                        for seed in seeds:
                            loss, accuracy = run_combination(l=l, bs=bs, ep=ep, pat=pat, md=md, seed=seed,
                                                             train_path=train_path, test_path=test_path)
                            results['lr'].append(l)
                            results['batch_size'].append(bs)
                            results['epochs'].append(ep)
                            results['patience'].append(pat)
                            results['min_delta'].append(md)
                            results['seed'].append(seed)
                            results['loss'].append(loss)
                            results['accuracy'].append(accuracy)
    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(result_path)


def initialize_seed(seed):
    """
    Initialize all relevant environments with the seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == '__main__':
    if IS_EXPERIMENT:
        # Experiments settings
        seeds = [0]
        lr = [0.00005]
        batch_size = [32]
        epochs = [10]
        patience = [5]
        min_delta = [0.1]
    else:
        # Final settings
        seeds = [0]
        lr = [0.00005]
        batch_size = [32]
        epochs = [10]
        patience = [5]
        min_delta = [0.1]

    print(os.name)
    start_time = time.time()
    print('Starting the experiments')
    run()
    print(f'Total Running Time: {time.time() - start_time}')
