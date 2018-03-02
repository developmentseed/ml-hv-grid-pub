"""
train_xcept.py

Train the Xception network to classify HV pylons
"""
import os
from os import path as op
from functools import partial
from datetime import datetime as dt
import pickle
import pprint

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, rmsprop, SGD
from keras.applications.xception import Xception, preprocess_input as xcept_preproc
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (ModelCheckpoint, EarlyStopping, TensorBoard,
                             ReduceLROnPlateau)
from hyperopt import fmin, Trials, STATUS_OK, tpe
import yaml

from utils import (print_start_details, print_end_details)
from utils_data import get_concatenated_data
from config import (get_params, tboard_dir, ckpt_dir, dataset_fpaths,
                    model_params as MP, train_params as TP)


def get_optimizer(opt_params, lr):
    """Helper to get optimizer from text params"""
    if opt_params['opt_func'] == 'sgd':
        return SGD(lr=lr, momentum=opt_params['momentum'])
    elif opt_params['opt_func'] == 'adam':
        return Adam(lr=lr)
    elif opt_params['opt_func'] == 'rmsprop':
        return rmsprop(lr=lr)
    else:
        raise ValueError


def xcept_net(params):
    """Train the Xception network"""
    K.clear_session()  # Remove any existing graphs
    mst_str = dt.now().strftime("%m%d_%H%M%S")

    print('\n' + '=' * 40 + '\nStarting model at {}'.format(mst_str))
    print('Model # %s' % len(trials))
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    ######################
    # Paths and Callbacks
    ######################
    ckpt_fpath = op.join(ckpt_dir, mst_str + '_L{val_loss:.2f}_E{epoch:02d}_weights.h5')
    tboard_model_dir = op.join(tboard_dir, mst_str)

    callbacks_phase1 = [TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                                    write_grads=False, embeddings_freq=0,
                                    embeddings_layer_names=['dense_preoutput', 'dense_output'])]
    callbacks_phase2 = [
        TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                    write_grads=False, embeddings_freq=0,
                    embeddings_layer_names=['dense_preoutput', 'dense_output']),
        ModelCheckpoint(ckpt_fpath, monitor='val_acc', save_weights_only=True,
                        save_best_only=True),
        EarlyStopping(min_delta=TP['early_stopping_min_delta'],
                      patience=TP['early_stopping_patience'], verbose=1),
        ReduceLROnPlateau(epsilon=TP['reduce_lr_epsilon'],
                          patience=TP['reduce_lr_patience'], verbose=1)]

    #########################
    # Construct model
    #########################
    # Get the original xception model pre-initialized weights
    base_model = Xception(weights='imagenet',
                          include_top=False,  # Peel off top layer
                          input_shape=TP['img_size'],
                          pooling='avg')  # Global average pooling

    x = base_model.output  # Get final layer of base XCeption model

    # Add a fully-connected layer
    x = Dense(params['dense_size'], activation=params['dense_activation'],
              kernel_initializer=params['weight_init'],
              name='dense_preoutput')(x)
    if params['dropout_rate'] > 0:
        x = Dropout(rate=params['dropout_rate'])(x)

    # Finally, add softmax output with 2 classes (since we have binary prediction)
    pred = Dense(2, activation='softmax', name='dense_output')(x)

    model = Model(inputs=base_model.input, outputs=pred)

    #####################
    # Save model details
    #####################
    model_yaml = model.to_yaml()
    save_template = op.join(ckpt_dir, mst_str + '_{}.{}')
    arch_fpath = save_template.format('arch', 'yaml')
    if not op.exists(arch_fpath):
        with open(arch_fpath.format('arch', 'yaml'), 'w') as yaml_file:
            yaml_file.write(model_yaml)

    # Save params to yaml file
    params_fpath = save_template.format('params', 'yaml')
    if not op.exists(params_fpath):
        with open(params_fpath, 'w') as yaml_file:
            yaml_file.write(yaml.dump(params))
            yaml_file.write(yaml.dump(TP))

    ####################
    # Train top layers
    ####################
    # Train the top layers which we just added by setting all orig layers untrainable
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model (do this after setting non-trainable layers)
    model.compile(optimizer=get_optimizer(params['optimizer'],
                                          lr=params['lr_phase1']),
                  loss=params['loss'], metrics=MP['metrics'])

    # Train top layers for a few epocs
    steps_per_epo = (len(X_train) * TP['prop_total_img_set']) // TP['batch_size']
    steps_per_val = len(X_test) // TP['batch_size']

    print('Phase 1, training near-output layer(s)')
    hist = model.fit_generator(
        train_gen.flow(X_train, Y_train, batch_size=TP['batch_size']),
        steps_per_epoch=steps_per_epo,
        epochs=params['n_epo_phase1'],
        #validation_data=test_gen.flow(X_test, Y_test, batch_size=TP['batch_size']),
        #validation_steps=steps_per_val,
        callbacks=callbacks_phase1,
        class_weight=TP['class_weight'],
        verbose=1)

    ###############################################
    # Train entire network to fine-tune performance
    ###############################################
    # Visualize layer names/indices to see how many layers to freeze:
    #print('Layer freeze cutoff = {}'.format(params['freeze_cutoff']))
    #for li, layer in enumerate(base_model.layers):
    #    print(li, layer.name)

    for layer in model.layers[params['freeze_cutoff']:]:
        layer.trainable = True
    for layer in model.layers[:params['freeze_cutoff']]:
        layer.trainable = False

    # Recompile model for second round of training
    model.compile(optimizer=get_optimizer(params['optimizer'], params['lr_phase2']),
                  loss=params['loss'], metrics=MP['metrics'])

    print('/nPhase 2, training from layer {} on.'.format(params['freeze_cutoff']))
    hist = model.fit_generator(
        train_gen.flow(X_train, Y_train, batch_size=TP['batch_size']),
        steps_per_epoch=steps_per_epo,
        epochs=params['n_epo_phase2'],
        validation_data=test_gen.flow(X_test, Y_test, batch_size=TP['batch_size']),
        validation_steps=steps_per_val,
        callbacks=callbacks_phase2,
        class_weight=['class_weight'],
        verbose=1)

    # Return best of last validation accuracies
    check_ind = -1 * (TP['early_stopping_patience'] + 1)
    result_dict = dict(loss=np.min(hist.history['val_loss'][check_ind:]),
                       status=STATUS_OK)

    return result_dict


if __name__ == '__main__':
    start_time = dt.now()
    print_start_details(start_time)

    #########################
    # Load data
    #########################
    data_set = get_concatenated_data(dataset_fpaths, True, seed=TP['shuffle_seed'])
    X_train = data_set['x_train']
    X_test = data_set['x_test']
    Y_train = data_set['y_train']
    Y_test = data_set['y_test']
    total_counts = np.sum(Y_train, axis=0) + np.sum(Y_test, axis=0)

    train_gen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        zoom_range=(1, 1.2),
        preprocessing_function=xcept_preproc)
    test_gen = ImageDataGenerator(
        preprocessing_function=xcept_preproc)

    trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=TP['n_rand_hp_iters'])
    argmin = fmin(xcept_net, space=get_params(MP, TP), algo=algo,
                  max_evals=TP['n_total_hp_iters'], trials=trials)

    end_time = dt.now()
    print_end_details(start_time, end_time)
    print("Evalutation of best performing model:")
    print(trials.best_trial['result']['loss'])

    with open(op.join(ckpt_dir, 'trials_{}.pkl'.format(start_time)), "wb") as pkl_file:
        pickle.dump(trials, pkl_file)
