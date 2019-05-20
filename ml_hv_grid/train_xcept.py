"""
train_xcept.py

Train the Xception network to classify HV towers and substations
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
from keras.optimizers import Adam, rmsprop, SGD, Nadam
from tensorflow.python.keras.optimizers import TFOptimizer
from keras.applications.xception import Xception, preprocess_input as xcept_preproc
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (ModelCheckpoint, EarlyStopping, TensorBoard,
                             ReduceLROnPlateau)

from hyperopt import fmin, Trials, STATUS_OK, tpe
import yaml

from utils import print_start_details, print_end_details
from utils_training import ClasswisePerformance
from config import (get_params, tboard_dir, ckpt_dir, data_dir,
                    model_params as MP, train_params as TP, data_flow as DF)


def get_optimizer(opt_params, lr):
    """Helper to get optimizer from text params

    Parameters
    ----------
    opt_params: dict
        Dictionary containing optimization function name and learning rate decay
    lr:  float
        Initial learning rate

    Return
    ------
    opt_function: Keras optimizer
    """

    if opt_params['opt_func'] == 'sgd':
        return SGD(lr=lr, momentum=opt_params['momentum'])
    elif opt_params['opt_func'] == 'adam':
        return Adam(lr=lr)
    elif opt_params['opt_func'] == 'rmsprop':
        return rmsprop(lr=lr)
    elif opt_params['opt_func'] == 'nadam':
        return Nadam(lr=lr)
    elif opt_params['opt_func'] == 'powersign':
        from tensorflow.contrib.opt.python.training import sign_decay as sd
        d_steps = opt_params['pwr_sign_decay_steps']
        # Define the decay function (if specified)
        if opt_params['pwr_sign_decay_func'] == 'lin':
            decay_func = sd.get_linear_decay_fn(d_steps)
        elif opt_params['pwr_sign_decay_func'] == 'cos':
            decay_func = sd.get_consine_decay_fn(d_steps)
        elif opt_params['pwr_sign_decay_func'] == 'res':
            decay_func = sd.get_restart_decay_fn(d_steps,
                                                 num_periods=opt_params['pwr_sign_decay_periods'])
        elif opt_params['decay_func'] is None:
            decay_func = None
        else:
            raise ValueError('decay function not specified correctly')

        # Use decay function in TF optimizer
        return TFOptimizer(PowerSignOptimizer(learning_rate=lr,
                                              sign_decay_fn=decay_func))
    else:
        raise ValueError


def xcept_net(params):
    """Train the Xception network

    Parmeters:
    ----------
    params: dict
        Parameters returned from config.get_params() for hyperopt

    Returns:
    --------
    result_dict: dict
        Results of model training for hyperopt.
    """
    
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

    print('Creating test generator.')
    test_iter = test_gen.flow_from_directory(
        directory=op.join(data_dir, 'test'), shuffle=False,  # Helps maintain consistency in testing phase
        **DF['flow_from_dir'])
    test_iter.reset()  # Reset for each model so it's consistent; ideally should reset every epoch

    callbacks_phase1 = [TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                                    write_grads=False, embeddings_freq=0)]
                                    #embeddings_layer_names=['dense_preoutput', 'dense_output'])]
    # Set callbacks to save performance to TB, modify learning rate, and stop poor trials early
    callbacks_phase2 = [
        TensorBoard(log_dir=tboard_model_dir, histogram_freq=0,
                    write_grads=False, embeddings_freq=0),
                    #embeddings_layer_names=['dense_preoutput', 'dense_output']),
        #ClasswisePerformance(test_iter, gen_steps=params['steps_per_test_epo']),
        ModelCheckpoint(ckpt_fpath, monitor='val_loss', mode='min',
                        save_weights_only=True, save_best_only=False),
        EarlyStopping(min_delta=TP['early_stopping_min_delta'],
                      patience=TP['early_stopping_patience'], verbose=1),
        ReduceLROnPlateau(min_delta=TP['reduce_lr_min_delta'],
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

    # Finally, add output layer
    pred = Dense(params['n_classes'],
                 activation=params['output_activation'],
                 name='dense_output')(x)

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
            yaml_file.write(yaml.dump(MP))
            yaml_file.write(yaml.dump(DF))

    ##########################
    # Train the new top layers
    ##########################
    # Train the top layers which we just added by setting all orig layers untrainable
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model (after setting non-trainable layers)
    model.compile(optimizer=get_optimizer(params['optimizer'],
                                          lr=params['lr_phase1']),
                  loss=params['loss'],
                  metrics=MP['metrics'])

    print('Phase 1, training near-output layer(s)')
    hist = model.fit_generator(
        train_gen.flow_from_directory(directory=op.join(data_dir, 'train'),
                                      **DF['flow_from_dir']),
        steps_per_epoch=params['steps_per_train_epo'],
        epochs=params['n_epo_phase1'],
        callbacks=callbacks_phase1,
        max_queue_size=params['max_queue_size'],
        workers=params['workers'],
        use_multiprocessing=params['use_multiprocessing'],
        class_weight=params['class_weight'],
        verbose=1)

    ###############################################
    # Train entire network to fine-tune performance
    ###############################################
    # Visualize layer names/indices to see how many layers to freeze:
    #print('Layer freeze cutoff = {}'.format(params['freeze_cutoff']))
    #for li, layer in enumerate(base_model.layers):
    #    print(li, layer.name)

    # Set all layers trainable
    for layer in model.layers:
        layer.trainable = True

    # Recompile model for second round of training
    model.compile(optimizer=get_optimizer(params['optimizer'],
                                          params['lr_phase2']),
                  loss=params['loss'],
                  metrics=MP['metrics'])

    print('\nPhase 2, training from layer {} on.'.format(params['freeze_cutoff']))
    test_iter.reset()  # Reset for each model so it's consistent; ideally should reset every epoch

    hist = model.fit_generator(
        train_gen.flow_from_directory(directory=op.join(data_dir, 'train'),
                                      **DF['flow_from_dir']),
        steps_per_epoch=params['steps_per_train_epo'],
        epochs=params['n_epo_phase2'],
        max_queue_size=params['max_queue_size'],
        workers=params['workers'],
        use_multiprocessing=params['use_multiprocessing'],
        validation_data=test_iter,
        validation_steps=params['steps_per_test_epo'],
        callbacks=callbacks_phase2,
        class_weight=params['class_weight'],
        verbose=1)

    # Return best of last validation accuracies
    check_ind = -1 * (TP['early_stopping_patience'] + 1)
    result_dict = dict(loss=np.min(hist.history['val_loss'][check_ind:]),
                       status=STATUS_OK)

    return result_dict


if __name__ == '__main__':
    start_time = dt.now()
    print_start_details(start_time)

    ###################################
    # Calculate number of train/test images
    ###################################
    total_test_images = 0
    # Print out how many images are available for train/test
    for fold in ['train', 'test']:
        for sub_fold in ['negatives', 'towers', 'substations']:
            temp_img_dir = op.join(data_dir, fold, sub_fold)
            n_fnames = len([fname for fname in os.listdir(temp_img_dir)
                            if op.splitext(fname)[1] in ['.png', 'jpg']])
            print('For {}ing, found {} {} images'.format(fold, n_fnames, sub_fold))

            if fold == 'test':
                total_test_images += n_fnames
    if TP['steps_per_test_epo'] is None:
        TP['steps_per_test_epo'] = int(np.ceil(total_test_images /
                                               DF['flow_from_dir']['batch_size']) + 1)

    ###################################
    # Set up generators
    ###################################
    train_gen = ImageDataGenerator(preprocessing_function=xcept_preproc,
                                   **DF['image_data_generator'])
    test_gen = ImageDataGenerator(preprocessing_function=xcept_preproc)

    ############################################################
    # Run training with hyperparam optimization (using hyperopt)
    ############################################################
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
