"""
test_xcept.py

Run testing on a trained Xception network that can classify HV towers and
substations
"""

import os
from os import path as op
from datetime import datetime as dt

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.xception import preprocess_input as xcept_preproc
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from sklearn.metrics import classification_report

import yaml

from utils import print_start_details, print_end_details, load_model
from config import (ckpt_dir, data_dir, preds_dir, pred_params as pred_p,
                    data_flow as DF)


########################################
# Calculate number of test images
########################################
test_data_dir = op.join(data_dir, 'test')
print('Using test images in {}\n'.format(test_data_dir))

total_test_images = 0
for sub_fold in ['negatives', 'towers', 'substations']:
    temp_img_dir = op.join(test_data_dir, sub_fold)
    n_fnames = len([fname for fname in os.listdir(temp_img_dir)
                    if op.splitext(fname)[1] in ['.png', 'jpg']])
    print('For testing, found {} {} images'.format(n_fnames, sub_fold))

    total_test_images += n_fnames

steps_per_test_epo = int(np.ceil(total_test_images /
                                 DF['flow_from_dir']['batch_size']) + 1)

# Set up generator
test_gen = ImageDataGenerator(preprocessing_function=xcept_preproc)

print('\nCreating test generator.')
test_iter = test_gen.flow_from_directory(directory=test_data_dir,
                                         shuffle=False,
                                         **DF['flow_from_dir'])
test_iter.batch_size = 1
test_iter.reset()  # Reset for each model to ensure consistency

####################################
# Load model and params
####################################
print('Loading model.')
if pred_p['n_gpus'] > 1:
    # Load weights on CPU to avoid taking up GPU space
    with tf.device('/cpu:0'):
        template_model = load_model(op.join(ckpt_dir, pred_p['model_arch_fname']),
                                    op.join(ckpt_dir, pred_p['model_weights_fname']))
    parallel_model = multi_gpu_model(template_model, gpus=pred_p['n_gpus'])
else:
    template_model = load_model(op.join(ckpt_dir, pred_p['model_arch_fname']),
                                op.join(ckpt_dir, pred_p['model_weights_fname']))
    parallel_model = template_model

# Turn of training. This is supposed to be faster (haven't seen this empirically though)
K.set_learning_phase = 0
for layer in template_model.layers:
    layer.trainable = False

# Load model parameters for printing
with open(op.join(ckpt_dir, pred_p['model_params_fname']), 'r') as f_model_params:
    params_yaml = f_model_params.read()
    model_params = yaml.load(params_yaml)
print('Loaded model: {}\n\twith params: {}, gpus: {}'.format(
    pred_p['model_arch_fname'], pred_p['model_weights_fname'], pred_p['n_gpus']))

#######################
# Run prediction
#######################
print('\nPredicting.')
start_time = dt.now()
print('Start time: ' + start_time.strftime('%d/%m %H:%M:%S'))

y_true = test_iter.classes
class_labels = list(test_iter.class_indices.keys())

# Leave steps=None to predict entire sequence
y_pred_probs = parallel_model.predict_generator(test_iter,
                                                steps=len(test_iter),
                                                workers=16,
                                                verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_true, y_pred))

#############
# End details
#############
end_time = dt.now()
print_end_details(start_time, end_time)
