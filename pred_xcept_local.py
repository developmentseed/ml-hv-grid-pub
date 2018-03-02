"""
pred_xcept_local.py

@author: developmentseed

Script to load a trained model, predict at scale on local images, and save output
"""
import os
from os import path as op
from datetime import datetime as dt

import numpy as np
import json
import tensorflow as tf
import boto3
from botocore.exceptions import ClientError
import numpy as np
import keras.backend as K
from keras.utils import multi_gpu_model
from keras.applications.xception import preprocess_input as xcept_preproc
from PIL import Image
import yaml

from utils import load_model
from config import ckpt_dir, preds_dir, pred_params as pred_p


def load_preds(loc_pred_fpath, bucket, s3_pred_fpath):
    s3 = boto3.resource('s3')

    loaded_preds = dict()
    # Try to download predictions json file and load into memory
    try:
        s3.Object(bucket, s3_pred_fpath).download_file(loc_pred_fpath)

        with open(loc_pred_fpath, 'r') as existing_pred_f:
            loaded_preds = json.load(existing_pred_f)
        print('Found existing predictions on S3')

    # If error, make sure the error was that the file doesn't exist
    except ClientError as e:
        if int(e.response['Error']['Code']) != 404:  # Key exist, some other error
            raise e
        print('No predictions found on S3')

    return loaded_preds

def save_preds(existing_preds, new_pred_dict, loc_pred_fpath, bucket, s3_pred_fpath):
    """Update/save a set of predictions to S3 depending on if it exists (and locally)"""
    s3 = boto3.resource('s3')

    # Update prediction dict and upload to S3
    existing_preds.update(new_pred_dict)
    with open(loc_pred_fpath, 'w') as new_pred_f:
        json.dump(existing_preds, new_pred_f)

    s3.Object(bucket, s3_pred_fpath).upload_file(loc_pred_fpath)
    print('Uploaded new predictions to S3')


####################################
# Load model and params
####################################
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

parallel_batch_size = pred_p['single_batch_size'] * pred_p['n_gpus']

# Load model parameters for printing
with open(op.join(ckpt_dir, pred_p['model_params_fname']), 'r') as f_model_params:
    params_yaml = f_model_params.read()
    model_params = yaml.load(params_yaml)
print('\n' + "=" * 40)
print('Loaded model: {}\n\twith params: {}, gpus: {}'.format(
    pred_p['model_arch_fname'], pred_p['model_weights_fname'], pred_p['n_gpus']))

pred_json_fpath_loc = op.join(preds_dir, pred_p['pred_fname'])
pred_json_fpath_S3 = op.join(pred_p['aws_pred_dir'], pred_p['pred_fname'])

####################################################
# Create filenames to process, exclude existing ones
####################################################

all_tile_fnames = os.listdir(pred_p['local_img_dir'])
all_tile_inds = [op.splitext(tile)[0] for tile in all_tile_fnames]
existing_preds = load_preds(pred_json_fpath_loc, pred_p['aws_bucket_name'],
                            pred_json_fpath_S3)

print('\n{} tiles to predict on'.format(len(all_tile_inds)))
all_tile_inds = list(set(all_tile_inds) - set(existing_preds.keys()))
all_tile_inds = [ind for ind in all_tile_inds if
                 op.isfile(op.join(pred_p['local_img_dir'], ind + '.jpg'))]
print('{} tiles to predict on after set exclusion and file check'.format(len(all_tile_inds)))

###############################
# Predict on batches of images
###############################
n_batches = -1
n_images_processed = 0
y_preds = []
pred_dict = dict()

st_dt = dt.now()
start_time = st_dt.strftime("%m%d_%H%M%S")
print('Start time: {}'.format(start_time))

batch_size = pred_p['single_batch_size'] * pred_p['n_gpus']
n_batches = len(all_tile_inds) // batch_size

for bi in range(n_batches - 1):
    batch_st = dt.now()

    ########################################
    # Load batch into memory
    ########################################
    batch_inds = all_tile_inds[bi * batch_size:(bi + 1) * batch_size]
    batch_fpaths = [op.join(pred_p['local_img_dir'], ind + '.jpg') for ind in batch_inds]
    img_batch = np.empty((len(batch_inds), 256, 256, 3))
    for ii, image_fpath in enumerate(batch_fpaths):
        try:
            img_batch[ii] = np.array(Image.open(image_fpath))
        except OSError as os_err:
            print("Error loading {}, subbing zeros for image".format(image_fpath))
            img_batch[ii] = np.zeros((256, 256, 3))

    '''
    load_st = dt.now()
    load_delta = load_st - batch_st
    print('{} images loaded in {}; {} per image'.format(
        img_batch.shape[0], load_delta, load_delta / img_batch.shape[0]))
    '''

    ########################################
    # Preproc and predict on batch of images
    ########################################
    image_tensor = xcept_preproc(img_batch)
    '''
    preproc_st = dt.now()
    preproc_delta = preproc_st - load_st
    print('{} images preproced in {}; {} per image'.format(
        img_batch.shape[0], preproc_delta, preproc_delta / img_batch.shape[0]))
    '''

    y_p = parallel_model.predict(image_tensor, batch_size=parallel_batch_size,
                                 verbose=0)
    '''
    pred_st = dt.now()
    pred_delta = pred_st - preproc_st
    print('{} images predicted in {}; {} per image'.format(
        img_batch.shape[0], pred_delta, pred_delta / img_batch.shape[0]))
     '''

    ########################################
    # Save batch into json dict
    ########################################
    # Get only prob of positive example and round
    temp_preds = np.around(y_p[:, 1], decimals=pred_p['deci_prec'])

    y_preds.extend(temp_preds)
    for d_key, d_pred in zip(batch_inds, temp_preds):
        pred_dict[d_key] = str(d_pred)

    n_images_processed += len(y_p)

    batch_delta = dt.now() - batch_st
    print('Batch {}/{}; elapsed time: {}, {} per image'.format(
        bi, n_batches - 2, batch_delta, batch_delta / len(y_p)))
    # Every once in a while, save predictions
    if bi % 200 == 0 or bi == n_batches - 2:
        #pred_dict['page_ind'] = bi
        existing_preds = load_preds(pred_json_fpath_loc,
                                    pred_p['aws_bucket_name'],
                                    pred_json_fpath_S3)
        save_preds(existing_preds, pred_dict, pred_json_fpath_loc,
                   pred_p['aws_bucket_name'], pred_json_fpath_S3)
        pred_dict = dict()  # Erase local copy now that it's saved
        print('Last key: {}'.format(batch_inds[-1]))


delta = dt.now() - st_dt
print('Elapsed time: %s, %s per image' % (delta, delta / n_images_processed))
