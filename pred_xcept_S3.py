"""
pred_xcept.py

@author: developmentseed

Script to load a trained model, predict at scale, and save output
"""

from os import path as op
from datetime import datetime as dt

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


def download_batch(page_list, bucket, filt_ext='jpg'):
    """Get all images keys from a paginator list"""
    images = []
    #s3 = boto3.resource('s3')
    #bkt = s3.Bucket(bucket)
    s3 = boto3.client('s3')

    # Filter by extension
    key_list = [page_list[ii]['Key'] for ii in range(len(page_list))
                if filt_ext in page_list[ii]['Key']]
    fname_list = [op.splitext(key.split('/')[-1])[0] for key in key_list]

    for key in key_list:
        #st = dt.now()

        #obj = bkt.Object(key).get()
        #img_bytes = obj.get('Body')
        img_bytes = s3.get_object(Bucket=bucket, Key=key)['Body']

        #byte_time = dt.now()

        images.append(np.array(Image.open(img_bytes)))

        #append_time = dt.now()

    return fname_list, np.array(images, dtype=np.float)


def save_preds(pred_dict, bucket, s3_pred_dir, s3_pred_fname):
    """Update/save a set of predictions to S3 depending on if they exist"""

    s3 = boto3.resource('s3')
    local_fpath = op.join(preds_dir, pred_p['pred_fname'])

    existing_preds = dict()

    # Try to download predictions json file
    key_fpath = op.join(s3_pred_dir, s3_pred_fname)
    try:
        s3.Object(bucket, key_fpath).download_file(local_fpath)

        with open(local_fpath, 'r') as existing_pred_f:
            existing_preds = json.load(existing_pred_f)
        print('Found existing predictions on S3')

    # If error, make sure the error was that the file doesn't exist
    except ClientError as e:
        if int(e.response['Error']['Code']) != 404:  # Key exist, some other error
            raise e

    # Update prediction dict and upload to S3
    existing_preds.update(pred_dict)
    with open(local_fpath, 'w') as new_pred_f:
        json.dump(existing_preds, new_pred_f)

    s3.Bucket(bucket).upload_file(local_fpath, key_fpath)
    print('Uploaded new predictions to S3')


####################################
# Load model and params
####################################
K.set_learning_phase = 0  # Make sure learning is turned off for optimization
# Load weights on CPU to avoid taking up GPU space
#with tf.device('/cpu:0'):
template_model = load_model(op.join(ckpt_dir, pred_p['model_arch_fname']),
                            op.join(ckpt_dir, pred_p['model_weights_fname']))
if pred_p['n_gpus'] > 1:
    parallel_model = multi_gpu_model(template_model, gpus=pred_p['n_gpus'])
else:
    parallel_model = template_model
parallel_batch_size = pred_p['single_batch_size'] * pred_p['n_gpus']

# Load model parameters for printing
with open(op.join(ckpt_dir, pred_p['model_params_fname']), 'r') as f_model_params:
    params_yaml = f_model_params.read()
    model_params = yaml.load(params_yaml)
print('Loaded model: {}\n\twith params: {}'.format(
    pred_p['model_arch_fname'], pred_p['model_weights_fname']))

#pred_fpath = op.join(preds_dir, 'pred_list_{}.txt'.format(pred_p['model_time']))

############################################
# Create paginator for iterating over images
############################################
# Set up interface for downloading tiles
client = boto3.client('s3', region_name='us-east-1')
paginator = client.get_paginator('list_objects')

#TODO: Could extend this with starting page index
page_size = pred_p['single_batch_size'] * pred_p['n_gpus']
op_params = dict(Bucket=pred_p['aws_bucket_name'],
                 Prefix=pred_p['aws_country_dir'],
                 PaginationConfig=dict(PageSize=page_size))

s3_pg_iter = paginator.paginate(**op_params)
print('Paginator created. Will run on {} GPUs w/ {} samples per GPU'.format(
    pred_p['n_gpus'], pred_p['single_batch_size']))

###############################
# Predict on batches of images
###############################
# TODO: get num batches
n_batches = -1
n_images_processed = 0
y_preds = []
pred_dict = dict()

st_dt = dt.now()
start_time = st_dt.strftime("%m%d_%H%M%S")
print('Start time: {}'.format(start_time))

for pi, page in enumerate(s3_pg_iter):
    batch_st = dt.now()
    #TODO: skip images that are already predicted on

    # Get/preprocess desired inds
    temp_fnames, img_batch = download_batch(page['Contents'],
                                            pred_p['aws_bucket_name'])

    down_delta = dt.now() - batch_st
    print('%s images downloaded in %s' % (img_batch.shape[0], down_delta))

    # Preproc and predict on batch of images
    image_tensor = xcept_preproc(img_batch)
    y_p = parallel_model.predict(image_tensor, batch_size=parallel_batch_size,
                                 verbose=2)

    # Get only prob of positive example and round
    temp_preds = np.around(y_p[:, 1], decimals=pred_p['deci_prec'])

    y_preds.extend(temp_preds)
    for d_key, d_pred in zip(temp_fnames, temp_preds):
        pred_dict[d_key] = str(d_pred)

    '''
    # Append results to file
    with open(pred_fpath, 'a') as pred_list_f:
        for tile_fname, y_p in zip(temp_fnames, temp_preds):
            pred_list_f.write('{}\t{}\n'.format(tile_fname, str(y_p)))
    '''
    n_images_processed += len(y_p)

    batch_delta = dt.now() - batch_st
    print('Batch %s; elapsed time: %s, %s per image' % (pi, batch_delta,
                                                        batch_delta / len(y_p)))
    # Every once in a while, save predictions
    if pi % 10 == 0:
        pred_dict['page_ind'] = pi
        save_preds(pred_dict, pred_p['aws_bucket_name'],
                   pred_p['aws_pred_dir'], pred_p['pred_fname'])
        pred_dict = dict()  # Erase local copy now that it's saved


delta = dt.now() - st_dt
print('Elapsed time: %s, %s per image' % (delta, delta / n_images_processed))
