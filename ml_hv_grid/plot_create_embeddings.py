"""
plot_create_embeddings.py

@author: developmentseed

Load a predictions, create prediction embeddings for tensorboard visualization
"""

import os.path as op
import numpy as np
from imageio import imwrite
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from utils import load_model
from utils_data import get_concatenated_data
from config import (plot_dir, ckpt_dir, dataset_fpaths,
                    model_params as MP, train_params as TP)


# Adapted from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

    #data = data.astype(np.float32)
    #min_val = np.min(data, axis=(1, 2, 3))
    #data = (data.transpose(1, 2, 3, 0) - min_val).transpose(3,0,1,2)
    #max_val = np.max(data, axis=(1, 2, 3))
    #data = (data.transpose(1, 2, 3, 0) / max_val).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
        (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)

    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
        + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #data *= 255
    data = data.astype(np.uint8)

    return data

model_time = '0124_070717'
y_pred_save_fpath = op.join(plot_dir, 'test_preds_{}.npz'.format(model_time))
# Reduce number of images; sprite must be <= 8192x8192; 1024 images for 256x256
img_lim = 1024


##################################
# Load model/data, get predictions
##################################
# Load original data and embeddings
data_set = get_concatenated_data(dataset_fpaths, True, seed=42)
model_output = np.load(y_pred_save_fpath)

# Reminder that `x_test` data is preprocessed before prediction
x_test = data_set['x_test'][:img_lim]          # Test images
y_test = data_set['y_test'][:img_lim, 1]       # Test image class
y_pred = model_output['y_pred'][:img_lim, 1]   # Model prediction (0..1)
y_embed = model_output['y_embed'][:img_lim]    # Embedding for each image
print('Loaded image data, predictions, and embedding.')

with tf.Session() as sess:
    print('Creating TF representation of embedding')
    # Store embedding as a variable
    embedding_var = tf.Variable(y_embed, name='embedding_layer')
    # XXX: Sometimes TB has problems finding ckpt file -- it adds additional
    #   characters on the end (e.g., `-1.data...`). Sometimes, you have to
    #   manually make sure that the .pbtext file refers to the right one
    ckpt_fpath = op.join(ckpt_dir, 'embedding_ckpt_file.ckpt')

    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(ckpt_dir)

    # Set up project and its config
    proj_config = projector.ProjectorConfig()
    proj_config.model_checkpoint_path = ckpt_fpath
    embedding = proj_config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    projector.visualize_embeddings(summary_writer, proj_config)

    print('Writing metadata concerning class info')
    metadata_fname = 'metadata_{}.tsv'.format(model_time)
    with open(op.join(ckpt_dir, metadata_fname), 'w') as metadata_file:
        metadata_file.write('Name\tClass\tPrediction\n')
        for ii, (y_t, y_p) in enumerate(zip(y_test, y_pred)):
            metadata_file.write('{:4.0f}\t{:1.0f}\t{:0.4f}\n'.format(
                int(ii), int(y_t), y_p))
        metadata_file.close()
    embedding.metadata_path = op.join(ckpt_dir, metadata_fname)

    print('Creating sprite sheet')
    sprite_fpath = op.join(ckpt_dir, 'sprite_{}'.format(model_time))
    sprite = images_to_sprite(x_test)
    imwrite(sprite_fpath + '.png', sprite)

    embedding.sprite.image_path = sprite_fpath + '.png'
    embedding.sprite.single_image_dim.extend([x_test.shape[1], x_test.shape[2]])

    print('Saving configuration')
    projector.visualize_embeddings(summary_writer, proj_config)
    saver = tf.train.Saver([embedding_var])

    saver.save(sess, ckpt_fpath)
