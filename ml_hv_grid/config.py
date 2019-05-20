"""
config.py

List some configuration parameters for training model
"""

import os
from os import path as op
#from hyperopt import hp

# Set filepaths pointing to data that will be used in training
data_fnames = [op.join('data_nigeria', 'data5.npz'),
               op.join('data_pakistan', 'data5.npz'),
               op.join('data_zambia', 'data5.npz')]
dataset_fpaths = [op.join(os.environ['BUILDS_DIR'], 'ml-hv-grid', fname)
                  for fname in data_fnames]

# Set directories for saving model weights and tensorboard information
if os.environ['USER'] == 'ec2-user':
    ckpt_dir = op.join('/mnt', 'models')
    tboard_dir = op.join('/mnt', 'tensorboard')
    preds_dir = op.join('/mnt', 'preds')
    cloud_comp = True
else:
    ckpt_dir = op.join(os.environ['BUILDS_DIR'], 'ml-hv-grid', 'models')
    tboard_dir = op.join(os.environ['BUILDS_DIR'], 'ml-hv-grid', 'tensorboard')
    preds_dir = op.join(os.environ['BUILDS_DIR'], 'ml-hv-grid', 'preds')
    plot_dir = op.join(os.environ['BUILDS_DIR'], 'ml-hv-grid', 'plots')
    cloud_comp = False

if not op.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
if not op.isdir(tboard_dir):
    os.mkdir(tboard_dir)

model_params = dict(loss=['binary_crossentropy'],
                    optimizer=[dict(opt_func='adam'),
                               dict(opt_func='rmsprop')],
                               # SGD as below performed notably poorer in 1st big hyperopt run
                               #dict(opt_func='sgd', momentum=hp.uniform('momentum', 0.5, 0.9))],
                    lr_phase1=[1e-4, 1e-3],  # learning rate for phase 1 (output layer only)
                    lr_phase2=[1e-5, 5e-4],  # learning rate for phase 2 (all layers beyond freeze_cutoff)
                    weight_init=['glorot_uniform'],
                    metrics=['accuracy'],
                    # Blocks organized in 10s, 66, 76, 86, etc.
                    freeze_cutoff=[0],  # Layer below which no training/updating occurs on weights
                    dense_size=[128, 256, 512],  # Number of nodes in 2nd to final layer
                    dense_activation=['relu', 'elu'],
                    dropout_rate=[0])  # Dropout in final layer

train_params = dict(n_rand_hp_iters=3,
                    n_total_hp_iters=100,
                    n_epo_phase1=[2, 4],  # number of epochs training only top layer
                    n_epo_phase2=18,  # number of epochs fine tuning whole model
                    batch_size=32,  # Want as large as GPU can handle, using batch-norm layers
                    prop_total_img_set=0.5,  # Proportion of total images per train epoch
                    img_size=(256, 256, 3),
                    early_stopping_patience=5,  # Number of iters w/out val_acc increase
                    early_stopping_min_delta=0.01,
                    reduce_lr_patience=2,  # Number of iters w/out val_acc increase
                    reduce_lr_epsilon=0.01,
                    class_weight={0: 1., 1: 5.},
                    shuffle_seed=42)  # Seed for random number generator


download_params = dict(aws_bucket_name='ds-ml-labs',
                       aws_dir='datasets/hv_pred_set/pakistan',
                       aws_region='us-east-1',
                       tile_ind_list=op.join(preds_dir, 'tile_inds_Pakistan_18.txt'),
                       tile_ind_list_format=['cover', 'tabbed', 'spaced'][2],  # cover.js or tab/space separated
                       n_green_threads=500,
                       download_prob=0.05,  # 0.1 downloads 10%, 0.9 downloads 90%
                       url_template='https://api.mapbox.com/v4/digitalglobe.2lnpeioh/{z}/{x}/{y}.png?access_token={token}'.format(
                           x='{x}', y='{y}', z='{z}', token=os.environ['VIVID_ACCESS_TOKEN']))

gen_tile_inds_params = dict(geojson_bounds=op.join(preds_dir, 'all_country_bounds.geojson'),
                            geojson_pakistan_bounds=op.join(preds_dir, 'pakistan_WB.geojson'),  # Seperate bounding box for Pakistan
                            country='Pakistan',
                            max_zoom=18)

pred_params = dict(aws_bucket_name='ds-ml-labs',
                   #aws_country_dir='datasets/hv_pred_set/Zambia',  # File dir for images
                   pred_fname='preds_zambia_147.json',  # File name for predictions
                   aws_pred_dir='datasets/hv_pred_set/',  # File dir for prediction values
                   local_img_dir=op.join(preds_dir, 'zambia_147'),
                   model_time='0129_052307',
                   single_batch_size=16,  # Number of images seen by a single GPU
                   n_gpus=1,
                   deci_prec=4)  # Number of decimal places in prediction precision
pred_params.update(dict(model_arch_fname='{}_arch.yaml'.format(pred_params['model_time']),
                        model_params_fname='{}_params.yaml'.format(pred_params['model_time']),
                        model_weights_fname='{}_L0.18_E16_weights.h5'.format(pred_params['model_time'])))

#pred_fnames = ['preds_nigeria_{}.json'.format(num) for num in range(133, 142)]
#geojson_out_fnames = ['maps_nigeria_{}.geojson'.format(num) for num in range(133, 142)]
pred_fnames = ['preds_nigeria_133.json',
               'preds_nigeria_134.json',
               'preds_nigeria_135.json',
               'preds_nigeria_136.json',
               'preds_nigeria_137.json',
               'preds_nigeria_138.json',
               'preds_nigeria_139.json',
               'preds_nigeria_140.json',
               'preds_nigeria_141.json']
geojson_out_fnames = ['maps_nigeria_133_92.geojson',
                      'maps_nigeria_134_92.geojson',
                      'maps_nigeria_135_92.geojson',
                      'maps_nigeria_136_92.geojson',
                      'maps_nigeria_137_92.geojson',
                      'maps_nigeria_138_92.geojson',
                      'maps_nigeria_139_92.geojson',
                      'maps_nigeria_140_92.geojson',
                      'maps_nigeria_141_92.geojson']
gen_geojson_params = dict(upper_thresh_lims=[0.92, 1.],
                          #upper_thresh_lims=[0.95, 0.98, 1.],
                          thresh_labels=['Maybe', 'Yes'],
                          #thresh_labels=['No', 'Maybe', 'Yes'],
                          thresh_cols=['#888888', '#ffff00'],
                          #thresh_cols=['#888888', ' #ff8000', '#ffff00'],
                          exclude_subthresh=True,
                          pred_fnames=pred_fnames,
                          geojson_out_fnames=geojson_out_fnames,
                          deci_prec=4)

######################
# Params for hyperopt
######################
def get_params(MP, TP):
    """Return hyperopt parameters"""
    return dict(
        optimizer=hp.choice('optimizer', MP['optimizer']),
        lr_phase1=hp.uniform('lr_phase1', MP['lr_phase1'][0], MP['lr_phase1'][1]),
        lr_phase2=hp.uniform('lr_phase2', MP['lr_phase2'][0], MP['lr_phase2'][1]),
        weight_init=hp.choice('weight_init', MP['weight_init']),
        freeze_cutoff=hp.choice('freeze_cutoff', MP['freeze_cutoff']),
        dropout_rate=hp.choice('dropout_rate', MP['dropout_rate']),
        dense_size=hp.choice('dense_size', MP['dense_size']),
        dense_activation=hp.choice('dense_activation', MP['dense_activation']),
        n_epo_phase1=hp.quniform('n_epo_phase1', TP['n_epo_phase1'][0], TP['n_epo_phase1'][1], 1),
        #n_epo_phase2=hp.quniform('n_epo_phase2', TP['n_epo_phase2'][0], TP['n_epo_phase2'][1], 1),
        n_epo_phase2=train_params['n_epo_phase2'],
        loss=hp.choice('loss', MP['loss']))
