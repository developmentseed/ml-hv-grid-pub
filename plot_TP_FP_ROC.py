"""
plot_TP_FP_ROC.py

@author: developmentseed

Load a model, get predictions on test data. Then plot ROC curve and TP/TN
distributions.
"""
import os.path as op
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import euclidean
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from keras.applications.xception import preprocess_input as xcept_preproc
import yaml
from tqdm import tqdm

from utils import load_model
from utils_data import get_concatenated_data
from config import (plot_dir, ckpt_dir, dataset_fpaths)

K.clear_session()  # Remove any existing graphs

model_time = '0129_052307'
model_arch_fname = '{}_arch.yaml'.format(model_time)
model_params_fname = '{}_params.yaml'.format(model_time)
model_weights_fname = '{}_L0.18_E16_weights.h5'.format(model_time)
y_pred_save_fpath = op.join(plot_dir, 'test_preds_{}'.format(model_time))

##################################
# Load model/data, get predictions
##################################
print('Loading model.')
model = load_model(op.join(ckpt_dir, model_arch_fname),
                   op.join(ckpt_dir, model_weights_fname))

with open(op.join(ckpt_dir, model_params_fname), 'r') as f_model_params:
    params_yaml = f_model_params.read()
    params = yaml.load(params_yaml)

print('Loading data.')
data_set = get_concatenated_data(dataset_fpaths, True, seed=42)
x_test = xcept_preproc(data_set['x_test'])
y_test = data_set['y_test']

print('Model and data loaded, predicting.')
y_pred = model.predict(x_test, batch_size=params['batch_size'], verbose=1)
assert np.allclose(np.sum(y_pred, axis=1),
                   np.ones((y_pred.shape[0]))), 'Probability must sum to 1'

######################################################
# Get output of embedding layer (2nd to last FC layer)
######################################################
print('Prediction complete; embedding {} points.'.format(x_test.shape[0]))
embedding_conv_layer = model.get_layer('dense_preoutput')
_embedding_output = K.function([model.input], [embedding_conv_layer.output])
K.set_learning_phase(0)

# TODO: Could be batched, will need to be able to handle uneven batches
y_embed = np.zeros((x_test.shape[0], embedding_conv_layer.get_config()['units']))
for ii, img in tqdm(enumerate(x_test)):
    temp = _embedding_output([img[np.newaxis, ...]])
    y_embed[ii] = temp[0].reshape(-1)

np.savez(y_pred_save_fpath, y_pred=y_pred, y_embed=y_embed, allow_pickle=False)

####################
# Calculate metrics
####################
y_pred = y_pred[:, 1]  # Only want predicted probability of 'True' label
y_test = y_test[:, 1]  # Same as above, only want 'True' label

y_neg_pred = y_pred[y_test == 0]  # Predictions for negative examples
y_pos_pred = y_pred[y_test == 1]  # Predictions for positive examples

# Accuracy (should match tensorboard)
acc = np.sum(y_test == np.round(y_pred)) / y_test.shape[0]
print('Accuracy: {:0.5f}'.format(acc))

# Compute FPR, TPR for '1' label (i.e., positive examples)
fpr, tpr, thresh = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Min corner dist (*one* optimal value for threshold derived from ROC curve)
corner_dists = np.empty((fpr.shape[0]))
for di, (x_val, y_val) in enumerate(zip(fpr, tpr)):
    corner_dists[di] = euclidean([0., 1.], [x_val, y_val])
opt_cutoff_ind = np.argmin(corner_dists)
min_corner_x = fpr[opt_cutoff_ind]
min_corner_y = tpr[opt_cutoff_ind]

####################
# Plot
####################
print('Plotting.')
plt.close('all')
sns.set()
sns.set_style('darkgrid', {"axes.facecolor": ".9"})
sns.set_context('talk', font_scale=1.1)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(fpr, tpr, lw=2, label='ROC curve (area={:0.2f})'.format(roc_auc))
ax.plot([min_corner_x, min_corner_x], [0, min_corner_y],
        color='r', lw=1, label='Min-corner distance\n(FPR={:0.2f}, thresh={:0.2f})'.format(min_corner_x, thresh[opt_cutoff_ind]))
plt.plot([0, 1], [0, 1], color='black', lw=0.75, linestyle='--')
ax.set_xlim([-0.03, 1.0])
ax.set_ylim([0.0, 1.03])
ax.set_xlabel('False Positive Rate\n(1 - Specificity)')
ax.set_ylabel('True Positive Rate\n(Sensitivity)')
ax.set_aspect('equal')
ax.set_title('ROC curve for high-voltage\ntower detection')
plt.legend(loc="lower right")
fig.tight_layout()
fig.savefig(op.join(plot_dir, 'roc_{}.png'.format(model_time)),
            dpi=150)

# Plot a kernel density estimate and rug plot
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
kde_kws = dict(shade=True, clip=[0., 1.], alpha=0.3)
rug_kws = dict(alpha=0.2)
sns.distplot(y_neg_pred, hist=False, kde=True, rug=True, norm_hist=True, color="b",
             kde_kws=kde_kws, rug_kws=rug_kws, label='True negatives', ax=ax2)
sns.distplot(y_pos_pred, hist=False, kde=True, rug=True, norm_hist=True, color="r",
             kde_kws=kde_kws, rug_kws=rug_kws, label='True positives', ax=ax2)
ax2.set_title('Predicted scores for true positives and true negatives')
ax2.set_xlim([0.0, 1.0])
ax2.set_xlabel("Model's predicted score")
ax2.set_ylabel('Probability density')
plt.legend(loc="best")
fig2.savefig(op.join(plot_dir, 'dist_fpr_tpr_{}.png'.format(model_time)),
             dpi=150)
