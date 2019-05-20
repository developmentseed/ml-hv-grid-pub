"""
Utility functions for the training process

@author: DevelopmentSeed

Code modified from Keras module
"""

import numpy as np

import keras
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import (precision_recall_fscore_support,
                             #f1_score, fbeta_score,
                             classification_report)


class ClasswisePerformance(keras.callbacks.Callback):
    """Callback to calculate precision, recall, F1-score after each epoch"""

    def __init__(self, test_gen, gen_steps=100):
        test_gen.shuffle = False
        self.test_gen = test_gen
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.gen_steps = gen_steps

    def on_train_begin(self, logs={}):
        self.precisions = []
        self.recalls = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):

        # TODO: Not efficient as this requires re-computing test data
        self.test_gen.reset()
        y_true = self.test_gen.classes
        class_labels = list(self.test_gen.class_indices.keys())

        # Leave steps=None to predict entire sequence
        y_pred_probs = self.model.predict_generator(self.test_gen,
                                                    steps=self.gen_steps)
        y_pred = np.argmax(y_pred_probs, axis=1)

        prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                              labels=class_labels)
        self.precisions.append(prec)
        self.recalls.append(recall)
        self.f1s.append(f1)

        self.test_gen.reset()
        print(classification_report(y_true, y_pred))


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

    return fbeta_score


def f1_score(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1.)


def f2_score(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=2.)


'''
def avg_f1_score(y_true, y_pred):
    """Wrapper for sklearn's f1-score. Returns unweighted F1 across all classes"""
    return f1_score(y_true, y_pred, average='macro')

def f2_score_towers(y_true, y_pred):
    """Wrapper for F_beta. Returns F2 for towers (weighted to reward recall)."""
    #return fbeta_score(y_true, y_pred, beta=2., labels=[1], average='macro')


def f2_score_substations(y_true, y_pred):
    """Wrapper for F_beta. Returns F2 for substations (weighted to reward recall)."""
    return fbeta_score(y_true, y_pred, beta=2., labels=[2], average='macro')

def recall_towers(y_true, y_pred):
    return tf.metrics.recall(y_true, y_pred, weights=[0, 1, 0])[0]
'''
