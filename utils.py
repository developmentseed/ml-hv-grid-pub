"""
utils.py

@author: Development Seed

Utility functions for printing training details
"""
import shutil
import pprint

import numpy as np
import matplotlib as mpl
from keras.models import model_from_yaml
from pygeotile.tile import Tile

import config as cf


def print_start_details(start_time):
    """Print config at the start of a run."""
    pp = pprint.PrettyPrinter(indent=4)

    print('Start time: ' + start_time.strftime('%d/%m %H:%M:%S'))

    print('\nDatasets used:')
    pp.pprint(cf.dataset_fpaths)
    print('\nTraining details:')
    pp.pprint(cf.train_params)
    print('\nModel details:')
    pp.pprint(cf.model_params)
    print('\n\n' + '=' * 40)


def print_end_details(start_time, end_time):
    """Print runtime information."""
    run_time = end_time - start_time
    hours, remainder = divmod(run_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print('\n\n' + '=' * 40)
    print('End time: ' + end_time.strftime('%d/%m %H:%M:%S'))
    print('Total runtime: %i:%02i:%02i' % (hours, minutes, seconds))


def copy_filenames_to_dir(file_list, dst_dir):
    """Copy a list of filenames (like images) to new directory."""
    for file_name in file_list:
        print('Copying: %s to %s' % (file_name, dst_dir))
        shutil.copy(file_name, dst_dir)

    print('Done.')


def save_model_yaml(model, model_fpath):
    from keras.models import model_from_yaml

    """Save pre-trained Keras model."""
    with open(model_fpath, "w") as yaml_file:
        yaml_file.write(model.to_yaml())


def load_model(model_fpath, weights_fpath):
    """Load a model from yaml architecture and h5 weights."""
    assert model_fpath[-5:] == '.yaml'
    assert weights_fpath[-3:] == '.h5'

    with open(model_fpath, "r") as yaml_file:
        yaml_architecture = yaml_file.read()

    model = model_from_yaml(yaml_architecture)
    model.load_weights(weights_fpath)

    return model


def make_geo_feature(tile_dict, pred, thresh_vals, thresh_cols, coord_prec=5,
                     cmap=None):
    """Create a GeoJSON feature

    Parameters
    ----------
    tile_dict: dict
        Dict with `x`, `y`, `z` defined
    pred: float
        Model's predicted probability for having a feature of interest.
    thresh_vals: list
        Thresholds for indicated that tile should be mapped
    thresh_cols: list
        Hex color codes for each threshold
    coord_prec: int
        Number of decimals to keep for prediction score
    cmap: function
        Maps value to color hex key
    """

    # Convert tile to lat/lon bounds
    tile = Tile.from_google(google_x=tile_dict['x'], google_y=tile_dict['y'], zoom=tile_dict['z'])

    # Get lower-left (most SW) point and upper-right (most NE) point
    ll = list(tile.bounds[0])
    ur = list(tile.bounds[1])

    # Round pred score to save bytes
    for di in range(len(ll)):
        ll[di] = np.around(ll[di], decimals=coord_prec)
        ur[di] = np.around(ur[di], decimals=coord_prec)

    # GeoJSON uses long/lat
    bbox = [[[ll[1], ll[0]],
             [ur[1], ll[0]],
             [ur[1], ur[0]],
             [ll[1], ur[0]],
             [ll[1], ll[0]]]]

    # From pred, get fill color
    if cmap is None:
        for ii, upper_thresh in enumerate(thresh_vals):
            if pred <= upper_thresh:
                fill = thresh_cols[ii]
                break
    else:
        fill = mpl.colors.to_hex(cmap.to_rgba(pred))

    # Create properties
    properties = {'X': tile_dict['x'],
                  'Y': tile_dict['y'],
                  'Z': tile_dict['z'],
                  'pred': pred,
                  'fill': fill}
                  #'fill-opacity': 0.5,
                  #'stroke-width': 1,
                  #'stroke-opacity': 0.5}

    # Generate feature dict
    feature = dict(geometry={'coordinates': bbox,
                             'type': 'Polygon'},
                   type='Feature',
                   properties=properties)

    return feature
