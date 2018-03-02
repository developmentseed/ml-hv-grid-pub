"""
gen_geojson.py

@author: developmentseed

Script to load predictions for a list of tiles and produce a geojson map
"""

from os import path as op

import json
from utils import make_geo_feature
from config import ckpt_dir, preds_dir, gen_geojson_params as geo_p

import matplotlib
import matplotlib.cm as cm

###########################
# Load predictions
###########################
for pred_fname, out_geoj_fname in zip(geo_p['pred_fnames'], geo_p['geojson_out_fnames']):
    pred_fpath = op.join(preds_dir, pred_fname)
    print('\nGenerating geojson prediction map for: {}'.format(pred_fname))

    with open(pred_fpath, 'r') as pred_f:
        pred_dict = json.load(pred_f)

    ###########################
    # Create geojson file
    ###########################
    feature_list = []

    norm = matplotlib.colors.Normalize(vmin=geo_p['upper_thresh_lims'][0], vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)

    for tile_ind, tile_pred in pred_dict.items():
        # Get tile index
        if tile_ind == 'page_ind':
            continue
        x, y, z = tile_ind.split('-')
        tile_dict = dict(x=int(x), y=int(y), z=int(z))

        feature = make_geo_feature(tile_dict, float(tile_pred),
                                   geo_p['upper_thresh_lims'],
                                   geo_p['thresh_cols'],
                                   geo_p['deci_prec'])
                                   #mapper)
        if not geo_p['exclude_subthresh']:
            feature_list.append(feature)
        elif (geo_p['exclude_subthresh'] and
              feature['properties']['pred'] > geo_p['upper_thresh_lims'][0]):
            feature_list.append(feature)


    #############################################
    # Assemble features into one geojson and save
    #############################################
    json_dict = {'type': 'FeatureCollection',
                 'features': feature_list}

    geojson_fpath = op.join(preds_dir, out_geoj_fname)
    print('Saved geojson map as {}'.format(geojson_fpath))
    with open(geojson_fpath, 'w') as geojson_file:
        json.dump(json_dict, geojson_file)
