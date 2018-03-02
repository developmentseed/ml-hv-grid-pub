"""
gen_tile_inds.py

@author: developmentseed

Generate a list of tiles via recursion
"""
import os.path as op
from queue import LifoQueue
from datetime import datetime as dt
from decimal import Decimal

import json
from shapely.geometry import shape, Polygon
from pygeotile.tile import Tile

from config import preds_dir, gen_tile_inds_params as tile_p


def get_quadrant_tiles(tile):
    """Return indicies of tiles at one higher zoom (in google tiling scheme)"""
    ul = (tile.google[0] * 2, tile.google[1] * 2)

    return [Tile.from_google(ul[0], ul[1], tile.zoom + 1),           # UL
            Tile.from_google(ul[0], ul[1] + 1, tile.zoom + 1),       # LL
            Tile.from_google(ul[0] + 1, ul[1], tile.zoom + 1),       # UR
            Tile.from_google(ul[0] + 1, ul[1] + 1, tile.zoom + 1)]   # LR


def calc_overlap(geom1, geom2):
    """Return area overlap"""

    return geom1.intersection(geom2).area


def check_tile(tile, roi_geom, completely_contained=False):
    """Return overlapping tiles recursively down to `max_zoom`
    Parameters
    ----------
    tile: pygeotile.tile.Tile
        Tile that is checked for overlap with `roi_geom`.
    roi_geom: shapely.geometry.shape
        Boundary of region-of-interest.
    completely_contained: bool
        Whether or not a tile is completely contained in the boundary.
        If a tile is found to have 100% overlap with boundary, set to `True`
        and algorithm can avoid calculating overlap for all future child tiles.
        Default False."""

    return_tiles = []
    quad_tiles = get_quadrant_tiles(tile)  # Compute four contained tiles

    # If sub-tiles are completely contained within boundary, no need to compute overlap
    if completely_contained:
        return [[qt, True] for qt in quad_tiles]

    # For each tile, compute overlap with ROI boundary
    for qt in quad_tiles:
        ll, ur = qt.bounds  # Get lower-left and upper-right points
        tile_pts = ((ll[1], ll[0]), (ur[1], ll[0]),
                    (ur[1], ur[0]), (ll[1], ur[0]))
        tile_polygon = Polygon(tile_pts)

        # Calculate overlap of tile with ROI
        overlap_area = calc_overlap(roi_geom, tile_polygon)

        # If 100% overlap, indicate this to avoid future area overlap checks
        if overlap_area == tile_polygon.area:
            return_tiles.append([qt, True])
        elif overlap_area > 0:
            return_tiles.append([qt, False])

    return return_tiles


if __name__ == "__main__":
    # TODO: Convert main to more independent API (for distribution later)
    # output text fname
    # zoom
    # bounds loaded from geojson
    # tile ind format

    #########################################
    # Load boundary, get file paths
    #########################################
    tile_output_fpath = op.join(preds_dir, 'tile_inds_{}_{}.txt'.format(
        tile_p['country'], tile_p['max_zoom']))
    if op.exists(tile_output_fpath):
        raise ValueError('Output file ({}) already exists'.format(tile_output_fpath))

    # Handle Pakistan seperately because of their border dispute with India
    if tile_p['country'] == 'Pakistan':
        with open(tile_p['geojson_pakistan_bounds'], 'r') as geojson_f:
            geojson = json.loads(geojson_f.read())
            print('Country bounds loaded from: {}'.format(tile_p['geojson_pakistan_bounds']))
            bound = geojson['features'][0]['geometry']
    else:
        with open(tile_p['geojson_bounds'], 'r') as geojson_f:
            # Get desired boundary
            geojson = json.loads(geojson_f.read())
            print('Country bounds loaded from: {}'.format(tile_p['geojson_bounds']))
            for fi, feature in enumerate(geojson['features']):
                if feature['properties']['SOVEREIGNT'] == tile_p['country']:
                    bound = feature['geometry']
                    break
    print('Computing tiles at zoom {} for {}'.format(
        tile_p['max_zoom'], tile_p['country']))

    #########################################
    # Setup boundary, initialize stack
    #########################################
    st_dt = dt.now()  # Start time

    boundary_shape = shape(bound)
    stack = LifoQueue()

    start_tile = Tile.from_google(0, 0, 0)
    max_stack_size = 0
    total_tiles = 0
    stack.put([start_tile, False])  # Add biggest tile to stack

    #########################################
    # Depth-first search on tile indices
    #########################################
    with open(tile_output_fpath, 'w') as tile_list_f:
        while not stack.empty():
            # Track maximum stack size
            if stack.qsize() > max_stack_size:
                max_stack_size = stack.qsize()

            # Pop the top tile in the stack
            top_tile, comp_contained = stack.get()

            # Check if desired zoom has been reached
            if top_tile.zoom >= tile_p['max_zoom']:
                print('{} {} {}'.format(top_tile.google[0], top_tile.google[1],
                                        top_tile.zoom), file=tile_list_f)
                total_tiles += 1

                if total_tiles % 1e5 == 0:
                    print('Tiles saved: {:.3E}'.format(Decimal(total_tiles)))

            # Otherwise, zoom in one increment and add tiles to the stack
            else:
                ret_tiles = check_tile(top_tile, boundary_shape,
                                       comp_contained)
                for rt in ret_tiles:
                    stack.put(rt)

    #########################################
    # Print some final details
    #########################################
    delta = dt.now() - st_dt
    print('Complete; Max stack size: {}'.format(max_stack_size))
    print('Elapsed time: {}'.format(delta))
    print('Tiles saved to: {}'.format(tile_output_fpath))
