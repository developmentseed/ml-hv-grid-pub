# Machine learning detection of high-voltage grid (ml-hv-grid)

## Scope & overall goal (from proposal)

Development Seed proposed to use machine assisted tracing to generate a high-voltage (HV) grid map for three countries: Pakistan, Nigeria and Zambia. A full report is [available online](http://devseed.com/ml-grid-docs/). The methodology that we proposed is based on a pilot R&D effort described in the report [Machine Learning for Africa’s Grid](http://devseed.com/ml-grid-detection/).

The goal here is to develop a cost-effective HV grid mapping approach that can be replicated in countries across the globe. At the end of this project, Development Seed is to publish and deliver the following to the energy team at the World Bank:
1. a complete and accurate (within 10 meters) map of the HV transmission network in three priority countries in Africa and Asia, delivered in GeoJSON format and added directly to OSM;
1. the training data and ML models;
1. a thoroughly documented approach that would allow the Bank or other organizations to replicate this in other countries.

All data is stored within [OpenStreetMap](https://www.openstreetmap.org/).

## Machine learning approach

The machine learning (ML) goal here is to detect HV towers as accurately as possible in satellite imagery. Here, we are using zoom 18 tiles, as the towers are large enough to be clearly visible. Therefore, the specific ML task is a basic form of classification -- take any zoom 18 RGB tile and calculate the probability (on the interval [0, 1]) that it contains a HV tower. By default, we can use a threshold of 0.5 as the cutoff, but this can be modified to adjust the false-positive rate. For training, several thousand images (from the Digital Globe standard layer) spanning all 3 target countries were manually checked by the Peru Data Team.

We used a ML model called the "Xception" network that was pre-trained on ImageNet -- a large corpus of natural images used as a benchmark by the ML community. The model is retrained using satellite imagery (i.e., we use a "transfer learning" approach) to detect HV towers. We evaluate performance based on raw detection accuracy, false- and true-positive rates, and the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve.

With a trained model, the final task is to run predictions entire countries (tiled to zoom 18). With this prediction map, the data-team can prioritize it's time to focus on only areas predicted to have high value.

## Files and workflow
* **/models**			contains model architecture and weights; Can be loaded within Keras
* **/training-roi** 	initial set of ROIs used by data-team to generate training data
* **/utils_cloud** 		files and utility scripts to run ML model on cloud
* **config.py** 		all configuration parameters for training and testing with the model as well as storing results
* **download_tiles**.py Fetches and saves satellite imagery tiles from a list of tile indices.
* **gen_tile_inds.py** Creates a list of tile indices from a geojson boundary
* **plot_TP_FP_ROC.py**	runs predictions on initial test data, plots ROC curve and TP/FP distributions
* **plot_create_embeddings.py** creates files needed for TF embedding plots. Run after "plot_TP_FP_ROC.py"
* **plot_mapping_speed.py** Script to create a plot showing relative mapping speed for towers, substations, and km^2 per hour
* **pred_xcept_local.py** predicts on large batches of tiles using a pre-trained model
* **train_xcept.py**	trains xception network, see config.py for setting parameters
* **utils.py**			utilities for printing information, loading/saving models, creating geojsons from predictions
* **utils_data.py**		utilities to prepare data
