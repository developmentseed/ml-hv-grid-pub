"""
Script demonstrating how to export a Keras model. Some version of the saved
ML HV grid model should be on disk locally. Or, just make sure you've
loaded some Keras model.

@author: Development Seed
"""

import os
import os.path as op

import tensorflow as tf
from utils import load_model

# Load the Keras model from disk. Skip this step if model is in memory
model_start_time = '181223_023807'
export_dir = op.join(os.environ['BUILDS_DIR'], 'ml-hv-grid',
                     'models_hvgrid_export', '002')

#######################################
# Input processing functions
#######################################
HEIGHT, WIDTH, CHANNELS = 256, 256, 3


def serving_input_receiver_fn():
    """Convert string encoded images into preprocessed tensors"""

    def decode_and_resize(image_str_tensor):
        """Decodes an image string, preprocesses/resizes it, and returns a uint8 tensor."""
        image = tf.image.decode_image(image_str_tensor, channels=CHANNELS,
                                      dtype=tf.uint8)
        image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])

        return image

    # Run preprocessing for batch prediction
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images_tensor = tf.map_fn(
        decode_and_resize, input_ph, back_prop=False, dtype=tf.uint8)

    # Cast to float32 and run Xception preprocessing on images
    #    (to scale [0, 255] to [-1, 1])
    images_tensor = tf.cast(images_tensor, dtype=tf.float32)
    images_tensor = tf.subtract(tf.divide(images_tensor, 127.5), 1)

    return tf.estimator.export.ServingInputReceiver(
        {'input_1': images_tensor},  # This key should match your model's first layer. Try `my_model.input_names`
        {'image_bytes': input_ph})   # You can specify the key here, but this is a good default


if __name__ == "__main__":
    # Get a Keras model
    model_dir = op.join(os.environ['BUILDS_DIR'], 'ml-hv-grid', 'models_ec2')
    model = load_model(op.join(model_dir, '{}_arch.yaml'.format(model_start_time)),
                       op.join(model_dir, '{}_weights.h5'.format(model_start_time)))

    # Compile model (necessary for creating an estimator). However, no training will be done here
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model_save_fpath = op.join(model_dir,
                               '{}_complete_model.h5'.format(model_start_time))
    model.save(model_save_fpath)

    # Load using tf's method
    keras_model = tf.keras.models.load_model(model_save_fpath)
    # Create an Estimator object, and save to disk with preprocessing function
    estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model,
                                                      model_dir=op.join(model_dir, 'estimator'))

    # The below function will be renamed to `estimator.export_saved_model` in TF 2.0
    # TF adds a `keras` subdirectory (I'm not sure why), so need to copy the `checkpoint` file up one level to the `estimator` directory
    estimator.export_savedmodel(export_dir,
                                serving_input_receiver_fn=serving_input_receiver_fn)

'''
# To run the docker container with this model, try something on the command
# line like below. `tf-nightly` is good as it implements GET requests to get
#   information about the contained model (as of late 2018).

docker run -p 8501:8501 --mount type=bind,source=/Users/wronk/Builds/ml-hv-grid/models_hvgrid_export,target=/models/models_hvgrid_export -e MODEL_NAME=models_hvgrid_export -t tensorflow/serving:nightly


# The above will connect to an exported model on disk. You can also save the
# model to the container if you want to upload the docker image to docker hub.
# See the example here: https://www.tensorflow.org/serving/docker

###########################
# Copying from the example above:
docker run -d --name serving_base tensorflow/serving
docker cp models/<my model> serving_base:/models/<my model>
docker commit --change "ENV MODEL_NAME <my model>" serving_base developmentseed/<image_name>:<version_tag>
docker kill serving_base
docker run -p 8501:8501 -t developmentseed/<image_name>:<version tag>

###################################
# Same example with filled in data:
docker run -d --name serving_base tensorflow/serving
docker cp models_hvgrid_export serving_base:/models/hv_grid
docker commit --change "ENV MODEL_NAME hv_grid" serving_base developmentseed/hv_grid:v1
docker kill serving_base
docker run -p 8501:8501 -t developmentseed/hv_grid:v1

###########################################
# To push to DevSeed's account on dockerhub
docker push developmentseed/<container_name>:<container_tag>

# To run the image on port 8501
docker run -p 8501:8501 -t developmentseed/<image_name>:<version_tag>

# To run the GPU version
# See tensorflow's website for requirements:https://www.tensorflow.org/serving/docker#serving_with_docker_using_your_gpu

docker run --runtime=nvidia -p 8501:8501 -t developmentseed/<image_name>:<version_tag w/ `-gpu`>
'''
