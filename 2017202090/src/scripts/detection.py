import os
import pathlib

import io
import scipy.misc
import numpy as np

import tensorflow_hub as hub
import settings as settings
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops


def convert_tensor_to_numpy_array(tensor):
    '''
    tensor shape: (batch, width, height, 3)
    '''
    im_width = tensor.shape[1]
    im_height = tensor.shape[2]
    return tensor.numpy().reshape((settings.BATCH_SIZE, im_height, im_width, 3)).astype(np.uint8)


class ObjectDetectionModel:
    def __init__(self, model_path, thresh=settings.thresh):
        self.hub_model = hub.load(model_path)
        self.thresh = thresh

    def __call__(self, image_tensor):
        image_np = convert_tensor_to_numpy_array(image_tensor)
        results = self.hub_model(image_np)
        result = {key: value.numpy() for key, value in results.items()}
        category_index = label_map_util.create_category_index_from_labelmap(
            settings.PATH_TO_LABELS, use_display_name=True)

        word_list = []
        for i, idx in enumerate(result['detection_classes'][0]):
            l = []
            if result['detection_scores'][0][i] > self.thresh:
                l.append(category_index[idx]['name'])
            word_list.append(l)
        return word_list
