import random
import json
import collections
from PIL import Image
import settings as settings
import tensorflow as tf
import numpy as np
from preprocess import text_preprocess, image_preprocess


def deal_annotations(annotation_file, PATH):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_path_to_caption = collections.defaultdict(list)
    for ann in annotations['annotations']:
        caption = f"<start> {ann['caption']} <end>"
        img_path = PATH + '%012d.jpg' % (ann['image_id'])
        image_path_to_caption[img_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    train_image_paths = image_paths[:6000]

    # 6000 images -> 30000 captions

    train_captions = []
    img_path_vector = []
    for p in train_image_paths:
        captions = image_path_to_caption[p]
        train_captions.extend(captions)
        img_path_vector.extend([p] * len(captions))

    return train_captions, img_path_vector


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


def build_dataset(img_path_vector, caption_vector):
    dataset = tf.data.Dataset.from_tensor_slices(
        (img_path_vector, caption_vector))

    # Use map to load the numpy files in parallel
    # refer to keras tutorial
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle & Batch & Prefetch
    dataset = dataset.shuffle(settings.BUFFER_SIZE).batch(settings.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


class ImageCaptionDataset:
    def __init__(self, annotation_file_path, train_image_folder_path):
        self.annotation_file = annotation_file_path
        self.PATH = train_image_folder_path
        train_captions, img_path_vector = deal_annotations(
            self.annotation_file, self.PATH)
        image_preprocess(img_path_vector)
        caption_vector, self.max_length, self.tokenizer = text_preprocess(
            train_captions, vocab_size=settings.vocab_size)
        dataset = build_dataset(img_path_vector, caption_vector)
        self.dataset = dataset
        self.total_num = len(img_path_vector)
