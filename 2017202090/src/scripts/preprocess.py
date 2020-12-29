import tensorflow as tf
import numpy as np
from tqdm import tqdm
from settings import vocab_size

# text preprocess


def text_preprocess(train_captions, vocab_size=vocab_size):
    vocab_size = 5000 + 1
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size-1,
                                                      oov_token="<unk>",
                                                      filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # padding
    caption_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    max_length = max(len(seq) for seq in train_seqs)

    return caption_vector, max_length, tokenizer


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (512, 512))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# image proprecess


def image_preprocess(img_path_vector):
    inception_v3 = tf.keras.applications.InceptionV3(include_top=False)
    new_input = inception_v3.input
    hidden_layer = inception_v3.layers[-1].output

    model = tf.keras.Model(new_input, hidden_layer)
    all_images = sorted(set(img_path_vector))
    image_data = tf.data.Dataset.from_tensor_slices(all_images)
    image_data = image_data.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for imgs, paths in tqdm(image_data):
        features = model(imgs)
        features = tf.reshape(
            features, (features.shape[0], -1, features.shape[3]))

        for feature, path in zip(features, paths):
            p = path.numpy().decode('utf-8')
            np.save(p, feature.numpy())
