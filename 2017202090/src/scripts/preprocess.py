import tensorflow as tf 

# hyper-parameters YAML
vocab_size = 5000 + 1
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size-1, 
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)

# TODO: