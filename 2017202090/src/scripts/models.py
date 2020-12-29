import tensorflow as tf
import numpy as np
from knowledge import get_knowledge


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # convert input size to embedding_dim
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        '''
          x shape: (batch_size, 64, 2048)
        '''
        x = self.fc(x)  # (batch_size, 64, embedding_dim)
        x = tf.nn.relu(x)  # (batch_size, 64, embedding_dim)
        return x


class SentinelBlock(tf.keras.Model):
    def __init__(self, units):
        super(SentinelBlock, self).__init__()
        self.units = units
        self.weight_x = tf.keras.layers.Dense(units)
        self.weight_h = tf.keras.layers.Dense(units)

    def call(self, x, hidden, memory_cell):
        '''
          x shape: (batch_size, 1, embedding_dim)
          hidden shape: (batch_size, 1, units)
          memory_cell shape: (batch_size, units)
        '''

        hidden = tf.reduce_sum(hidden, axis=1)
        x = tf.reduce_sum(x, axis=1)

        gate_t = (tf.nn.sigmoid(self.weight_x(x) +
                                self.weight_h(hidden)))  # (batch_size, units)

        state_t = gate_t * tf.nn.tanh(memory_cell)  # (batch_size, units)

        return state_t


class Attention(tf.keras.Model):
    # Adaptive Attention: visual sentinel
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W_v = tf.keras.layers.Dense(units)
        self.W_g = tf.keras.layers.Dense(units)
        self.W_s = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden, sentinel):
        '''
          features shape: (batch_size, 64, embedding_dim)
          hidden shape: (batch_size, units)
          sentinel shape: (batch_size, units)
        '''

        sentinel = tf.expand_dims(sentinel, axis=1)  # (batch_size, 1, units)

        attention_hidden_layer = (tf.nn.tanh(self.W_v(features) +
                                             self.W_g(hidden)))  # (batch_size, 64, units)

        score = self.V(attention_hidden_layer)  # (batch_size, 64, 1)

        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, 64, 1)

        context_vector = attention_weights * \
            features  # (batch_size, 1, embedding)
        context_vector = tf.reduce_sum(
            context_vector, axis=1)  # (batch_size, embedding)

        # sentinel beta_t calculaltion
        # (batch_size, 1, units)
        score0 = self.V(tf.nn.tanh(self.W_s(sentinel) + self.W_g(hidden)))

        attention_weights_new = tf.nn.softmax(
            tf.concat([score0, score], axis=1), axis=1)  # (batch_size, 65, 1)
        beta_t = tf.reduce_sum(attention_weights_new, axis=-1)[:, -1]

        return context_vector, attention_weights, beta_t


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, tokenizer):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(self.units)

        self.sentinel = SentinelBlock(self.units)
        self.tokenizer = tokenizer

    def call(self, words, features, hidden, object_seq):
        '''
          words shape: (batch_size, 1)
          features shape: (batch_size, 64, embedding_dim)
          object_seq: (batch_size, max_object_num)
        '''

        # apply attention later
        # context_vector, attention_weights = self.attention(features, hidden)

        # get the word embedding presentation
        x = self.embedding(words)  # ( batch_size, 1, embedding_dim)

        # integrating knowledge
        # (batch_size, max_object_num, embedding_dim)
        object_embedding = self.embedding(object_seq)
        similar_object_index = self.get_similar_object(
            object_embedding, x)  # (batch, 1, embedding_dim)

        external_know = np.zeros(x.shape[0], 1)
        for i, obj in enumerate(similar_object_index):
            obj = self.tokenizer.index_word[object_seq[i, obj]]
            scene, _ = get_knowledge(obj)
            if(len(scene_info) != 0):
                    # no external knowledge
                external_know[i] = self.tokenizer.word_index[scene[0]]
        external_know = self.embedding(
            external_know)  # (batch, 1, embedding_dim)
        external_know = tf.reshape(
            external_know, (-1, external_know.shape[-1]))

        # apply GRU
        # output: (batch_size, 1, units) state: batch_size, units
        output, state = self.gru(x)

        # apply sentinel
        s_t = self.sentinel(x, output, state)  # (batch_size, units)

        # apply attention
        context_vector, attention_weights, beta_t = self.attention(
            features, output, s_t)  # context_vector: (batch_size, units)

        # calculate c_hat_t
        beta_t = tf.expand_dims(beta_t, axis=-1)
        context_vector_new = beta_t * s_t + \
            (1 - beta_t) * context_vector  # (batch_size, units)

        # alignment
        x = tf.concat([tf.expand_dims(context_vector_new, 1),
                       output], axis=-1)  # (batch_size, 1, units)

        x = self.fc1(output)  # (batch_size, max_length, units)
        x = tf.reshape(x, (-1, x.shape[2]))  # (batch_size * max_length, units)
        x = self.fc2(tf.concate([x, external_know], axis=-1))

        return x, output, attention_weights

    def get_similar_object(self, object_embedding, x):
        '''
        object_embedding: (batch, max_object_num, embedding)
        x: (batch, 1, embedding)
        '''
        loss = tf.keras.losses.cosine_similarity(object_embedding, x, axis=-1)
        max_indice = np.argmax(loss, axis=-1)  # (batch, 1, embedding)
        return max_indice

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


if __name__ == '__main__':
    '''
    test 
    '''
    encoder = CNN_Encoder(embedding_dim=256)
    decoder = RNN_Decoder(embedding_dim=256, units=256, vocab_size=2048)

    batch_size = 4

    x = np.ones([batch_size, 64, 2048])
    dec_input = tf.expand_dims([0] * batch_size, 1)

    features = encoder(x)
    hidden = decoder.reset_state(batch_size=4)
    output, hidden, _ = decoder(dec_input, features, hidden)
