from models import CNN_Encoder, RNN_Decoder
import tensorflow as tf
import time
import settings as settings
from dataset import ImageCaptionDataset
from detection import ObjectDetectionModel


def word_list_to_seq(word_list, tokenizer):
    for i, l in enumerate(word_list):
        for j, word in enumerate(l):
            word_list[i][j] = tokenizer.word_index[word]
    seq = tokenizer.fit_on_texts(word_list)
    seq = tokenizer.fit_on_sequences(seq)
    seq = tf.keras.preprocessing.sequence.pad_sequences(
        seq)  # (batch, max_object_num)
    return seq


class Trainer():
    def __init__(self, start_epoch, embedding_dim, units, vocab_size, tokenizer):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size, tokenizer)
        self.checkpoint_path = "./checkpoints/train"
        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                        decoder=self.decoder,
                                        optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_path, max_to_keep=5)

        self.start_epoch = start_epoch
        if self.ckpt_manager.latest_checkpoint:
            start_epoch = int(
                self.ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

        self.tokenizer = tokenizer

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target, word_list):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims(
            [self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)
            object_seq = word_list_to_seq(
                word_list, self.tokenizer)  # (batch_size, max_object_num)
            for i in range(1, target.shape[1]):
                predictions, hidden, _ = self.decoder(
                    dec_input, features, hidden, object_seq)
                loss += self.loss_function(target[:, i], predictions)
                dec_input = tf.expand_dims(
                    target[:, i], 1)  # 这个batch的第i个单词的index

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + \
            self.decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss


if __name__ == '__main__':

    dataset = ImageCaptionDataset(
        annotation_file_path=settings.annotation_file, train_image_folder_path=settings.PATH)
    start_epoch = 0

    trainer = Trainer(start_epoch=start_epoch, embedding_dim=256,
                      units=256, vocab_size=settings.vocab_size, tokenizer=dataset.tokenizer)

    detection_model = ObjectDetectionModel(settings.model_path)

    # start training
    loss_plot = []
    EPOCHS = 20
    num_steps = dataset.total_num // settings.BATCH_SIZE

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            # word_list: [[word1_from_img1, word2_from_img1], [word_from_img2, word2_from_img2], ...]
            word_list = detection_model(img_tensor)

            batch_loss, t_loss = trainer.train_step(
                img_tensor, target, word_list)
            total_loss += t_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

        # loss_plot.append(total_loss / num_steps)

        if (epoch + 1) % 5 == 0:
            trainer.ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
