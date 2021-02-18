import tensorflow as tf
import math

class ResnetTrainable():
    def __init__(self, logger, optimizer, model, train_dataset, test_dataset, epochs):
        self.logger = logger
        self.optimizer = optimizer
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # For logging metrics
        self.train_loss = tf.keras.metrics.Mean()
        self.train_acc = tf.keras.metrics.Mean()
        self.test_loss = tf.keras.metrics.Mean()
        self.test_acc = tf.keras.metrics.Mean()
        self.test_best_acc = -math.inf

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(data)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.argmax(predictions, axis=-1) == tf.cast(tf.squeeze(labels), tf.int64), tf.float32))
        self.train_acc.update_state(accuracy)

    @tf.function
    def test_step(self, data, labels):
        predictions = self.model(data)
        loss = self.loss_object(labels, predictions)

        self.test_loss.update_state(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.argmax(predictions, axis=-1) == tf.cast(tf.squeeze(labels), tf.int64), tf.float32))
        self.test_acc.update_state(accuracy)

    def run(self):
        for epoch in range(self.epochs):
            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.test_loss.reset_states()
            self.test_acc.reset_states()

            for index, (data, labels) in enumerate(self.train_dataset):
                self.train_step(data, labels)
                print(f'Epoch {epoch}/{self.epochs}: Train {index}/{len(self.train_dataset)} | '
                  f'Loss: {self.train_loss.result():.5f} | '
                  f'Acc: {self.train_acc.result():.5f}')

            for index, (data, labels) in enumerate(self.test_dataset):
                self.test_step(data, labels)
                print(f'Epoch {epoch}/{self.epochs}: Train {index}/{len(self.train_dataset)} | '
                      f'Loss: {self.test_loss.result():.5f} | '
                      f'Acc: {self.test_acc.result():.5f}')

            if self.test_acc.result() > self.test_best_acc:
                self.test_best_acc = self.test_acc.result()
                self.logger.save_state()

            """ ------------------- Logging Stuff --------------------------"""
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('Train-Loss', self.train_loss.result())
            self.logger.log_tabular('Test-Loss', self.test_loss.result())
            self.logger.log_tabular('Train-Accuracy', self.train_acc.result())
            self.logger.log_tabular('Test-Accuracy', self.test_acc.result())
            self.logger.dump_tabular()

            self.logger.tf_board_save_scaler(epoch, self.train_loss.result(), 'loss', 'train')
            self.logger.tf_board_save_scaler(epoch, self.test_loss.result(), 'loss', 'test')
            self.logger.tf_board_save_scaler(epoch, self.train_acc.result(), 'accuracy', 'train')
            self.logger.tf_board_save_scaler(epoch, self.test_acc.result(), 'accuracy', 'test')
