"""Functions for downloading and reading CIFAR10 data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


__all__ = ('get_mnist_tf',)


def get_mnist_tf(batch_size=1, shape=[28, 28], split=None,
                 start_queue_runner=True):
    with tf.name_scope('get_mnist_tf'):
        mnist = input_data.read_data_sets('MNIST_data')

        if split == 'train':
            images = mnist.train.images
            labels = mnist.train.labels
        elif split == 'train':
            images = mnist.test.images
            labels = mnist.test.labels
        elif split == None:
            images = np.concatenate([mnist.train.images, mnist.test.images],
                                    axis=0)
            labels = np.concatenate([mnist.train.labels, mnist.test.labels],
                                    axis=0)
        images = np.reshape(images, [-1, 28, 28, 1])

        images = tf.constant(images, dtype='float32')
        labels = tf.constant(labels, dtype='int32')

        image, label = tf.train.slice_input_producer([images, labels],
                                                     shuffle=True)

        images_batch, labels_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            capacity=5000 + 3*batch_size,
            num_threads=8)

        if shape != [28, 28]:
            images_batch = tf.image.resize_bilinear(images_batch,
                                                    [shape[0], shape[1]])

        return images_batch, labels_batch

if __name__ == '__main__':
    with tf.Session() as sess:
        images, labels = get_mnist_tf()

        # Required to get the filename matching to run.
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            # Get an image tensor and print its value.
            image_tensor, label_tensor = sess.run([images, labels])
            print(image_tensor.shape, labels.shape)


    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)