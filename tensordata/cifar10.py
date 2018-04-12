"""Functions for downloading and reading CIFAR10 data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import numpy as np
import tarfile
import pickle
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensordata.augmentation import random_flip


__all__ = ('get_cifar10_dataset', 'get_cifar10_tf', 'CIFAR10_LABELS')


SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'


CIFAR10_LABELS = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']


class ClassificationDataSet(object):

    """Dataset for classification problems."""

    def __init__(self,
                 images,
                 labels):
        assert images.shape[0] == labels.shape[0]
        assert images.ndim == 4
        assert labels.ndim == 1

        self.num_examples = images.shape[0]

        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= self.num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.images[start:end], self.labels[start:end]


def maybe_download(filename, work_directory, source_url):
    if gfile.Exists(work_directory):
        return

    gfile.MakeDirs(work_directory)

    filepath = os.path.join(work_directory, filename)
    if not gfile.Exists(filepath):
        print('Downloading', filename)
        temp_file_name, _ = urllib.request.urlretrieve(source_url)
        gfile.Copy(temp_file_name, filepath)
        with gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')

        print("Extracting: " + filepath)
        base_path = os.path.dirname(filepath)
        with tarfile.open(filepath, "r:gz") as tf:
            tf.extractall(base_path)
        os.remove(filepath)
    return filepath


def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct


def get_cifar10_dataset(split=None):
    filename = 'cifar-10-python.tar.gz'
    train_dir = os.path.dirname(__file__) + '/data/CIFAR10-data'

    maybe_download(filename, train_dir,
                   SOURCE_URL + filename)

    data = []
    labels = []
    for i in range(1, 7):
        if i < 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'data_batch_{}'.format(i))
        if i < 6:
            path = os.path.join(train_dir, 'cifar-10-batches-py', 'test_batch')
        dct = unpickle(path)
        data.append(dct[b'data'])
        labels.append(np.array(dct[b'labels']))

    data_arr = np.concatenate(data, axis=0)
    raw_float = np.array(data_arr, dtype=float) / 255.0
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])

    labels = np.concatenate(labels, axis=0)

    if split is None:
        pass
    elif split == 'train':
        images = images[:-10000]
        labels = labels[:-10000]
    elif split == 'test':
        images = images[-10000:]
        labels = labels[-10000:]
    else:
        raise ValueError('unknown split')

    dataset = ClassificationDataSet(images,
                                    labels)

    return dataset


def get_cifar10_tf(batch_size=2, shape=[32, 32], split=None, augment=True):
    with tf.name_scope('get_cifar10_tf'):
        dataset = get_cifar10_dataset(split=split)

        images = tf.constant(dataset.images, dtype='float32')
        labels = tf.constant(dataset.labels, dtype='int32')

        images_batch, labels_batch = tf.train.shuffle_batch(
            [images, labels],
            batch_size=batch_size,
            num_threads=8,
            capacity=10 * batch_size,
            min_after_dequeue=3 * batch_size,
            enqueue_many=True)

        if augment:
            images_batch = random_flip(images_batch)

        if shape != [32, 32]:
            images_batch = tf.image.resize_bilinear(images_batch,
                                                    [shape[0], shape[1]])

        return tf.to_float(images_batch), labels_batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Simple dataset
    dataset = get_cifar10_dataset()

    batch_images, test_images = dataset.next_batch(5)

    for image, label in zip(batch_images, test_images):
        plt.figure()
        plt.title(CIFAR10_LABELS[label])
        plt.imshow(image)

    # Pure tensorflow
    # Start a new session to show example output.
    with tf.Session() as sess:
        images, labels = get_cifar10_tf()

        # Required to get the filename matching to run.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            # Get an image tensor and print its value.
            image, label = sess.run([images, labels])
            print(image.shape, label)
