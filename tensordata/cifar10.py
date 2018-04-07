"""Functions for downloading and reading CIFAR10 data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import ClassificationDataSet

import os
import urllib
import numpy as np
import tarfile
import pickle
from tensorflow.python.platform import gfile


__all__ = ('cifar10',)


SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'


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


def cifar10(train_dir='data/CIFAR10-data',
            test_size=5000):
    filename = 'cifar-10-python.tar.gz'

    maybe_download(filename, train_dir,
                   SOURCE_URL + filename)

    data = []
    labels = []
    for i in range(1, 6):
        path = os.path.join(train_dir, 'cifar-10-batches-py', 'data_batch_{}'.format(i))
        dct = unpickle(path)
        data.append(dct[b'data'])
        labels.append(np.array(dct[b'labels']))

    data_arr = np.concatenate(data, axis=0)
    raw_float = np.array(data_arr, dtype=float) / 255.0
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])

    labels = np.concatenate(labels, axis=0)

    assert 0 <= test_size <= len(data_arr)

    validation_images = images[:test_size]
    validation_labels = labels[:test_size]
    train_images = images[test_size:]
    train_labels = labels[test_size:]

    label_names = ['airplane',
                   'automobile',
                   'bird',
                   'cat',
                   'deer',
                   'dog',
                   'frog',
                   'horse',
                   'ship',
                   'truck']

    train = ClassificationDataSet(train_images,
                                  train_labels,
                                  label_names=label_names)
    test = ClassificationDataSet(validation_images,
                                 validation_labels,
                                 label_names=label_names)

    return train, test


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train, test = cifar10()

    batch_images, test_images = train.next_batch(5)

    for image, label in zip(batch_images, test_images):
        plt.figure()
        plt.title(train.label_names[label])
        plt.imshow(image)
