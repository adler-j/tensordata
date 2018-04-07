# TensorData

A package to easily download tensorflow data without having to deal with
folders, loading etc all the time.

# Installation

Simply run

    pip install https://github.com/adler-j/tensordata/archive/master.zip

# Examples

Get a batch of data from cifar10:

    >>> import tensordata
    >>> train, test = cifar10()
    >>> batch_images, test_images = train.next_batch(5)
