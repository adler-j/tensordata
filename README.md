# TensorData

A package to easily download TensorFlow data without having to deal with folders, loading etc all the time.

# Installation

Simply run

    pip install https://github.com/adler-j/tensordata/archive/master.zip

# Examples

Get a batch of data from cifar10:

    import tensordata
    train, test = cifar10()
    batch_images, test_images = train.next_batch(5)

Or use proper TensorFlow input pipelines:

    import tensordata
    with tf.Session() as sess:
        images, labels = get_cifar10_tf()

        for i in range(100):
            image, label = sess.run([images, labels])