import tensorflow as tf

if __name__ == '__main__':
    import tensordata

    cifar_images, _ = tensordata.get_cifar10_tf(batch_size=1000, split='train')
    cifar_images = tf.image.resize_bilinear(cifar_images, [299, 299])
    cifar_images = (cifar_images - 0.5) * 2
    cifar_inception_score = tf.contrib.gan.eval.inception_score(cifar_images, num_batches=1)

    """
    cifar_images2, _ = tensordata.get_cifar10_tf(batch_size=1000, split='test')
    cifar_images2 = tf.image.resize_bilinear(cifar_images2, [299, 299])
    cifar_images2 = (cifar_images2 - 0.5) * 2
    cifar_frechet_score = tf.contrib.gan.eval.frechet_inception_distance(cifar_images, cifar_images2,
                                                                         num_batches=10)
    """

    # Pure tensorflow
    # Start a new session to show example output.
    with tf.Session() as sess:
        # Required to get the filename matching to run.
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):
            # Get an image tensor and print its value.
            images, inception_score = sess.run(
                    [cifar_images, cifar_inception_score])
            print(images.min(), images.max(), images.shape, inception_score)
