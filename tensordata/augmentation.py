import tensorflow as tf

def random_flip(x, up_down=False, left_right=True):
    """Randomly flip a batch of images image.

    Parameters
    ----------
    x : tf.Tensor with shape (Batch, nx, ny, channels)
        Images to flip.
    up_down : boolean
        If the images should be flipped in the up/down direction (axis 1).
    up_down : boolean
        If the images should be flipped in the left/right direction (axis 2).
    """
    with tf.name_scope('random_flip'):
        s = tf.shape(x)
        if up_down:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[1]))
        if left_right:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[2]))
        return x