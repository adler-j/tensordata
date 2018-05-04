"""Loaders for CelebA dataset."""

from __future__ import print_function
import os
import shutil
import zipfile
import requests
from tqdm import tqdm
import tensorflow as tf
from tensordata.augmentation import random_flip

__all__ = ('get_celeba_tf',)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                          unit='B', unit_scale=32*1024, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip(filepath):
    print("Extracting: " + filepath)
    base_path = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(base_path)
    os.remove(filepath)

def download_celeb_a(base_path):
    data_path = os.path.join(base_path, 'CelebA')
    images_path = os.path.join(data_path, 'images')
    if os.path.exists(data_path):
        print('[!] Found Celeb-A - skip')
        return

    filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(base_path, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    with zipfile.ZipFile(save_path) as zf:
        members = zf.namelist()
        for zipinfo in tqdm(members, 'unzipping'):
            zf.extract(zipinfo, base_path)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    os.rename(os.path.join(base_path, "img_align_celeba"), images_path)
    os.remove(save_path)


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_copy(in_dir, basename, out_dir):
    in_file = os.path.join(in_dir, basename)
    if os.path.exists(in_file):
        out_file = os.path.join(out_dir, basename)
        shutil.copyfile(in_file, out_file)


def add_splits(base_path):
    data_path = os.path.join(base_path, 'CelebA')
    images_path = os.path.join(data_path, 'images')
    train_dir = os.path.join(data_path, 'splits', 'train')
    valid_dir = os.path.join(data_path, 'splits', 'valid')
    test_dir = os.path.join(data_path, 'splits', 'test')
    make_directory(train_dir)
    make_directory(valid_dir)
    make_directory(test_dir)

    # these constants based on the standard CelebA splits
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637

    for i in tqdm(range(0, TRAIN_STOP), 'creating train data'):
        basename = "{:06d}.jpg".format(i+1)
        create_copy(images_path, basename, train_dir)
    for i in tqdm(range(TRAIN_STOP, VALID_STOP), 'creating test data'):
        basename = "{:06d}.jpg".format(i+1)
        create_copy(images_path, basename, valid_dir)
    for i in tqdm(range(VALID_STOP, NUM_EXAMPLES), 'creating validation data'):
        basename = "{:06d}.jpg".format(i+1)
        create_copy(images_path, basename, test_dir)


def maybe_download_celeba():
    base_path = os.path.join(os.path.dirname(__file__), 'data')
    marker_file = os.path.join(base_path, "CELEB_A_READY")
    if os.path.exists(marker_file):
        return  # data downloaded

    download_celeb_a(base_path)
    add_splits(base_path)

    with open(marker_file, "a+"):
        pass # Create marker file


def get_celeba_tf(batch_size=1, shape=[64, 64], split=None, augment=True):
    with tf.name_scope('get_celeba'):
        maybe_download_celeba()

        base_path = os.path.join(os.path.dirname(__file__), 'data', 'CelebA')

        if split is None:
            filename_queue = tf.train.string_input_producer(
                    tf.train.match_filenames_once(base_path + "/images/*.jpg"))
        elif split == 'train':
            filename_queue = tf.train.string_input_producer(
                    tf.train.match_filenames_once(base_path + "/splits/train/*.jpg"))
        elif split == 'test':
            filename_queue = tf.train.string_input_producer(
                    tf.train.match_filenames_once(base_path + "/splits/test/*.jpg"))
        else:
            raise ValueError('unknown split')

        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_jpeg(image_file)
        image.set_shape((218, 178, 3))

        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * batch_size

        images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=8,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)

        images = tf.to_float(images)
        images = images / 256.0

        if augment:
            images = random_flip(images)
            images += tf.random_uniform(tf.shape(images),
                                        0.0, 1.0/256.0)

        images = tf.image.crop_to_bounding_box(images, 50, 25, 128, 128)
        images = tf.image.resize_bilinear(images, [shape[0], shape[1]])

        return images


if __name__ == '__main__':
    # Start a new session to show example output.
    with tf.Session() as sess:
        images = get_celeba_tf()

        # Required to get the filename matching to run.
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            # Get an image tensor and print its value.
            image_tensor = sess.run(images)
            print(image_tensor.shape)


    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)