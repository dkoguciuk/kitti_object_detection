"""
TF records help methods.

Author: Charles R. Qi, Kui Xu
Modified by: Daniel Koguciuk
"""
from __future__ import print_function

import os
import argparse
import tensorflow as tf

import utils.viz_utils as viz_utils
import utils.tfrecord_utils as tfrecord_utils


def dataset_read_tfrecords(arguments):

    # Filenames
    tfrecord_filenames = [f for f in os.listdir(arguments.tfrecords_path) if '.tfrecord' in f]
    tfrecord_filenames.sort()
    tfrecord_filepaths = [os.path.join(args.tfrecords_path, f) for f in tfrecord_filenames]

    # TFdataset
    tfdataset = tf.data.TFRecordDataset(tfrecord_filepaths)
    tfdataset = tfdataset.map(tfrecord_utils.tfexample_to_point_cloud)

    # Iterator
    iterator = tfdataset.make_initializable_iterator()

    with tf.Session('') as sess:
        sess.run(iterator.initializer)
        data = iterator.get_next()
        x, y = sess.run(data)
        data = iterator.get_next()
        x, y = sess.run(data)

        viz_utils.draw_point_cloud_with_bboxes_as_a_list(x, y['3dbbox'], verbose=False)


if __name__ == '__main__':

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_path', type=str, help='output path where to put generated tfrecords')
    args = parser.parse_args()

    # Read tf records
    dataset_read_tfrecords(args)
